#include "theia/renderer/GpuCullPass.hpp"

#include <array>
#include <cstdint>
#include <slang-math/slang-math.hpp>
#include <utility>
#include <vector>

#include "harmonia/core/Logger.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "theia/renderer/RendererConstants.hpp"
#include "theia/renderer/ShaderPath.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

// Push constant block matching forward_cull.comp.slang CullPC (80 bytes).
struct alignas(4) CullPC {
    sm::float4x4 viewProj;                         // 64 bytes — row-major VP matrix
    std::uint32_t instanceCount{};                 // 4 bytes
    std::uint32_t _pad0 = 0, _pad1 = 0, _pad2 = 0; // 12 bytes pad
};
static_assert(sizeof(CullPC) == 80);

GpuCullPass::~GpuCullPass() {
    shutdown();
}

GpuCullPass::GpuCullPass(GpuCullPass&& o) noexcept
    : m_ctx(o.m_ctx),
      m_pipeline(std::exchange(o.m_pipeline, VK_NULL_HANDLE)),
      m_pipelineLayout(std::exchange(o.m_pipelineLayout, VK_NULL_HANDLE)),
      m_setLayout(std::exchange(o.m_setLayout, VK_NULL_HANDLE)),
      m_pool(std::exchange(o.m_pool, VK_NULL_HANDLE)),
      m_set(std::exchange(o.m_set, VK_NULL_HANDLE)),
      m_compactInstanceListBuf(std::move(o.m_compactInstanceListBuf)),
      m_indirectDrawBuf(std::move(o.m_indirectDrawBuf)) {
    o.m_ctx = nullptr;
}

GpuCullPass& GpuCullPass::operator=(GpuCullPass&& o) noexcept {
    if (this != &o) {
        shutdown();
        m_ctx = o.m_ctx;
        m_pipeline = std::exchange(o.m_pipeline, VK_NULL_HANDLE);
        m_pipelineLayout = std::exchange(o.m_pipelineLayout, VK_NULL_HANDLE);
        m_setLayout = std::exchange(o.m_setLayout, VK_NULL_HANDLE);
        m_pool = std::exchange(o.m_pool, VK_NULL_HANDLE);
        m_set = std::exchange(o.m_set, VK_NULL_HANDLE);
        m_compactInstanceListBuf = std::move(o.m_compactInstanceListBuf);
        m_indirectDrawBuf = std::move(o.m_indirectDrawBuf);
        o.m_ctx = nullptr;
    }
    return *this;
}

bool GpuCullPass::initialize(const DeviceContext& ctx, const char* spvFilename) {
    shutdown();
    m_ctx = &ctx;

    // --- Output buffers ---
    const VkDeviceSize kCompactListSize = static_cast<VkDeviceSize>(kMaxInstances) * sizeof(std::uint32_t);
    // vkCmdFillBuffer zeros all 12 bytes before each dispatch; the compute shader then
    // restores groupCountY=1 and groupCountZ=1, and accumulates groupCountX atomically.
    constexpr VkDeviceSize kIndirectBufSize = kMeshTaskIndirectCmdSize;

    auto compactBuf = Buffer::create(ctx,
                                     kCompactListSize,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                     "theia.gpuCull.compactInstanceList");
    auto indirectBuf = Buffer::create(ctx,
                                      kIndirectBufSize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      "theia.gpuCull.indirectDrawBuf");

    if (!compactBuf || !indirectBuf) {
        Logger::error("GpuCullPass: failed to allocate output buffers");
        return false;
    }
    m_compactInstanceListBuf = std::move(*compactBuf);
    m_indirectDrawBuf = std::move(*indirectBuf);

    // --- Descriptor set layout ---
    // binding 0: instances          (STORAGE, read)
    // binding 1: instanceBounds     (STORAGE, read)
    // binding 2: compactInstanceList (STORAGE, write)
    // binding 3: indirectDrawBuf     (STORAGE, write — RWByteAddressBuffer for atomic groupCountX)
    const std::array<VkDescriptorSetLayoutBinding, 4> bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    constexpr VkDescriptorBindingFlags kUAB = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    constexpr std::array<VkDescriptorBindingFlags, 4> bindingFlags{kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &flagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &setLayoutInfo, nullptr, &m_setLayout) != VK_SUCCESS) {
        Logger::error("GpuCullPass: failed to create descriptor set layout");
        return false;
    }

    // --- Pipeline layout ---
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = static_cast<std::uint32_t>(sizeof(CullPC)),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_setLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    if (vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        Logger::error("GpuCullPass: failed to create pipeline layout");
        return false;
    }

    // --- Compute pipeline ---
    auto spirv = harmonia::readSpirv(shaderPath(spvFilename));
    if (!spirv) {
        Logger::error("GpuCullPass: failed to read SPIR-V from '{}'", spvFilename);
        return false;
    }
    const VkShaderModuleCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv->size() * sizeof(std::uint32_t),
        .pCode = spirv->data(),
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        Logger::error("GpuCullPass: failed to create shader module");
        return false;
    }
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shaderModule,
                .pName = "main",
            },
        .layout = m_pipelineLayout,
    };
    const VkResult pipeResult =
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);
    vkDestroyShaderModule(ctx.device, shaderModule, nullptr);
    if (pipeResult != VK_SUCCESS) {
        Logger::error("GpuCullPass: failed to create compute pipeline");
        return false;
    }

    // --- Descriptor pool + set ---
    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &m_pool) != VK_SUCCESS) {
        Logger::error("GpuCullPass: failed to create descriptor pool");
        return false;
    }
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_setLayout,
    };
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &m_set) != VK_SUCCESS) {
        Logger::error("GpuCullPass: failed to allocate descriptor set");
        return false;
    }

    // Pre-write static output bindings (2–3 never change after initialization).
    const VkDescriptorBufferInfo compactInfo{m_compactInstanceListBuf.handle(), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo indirectInfo{m_indirectDrawBuf.handle(), 0, VK_WHOLE_SIZE};
    const std::array<VkWriteDescriptorSet, 2> staticWrites{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set,
                             2,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &compactInfo,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set,
                             3,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &indirectInfo,
                             nullptr},
    };
    vkUpdateDescriptorSets(
        ctx.device, static_cast<std::uint32_t>(staticWrites.size()), staticWrites.data(), 0, nullptr);

    Logger::info("GpuCullPass: initialized (max {} instances)", kMaxInstances);
    return true;
}

void GpuCullPass::shutdown() {
    if (!m_ctx)
        return;

    m_compactInstanceListBuf = {};
    m_indirectDrawBuf = {};

    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_ctx->device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
        m_set = VK_NULL_HANDLE;
    }
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx->device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_ctx->device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_setLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_setLayout, nullptr);
        m_setLayout = VK_NULL_HANDLE;
    }
    m_ctx = nullptr;
}

void GpuCullPass::dispatch(VkCommandBuffer cmd,
                           VkBuffer instanceBuf,
                           VkBuffer instanceBoundsBuf,
                           std::uint32_t instanceCount,
                           const sm::float4x4& viewProj) {
    if (!m_ctx || m_pipeline == VK_NULL_HANDLE || instanceCount == 0)
        return;

    // Zero all 12 bytes so thread 0 can re-initialize groupCountY=1 and groupCountZ=1,
    // and all visible-instance threads accumulate groupCountX from 0.
    vkCmdFillBuffer(cmd, m_indirectDrawBuf.handle(), 0, VK_WHOLE_SIZE, 0u);

    // Barrier: fill → compute read/write on indirectDrawBuf.
    const VkBufferMemoryBarrier2 fillBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .buffer = m_indirectDrawBuf.handle(),
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    const VkDependencyInfo fillDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &fillBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &fillDep);

    // Update scene-bound input descriptors (bindings 0–1 may change if scene switches).
    const VkDescriptorBufferInfo instInfo{instanceBuf, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo boundsInfo{instanceBoundsBuf, 0, VK_WHOLE_SIZE};
    const std::array<VkWriteDescriptorSet, 2> dynWrites{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set,
                             0,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &instInfo,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set,
                             1,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &boundsInfo,
                             nullptr},
    };
    vkUpdateDescriptorSets(m_ctx->device, static_cast<std::uint32_t>(dynWrites.size()), dynWrites.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    const CullPC pc{viewProj, instanceCount};
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPC), &pc);

    const std::uint32_t groups = (instanceCount + 63u) / 64u;
    vkCmdDispatch(cmd, groups, 1, 1);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
