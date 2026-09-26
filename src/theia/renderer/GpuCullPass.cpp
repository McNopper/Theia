#include "theia/renderer/GpuCullPass.hpp"

#include <array>
#include <cstdint>
#include <cstring>
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

bool GpuCullPass::initialize(const harmonia::DeviceContext& ctx, const char* spvFilename) {
    shutdown();
    m_ctx = &ctx;

    // --- Output buffers ---
    const VkDeviceSize kCompactListSize = static_cast<VkDeviceSize>(kMaxInstances) * sizeof(std::uint32_t);
    // vkCmdFillBuffer zeros all 12 bytes before each dispatch; the compute shader then
    // restores groupCountY=1 and groupCountZ=1, and accumulates groupCountX atomically.
    constexpr VkDeviceSize kIndirectBufSize = kMeshTaskIndirectCmdSize;

    auto compactBuf = harmonia::Buffer::create(ctx,
                                               kCompactListSize,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                               "theia.gpuCull.compactInstanceList");
    auto indirectBuf =
        harmonia::Buffer::create(ctx,
                                 kIndirectBufSize,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                 "theia.gpuCull.indirectDrawBuf");

    if (!compactBuf || !indirectBuf) {
        harmonia::Logger::error("GpuCullPass: failed to allocate output buffers");
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
    // MOD1: no UPDATE_AFTER_BIND — incompatible with DESCRIPTOR_BUFFER layouts.
    // The descriptor buffer is host-visible and written directly (no update-after-bind
    // semantics needed).
    constexpr std::array<VkDescriptorBindingFlags, 4> bindingFlags{0, 0, 0, 0};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &flagsInfo,
        // MOD1: descriptor buffer path — this layout is used with a GPU descriptor
        // buffer (vkCmdBindDescriptorBuffersEXT), not with a descriptor pool/set.
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    VkDescriptorSetLayout setLayout{};
    if (vkCreateDescriptorSetLayout(ctx.device, &setLayoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
        harmonia::Logger::error("GpuCullPass: failed to create descriptor set layout");
        return false;
    }
    m_setLayout = harmonia::UniqueDescriptorSetLayout{ctx.device, setLayout};

    // --- harmonia::Pipeline layout ---
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = static_cast<std::uint32_t>(sizeof(CullPC)),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_setLayout.ptr(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    VkPipelineLayout pipelineLayout{};
    if (vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        harmonia::Logger::error("GpuCullPass: failed to create pipeline layout");
        return false;
    }
    m_pipelineLayout = harmonia::UniquePipelineLayout{ctx.device, pipelineLayout};

    // --- Compute pipeline ---
    auto spirv = harmonia::readSpirv(shaderPath(spvFilename));
    if (!spirv) {
        harmonia::Logger::error("GpuCullPass: failed to read SPIR-V from '{}'", spvFilename);
        return false;
    }
    const VkShaderModuleCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv->size() * sizeof(std::uint32_t),
        .pCode = spirv->data(),
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        harmonia::Logger::error("GpuCullPass: failed to create shader module");
        return false;
    }
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        // MOD1: this pipeline uses descriptor buffers (not descriptor sets).
        .flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
        .stage =
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shaderModule,
                .pName = "main",
            },
        .layout = m_pipelineLayout,
    };
    VkPipeline pipeline{};
    const VkResult pipeResult =
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(ctx.device, shaderModule, nullptr);
    if (pipeResult != VK_SUCCESS) {
        harmonia::Logger::error("GpuCullPass: failed to create compute pipeline");
        return false;
    }
    m_pipeline = harmonia::UniquePipeline{ctx.device, pipeline};

    // --- MOD1: descriptor buffer (replaces the pool + set) ---
    if (!m_descWriter.init(ctx, setLayout, 4, "theia.cull.descriptorBuffer")) {
        return false;
    }

    // Pre-write static output bindings (2–3 never change after initialization).
    m_descWriter.writeStorageBuffer(ctx, 2, m_compactInstanceListBuf.deviceAddress(), kCompactListSize);
    m_descWriter.writeStorageBuffer(ctx, 3, m_indirectDrawBuf.deviceAddress(), kIndirectBufSize);

    harmonia::Logger::info("GpuCullPass: initialized (max {} instances)", kMaxInstances);
    return true;
}

void GpuCullPass::shutdown() {
    m_compactInstanceListBuf = {};
    m_indirectDrawBuf = {};
    m_descWriter.shutdown(); // MOD1: replaces pool + set

    m_pipeline.reset();
    m_pipelineLayout.reset();
    m_setLayout.reset();
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

    // MOD1: update scene-bound input descriptors (bindings 0–1) via the descriptor buffer writer.
    {
        const VkBufferDeviceAddressInfo bufAddrInfo0{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = instanceBuf};
        VkMemoryRequirements memReq0{};
        vkGetBufferMemoryRequirements(m_ctx->device, instanceBuf, &memReq0);
        m_descWriter.writeStorageBuffer(*m_ctx, 0, vkGetBufferDeviceAddress(m_ctx->device, &bufAddrInfo0),
                                        memReq0.size);

        const VkBufferDeviceAddressInfo bufAddrInfo1{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = instanceBoundsBuf};
        VkMemoryRequirements memReq1{};
        vkGetBufferMemoryRequirements(m_ctx->device, instanceBoundsBuf, &memReq1);
        m_descWriter.writeStorageBuffer(*m_ctx, 1, vkGetBufferDeviceAddress(m_ctx->device, &bufAddrInfo1),
                                        memReq1.size);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    // MOD1: bind the descriptor buffer (replaces vkCmdBindDescriptorSets).
    m_descWriter.bind(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout);

    const CullPC pc{viewProj, instanceCount};
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPC), &pc);

    const std::uint32_t groups = (instanceCount + 63u) / 64u;
    vkCmdDispatch(cmd, groups, 1, 1);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
