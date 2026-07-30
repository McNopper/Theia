#include "theia/renderer/LightCuller.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <slang-math/slang-math.hpp>
#include <utility>
#include <vector>

#include "harmonia/core/Logger.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "theia/renderer/ShaderPath.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

namespace {

// Push constant layout must match LightCullPC in light_cull.comp.slang:
//   proj (64), view (64), tilesXY (8), screenSize (8), lightCount (4), nearZ (4), farZ (4), _pad (4) = 160 bytes
struct LightCullPC {
    sm::float4x4 proj;
    sm::float4x4 view;
    sm::uint2 tilesXY;
    sm::uint2 screenSize;
    std::uint32_t lightCount{};
    float nearZ{};
    float farZ{};
    std::uint32_t _pad{};
};
static_assert(sizeof(LightCullPC) == 160);

[[nodiscard]] std::vector<std::uint32_t> readSpirv(const char* filename) {
    auto code = harmonia::readSpirv(shaderPath(filename));
    if (!code)
        return {};
    return std::move(*code);
}

} // namespace

LightCuller::~LightCuller() {
    shutdown();
}

bool LightCuller::initialize(const harmonia::DeviceContext& ctx, std::uint32_t w, std::uint32_t h, const char* spvFilename) {
    shutdown();
    m_ctx = &ctx;
    m_screenWidth = w;
    m_screenHeight = h;
    m_tilesX = (w + kTileSize - 1) / kTileSize;
    m_tilesY = (h + kTileSize - 1) / kTileSize;

    const VkDeviceSize countsBufSize = static_cast<VkDeviceSize>(m_tilesX * m_tilesY) * sizeof(std::uint32_t);
    const VkDeviceSize indicesBufSize =
        static_cast<VkDeviceSize>(m_tilesX * m_tilesY) * kMaxLightsPerTile * sizeof(std::uint32_t);

    auto counts = harmonia::Buffer::create(ctx,
                                 countsBufSize,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                 "theia.tileLightCounts");
    auto indices = harmonia::Buffer::create(ctx,
                                  indicesBufSize,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  "theia.tileLightIndices");
    if (!counts || !indices) {
        harmonia::Logger::error("LightCuller: failed to allocate tile buffers");
        return false;
    }
    m_tileLightCountsBuf = std::move(*counts);
    m_tileLightIndicesBuf = std::move(*indices);

    // Descriptor set layout: binding 0=lights (read), 1=counts (RW), 2=indices (RW)
    const std::array<VkDescriptorSetLayoutBinding, 3> bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    constexpr VkDescriptorBindingFlags kUAB = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    constexpr std::array<VkDescriptorBindingFlags, 3> bindingFlags{kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &bindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    VkDescriptorSetLayout setLayout{};
    if (vkCreateDescriptorSetLayout(ctx.device, &setLayoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
        harmonia::Logger::error("LightCuller: failed to create set layout");
        return false;
    }
    m_setLayout = harmonia::UniqueDescriptorSetLayout{ctx.device, setLayout};

    // Push constant: LightCullPC struct in shader (4×4 + 4×4 + 2 + 2 + 3 + 1 uints + floats)
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(LightCullPC),
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
        harmonia::Logger::error("LightCuller: failed to create pipeline layout");
        return false;
    }
    m_pipelineLayout = harmonia::UniquePipelineLayout{ctx.device, pipelineLayout};

    // Compile compute shader
    auto spirv = readSpirv(spvFilename);
    if (spirv.empty()) {
        harmonia::Logger::error("LightCuller: failed to read SPIR-V from '{}'", spvFilename);
        return false;
    }
    const VkShaderModuleCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size() * sizeof(std::uint32_t),
        .pCode = spirv.data(),
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        harmonia::Logger::error("LightCuller: failed to create shader module");
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
    VkPipeline pipeline{};
    const VkResult pipeResult =
        vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(ctx.device, shaderModule, nullptr);
    if (pipeResult != VK_SUCCESS) {
        harmonia::Logger::error("LightCuller: failed to create compute pipeline");
        return false;
    }
    m_pipeline = harmonia::UniquePipeline{ctx.device, pipeline};

    // Descriptor pool + set
    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    VkDescriptorPool pool{};
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
        harmonia::Logger::error("LightCuller: failed to create descriptor pool");
        return false;
    }
    m_pool = harmonia::UniqueDescriptorPool{ctx.device, pool};
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = m_setLayout.ptr(),
    };
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &m_set) != VK_SUCCESS) {
        harmonia::Logger::error("LightCuller: failed to allocate descriptor set");
        return false;
    }

    // Pre-write the static tile buffer bindings (bindings 1 and 2 never change).
    const VkDescriptorBufferInfo countsInfo{m_tileLightCountsBuf.handle(), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo indicesInfo{m_tileLightIndicesBuf.handle(), 0, VK_WHOLE_SIZE};
    const std::array<VkWriteDescriptorSet, 2> staticWrites{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set,
                             1,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &countsInfo,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_set,
                             2,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &indicesInfo,
                             nullptr},
    };
    vkUpdateDescriptorSets(ctx.device, 2, staticWrites.data(), 0, nullptr);

    harmonia::Logger::info("LightCuller: initialized {}×{} tiles ({}×{} px each) for {}×{} screen",
                 m_tilesX,
                 m_tilesY,
                 kTileSize,
                 kTileSize,
                 w,
                 h);
    return true;
}

void LightCuller::shutdown() {
    m_tileLightCountsBuf = {};
    m_tileLightIndicesBuf = {};

    m_pool.reset();
    m_set = VK_NULL_HANDLE;
    m_pipeline.reset();
    m_pipelineLayout.reset();
    m_setLayout.reset();
    m_ctx = nullptr;
}

void LightCuller::dispatch(VkCommandBuffer cmd,
                           VkBuffer lightBuffer,
                           std::uint32_t lightCount,
                           const sm::float4x4& proj,
                           const sm::float4x4& view,
                           float nearZ,
                           float farZ) {
    if (!m_ctx || m_pipeline == VK_NULL_HANDLE || lightBuffer == VK_NULL_HANDLE)
        return;

    // Zero tile light counts so we accumulate fresh this frame.
    vkCmdFillBuffer(cmd, m_tileLightCountsBuf.handle(), 0, VK_WHOLE_SIZE, 0u);

    // Barrier: fill → compute read/write
    const VkBufferMemoryBarrier2 fillBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .buffer = m_tileLightCountsBuf.handle(),
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    const VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &fillBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);

    // Update binding 0 (light buffer) — may change per-frame (scene switch)
    const VkDescriptorBufferInfo lightInfo{lightBuffer, 0, VK_WHOLE_SIZE};
    const VkWriteDescriptorSet lightWrite{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        m_set,
        0,
        0,
        1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        nullptr,
        &lightInfo,
        nullptr,
    };
    vkUpdateDescriptorSets(m_ctx->device, 1, &lightWrite, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    const LightCullPC pc{
        .proj = proj, // row-major for Slang
        .view = view,
        .tilesXY = {m_tilesX, m_tilesY},
        .screenSize = {m_screenWidth, m_screenHeight},
        .lightCount = lightCount,
        .nearZ = nearZ,
        .farZ = farZ,
        ._pad = 0,
    };
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Dispatch: one thread per light, round up to 64
    const std::uint32_t groups = (lightCount + 63) / 64;
    if (groups > 0) {
        vkCmdDispatch(cmd, groups, 1, 1);
    }

    // Barrier: compute write → fragment shader read
    const std::array<VkBufferMemoryBarrier2, 2> readBarriers{
        VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .buffer = m_tileLightCountsBuf.handle(),
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        },
        VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .buffer = m_tileLightIndicesBuf.handle(),
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        },
    };
    const VkDependencyInfo readDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = static_cast<std::uint32_t>(readBarriers.size()),
        .pBufferMemoryBarriers = readBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &readDep);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
