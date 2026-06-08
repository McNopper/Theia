#include "theia/renderer/LightCuller.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <utility>
#include <vector>

#include "hyperion/core/Logger.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

namespace {

[[nodiscard]] std::vector<uint32_t> readSpirv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return {};
    const auto sz = f.tellg();
    if (sz <= 0 || sz % 4 != 0)
        return {};
    std::vector<uint32_t> code(static_cast<size_t>(sz) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), sz);
    return code;
}

} // namespace

LightCuller::~LightCuller() {
    shutdown();
}

LightCuller::LightCuller(LightCuller&& o) noexcept
    : m_ctx(o.m_ctx),
      m_pipeline(std::exchange(o.m_pipeline, VK_NULL_HANDLE)),
      m_pipelineLayout(std::exchange(o.m_pipelineLayout, VK_NULL_HANDLE)),
      m_setLayout(std::exchange(o.m_setLayout, VK_NULL_HANDLE)),
      m_pool(std::exchange(o.m_pool, VK_NULL_HANDLE)),
      m_set(std::exchange(o.m_set, VK_NULL_HANDLE)),
      m_tileLightCountsBuf(std::move(o.m_tileLightCountsBuf)),
      m_tileLightIndicesBuf(std::move(o.m_tileLightIndicesBuf)),
      m_tilesX(o.m_tilesX),
      m_tilesY(o.m_tilesY),
      m_screenWidth(o.m_screenWidth),
      m_screenHeight(o.m_screenHeight) {
    o.m_ctx = nullptr;
}

LightCuller& LightCuller::operator=(LightCuller&& o) noexcept {
    if (this != &o) {
        shutdown();
        m_ctx = o.m_ctx;
        m_pipeline = std::exchange(o.m_pipeline, VK_NULL_HANDLE);
        m_pipelineLayout = std::exchange(o.m_pipelineLayout, VK_NULL_HANDLE);
        m_setLayout = std::exchange(o.m_setLayout, VK_NULL_HANDLE);
        m_pool = std::exchange(o.m_pool, VK_NULL_HANDLE);
        m_set = std::exchange(o.m_set, VK_NULL_HANDLE);
        m_tileLightCountsBuf = std::move(o.m_tileLightCountsBuf);
        m_tileLightIndicesBuf = std::move(o.m_tileLightIndicesBuf);
        m_tilesX = o.m_tilesX;
        m_tilesY = o.m_tilesY;
        m_screenWidth = o.m_screenWidth;
        m_screenHeight = o.m_screenHeight;
        o.m_ctx = nullptr;
    }
    return *this;
}

bool LightCuller::initialize(const DeviceContext& ctx, uint32_t w, uint32_t h, const char* spvPath) {
    shutdown();
    m_ctx = &ctx;
    m_screenWidth = w;
    m_screenHeight = h;
    m_tilesX = (w + kTileSize - 1) / kTileSize;
    m_tilesY = (h + kTileSize - 1) / kTileSize;

    const VkDeviceSize countsBufSize = static_cast<VkDeviceSize>(m_tilesX * m_tilesY) * sizeof(uint32_t);
    const VkDeviceSize indicesBufSize =
        static_cast<VkDeviceSize>(m_tilesX * m_tilesY) * kMaxLightsPerTile * sizeof(uint32_t);

    auto counts = Buffer::create(ctx,
                                 countsBufSize,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                 "theia.tileLightCounts");
    auto indices = Buffer::create(ctx,
                                  indicesBufSize,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  "theia.tileLightIndices");
    if (!counts || !indices) {
        Logger::error("LightCuller: failed to allocate tile buffers");
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
    const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    if (vkCreateDescriptorSetLayout(ctx.device, &setLayoutInfo, nullptr, &m_setLayout) != VK_SUCCESS) {
        Logger::error("LightCuller: failed to create set layout");
        return false;
    }

    // Push constant: LightCullPC struct in shader (4×4 + 4×4 + 2 + 2 + 3 + 1 uints + floats)
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 160, // 2×64 (mat4) + 32 (vec4s + uints + floats) = 160 bytes
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_setLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    if (vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        Logger::error("LightCuller: failed to create pipeline layout");
        return false;
    }

    // Compile compute shader
    auto spirv = readSpirv(spvPath);
    if (spirv.empty()) {
        Logger::error("LightCuller: failed to read SPIR-V from '{}'", spvPath);
        return false;
    }
    const VkShaderModuleCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size() * sizeof(uint32_t),
        .pCode = spirv.data(),
    };
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        Logger::error("LightCuller: failed to create shader module");
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
        Logger::error("LightCuller: failed to create compute pipeline");
        return false;
    }

    // Descriptor pool + set
    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &m_pool) != VK_SUCCESS) {
        Logger::error("LightCuller: failed to create descriptor pool");
        return false;
    }
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_setLayout,
    };
    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &m_set) != VK_SUCCESS) {
        Logger::error("LightCuller: failed to allocate descriptor set");
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

    Logger::info("LightCuller: initialized {}×{} tiles ({}×{} px each) for {}×{} screen",
                 m_tilesX,
                 m_tilesY,
                 kTileSize,
                 kTileSize,
                 w,
                 h);
    return true;
}

void LightCuller::shutdown() {
    if (!m_ctx)
        return;

    m_tileLightCountsBuf = {};
    m_tileLightIndicesBuf = {};

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

void LightCuller::dispatch(VkCommandBuffer cmd,
                           VkBuffer lightBuffer,
                           uint32_t lightCount,
                           const glm::mat4& proj,
                           const glm::mat4& view,
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

    // Push constant layout must match LightCullPC in light_cull.comp.slang:
    //   proj (64), view (64), tilesXY (8), screenSize (8), lightCount (4), nearZ (4), farZ (4), _pad (4) = 160 bytes
    struct LightCullPC {
        glm::mat4 proj;
        glm::mat4 view;
        glm::uvec2 tilesXY;
        glm::uvec2 screenSize;
        uint32_t lightCount;
        float nearZ;
        float farZ;
        uint32_t _pad;
    };
    static_assert(sizeof(LightCullPC) == 160);

    const LightCullPC pc{
        .proj = glm::transpose(proj), // row-major for Slang
        .view = glm::transpose(view),
        .tilesXY = {m_tilesX, m_tilesY},
        .screenSize = {m_screenWidth, m_screenHeight},
        .lightCount = lightCount,
        .nearZ = nearZ,
        .farZ = farZ,
        ._pad = 0,
    };
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Dispatch: one thread per light, round up to 64
    const uint32_t groups = (lightCount + 63) / 64;
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
        .bufferMemoryBarrierCount = static_cast<uint32_t>(readBarriers.size()),
        .pBufferMemoryBarriers = readBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &readDep);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
