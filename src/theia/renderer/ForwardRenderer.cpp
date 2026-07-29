#include "theia/renderer/ForwardRenderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <harmonia/core/Logger.hpp>
#include <harmonia/core/ShaderModule.hpp>
#include <numeric>
#include <slang-math/slang-math.hpp>
#include <theia/renderer/CameraJitter.hpp>
#include <theia/renderer/ShaderPath.hpp>
#include <theia/scene/Scene.hpp>
#include <vector>

#include "theia/renderer/ForwardRenderer.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

ForwardRenderer::~ForwardRenderer() {
    shutdown();
}

bool ForwardRenderer::initialize(const DeviceContext& ctx, const Config& config) {
    m_config = config;
    m_ctx = &ctx;

    if (!createDepthTarget()) {
        Logger::error("Failed to create depth target");
        return false;
    }

    // Hi-Z depth pyramid for two-pass occlusion culling. A pipeline failure (e.g. missing
    // shader) is non-fatal: the image is still created, the occlusion test disabled, and the
    // renderer falls back to drawing all meshlets. Only a failure to create the image itself
    // (catastrophic / OOM) is fatal.
    (void)m_hiZPass.initialize(ctx, config.width, config.height);
    if (m_hiZPass.sampledView() == VK_NULL_HANDLE) {
        Logger::error("Failed to create Hi-Z image");
        return false;
    }

    // GPU-driven frustum cull pass (GD2/GD3). Non-fatal on failure: renderer falls back to
    // vkCmdDrawMeshTasksEXT with the CPU-known instance count.
    if (!m_gpuCullPass.initialize(ctx)) {
        Logger::warn("GpuCullPass: initialization failed — falling back to CPU-count draw");
    }

    // Debug/A-B toggle: set THEIA_FORCE_GD3 to skip DGC layout creation and force the
    // vkCmdDrawMeshTasksIndirectEXT (GD3) draw path. Used to isolate DGC-specific regressions.
    bool forceGd3 = false;
    {
        char* forceGd3Raw = nullptr;
        std::size_t forceGd3Len = 0;
        if (_dupenv_s(&forceGd3Raw, &forceGd3Len, "THEIA_FORCE_GD3") == 0 && forceGd3Raw != nullptr) {
            forceGd3 = (forceGd3Raw[0] != '\0' && forceGd3Raw[0] != '0');
            std::free(forceGd3Raw);
            if (forceGd3) {
                Logger::info("THEIA_FORCE_GD3 set — DGC disabled, using indirect draw fallback");
            }
        }
    }

    // GD6: create DGC indirect commands layout when VK_EXT_device_generated_commands is available.
    // One token: DRAW_MESH_TASKS_EXT (stride=12 = VkDrawMeshTasksIndirectCommandEXT).
    // sequenceCountAddress will point to indirectDrawBuf[0..3] (visible instance count).
    if (ctx.dgcSupported && m_gpuCullPass.isInitialized() && !forceGd3) {
        const VkIndirectCommandsLayoutTokenEXT dgcToken{
            .sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,
            .type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_EXT,
            .offset = 0,
        };
        const VkIndirectCommandsLayoutCreateInfoEXT dgcLayoutCI{
            .sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
            .flags = 0,
            .shaderStages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .indirectStride = 12u,            // sizeof(VkDrawMeshTasksIndirectCommandEXT)
            .pipelineLayout = VK_NULL_HANDLE, // no push-constant token
            .tokenCount = 1,
            .pTokens = &dgcToken,
        };
        if (vkCreateIndirectCommandsLayoutEXT(ctx.device, &dgcLayoutCI, nullptr, &m_dgcLayout) != VK_SUCCESS) {
            Logger::warn("ForwardRenderer: vkCreateIndirectCommandsLayoutEXT failed — GD3 fallback");
            m_dgcLayout = VK_NULL_HANDLE;
        }
    }

    if (!createPipeline()) {
        Logger::error("Failed to create pipeline");
        return false;
    }

    // GD6: query and allocate preprocess buffer after pipeline creation (requires pipeline handle).
    // VkBufferUsageFlags2CreateInfo is needed because VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT
    // is a 64-bit flag not representable in the 32-bit VkBufferUsageFlags enum.
    if (m_dgcLayout != VK_NULL_HANDLE) {
        const VkGeneratedCommandsPipelineInfoEXT memReqPipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,
            .pipeline = m_graphicsPipeline,
        };
        const VkGeneratedCommandsMemoryRequirementsInfoEXT memReqInfo{
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT,
            .pNext = &memReqPipelineInfo,
            .indirectExecutionSet = VK_NULL_HANDLE,
            .indirectCommandsLayout = m_dgcLayout,
            .maxSequenceCount = 1u, // single GPU-generated draw (see drawOpaque/drawTransparent)
            .maxDrawCount = 1,
        };
        VkMemoryRequirements2 memReq{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        vkGetGeneratedCommandsMemoryRequirementsEXT(ctx.device, &memReqInfo, &memReq);
        m_dgcPreprocessSize = memReq.memoryRequirements.size;

        if (m_dgcPreprocessSize > 0) {
            constexpr VkBufferUsageFlags2KHR kPreprocessUsage =
                VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR;
            const VkBufferUsageFlags2CreateInfo usageFlags2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
                .usage = kPreprocessUsage,
            };
            const VkBufferCreateInfo ppBufCI{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = &usageFlags2,
                .size = m_dgcPreprocessSize,
                .usage = 0, // usage provided via VkBufferUsageFlags2CreateInfo
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            const VmaAllocationCreateInfo ppAllocCI{
                .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            };
            if (vmaCreateBuffer(
                    ctx.allocator, &ppBufCI, &ppAllocCI, &m_dgcPreprocessBuf, &m_dgcPreprocessAlloc, nullptr) ==
                VK_SUCCESS) {
                const VkBufferDeviceAddressInfo addrInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                    .buffer = m_dgcPreprocessBuf,
                };
                m_dgcPreprocessAddr = vkGetBufferDeviceAddress(ctx.device, &addrInfo);
            } else {
                Logger::warn("ForwardRenderer: failed to allocate DGC preprocess buffer — GD3 fallback");
                vkDestroyIndirectCommandsLayoutEXT(ctx.device, m_dgcLayout, nullptr);
                m_dgcLayout = VK_NULL_HANDLE;
                m_dgcPreprocessSize = 0;
            }
        }
    }

    char* debugModeRaw = nullptr;
    std::size_t debugModeLen = 0;
    if (_dupenv_s(&debugModeRaw, &debugModeLen, "THEIA_DEBUG_RAY_HIT_MODE") == 0 && debugModeRaw != nullptr) {
        m_debugRayHitMode = std::clamp(std::strtof(debugModeRaw, nullptr), 0.0f, 6.0f);
        std::free(debugModeRaw);
        if (m_debugRayHitMode > 0.0f) {
            Logger::info("THEIA_DEBUG_RAY_HIT_MODE = {:.1f}", m_debugRayHitMode);
        }
    }

    // Debug/A-B toggle: set THEIA_DISABLE_HIZ to draw all meshlets (two passes, no occlusion
    // test). Used to verify Hi-Z culling is visually equivalent to the uncullered path.
    char* hiZDisableRaw = nullptr;
    std::size_t hiZDisableLen = 0;
    if (_dupenv_s(&hiZDisableRaw, &hiZDisableLen, "THEIA_DISABLE_HIZ") == 0 && hiZDisableRaw != nullptr) {
        m_hiZDebugDisabled = (hiZDisableRaw[0] != '\0' && hiZDisableRaw[0] != '0');
        std::free(hiZDisableRaw);
        if (m_hiZDebugDisabled) {
            Logger::info("THEIA_DISABLE_HIZ set — Hi-Z occlusion test disabled");
        }
    }

    char* singlePassRaw = nullptr;
    std::size_t singlePassLen = 0;
    if (_dupenv_s(&singlePassRaw, &singlePassLen, "THEIA_SINGLE_PASS") == 0 && singlePassRaw != nullptr) {
        m_forceSinglePass = (singlePassRaw[0] != '\0' && singlePassRaw[0] != '0');
        std::free(singlePassRaw);
        if (m_forceSinglePass) {
            Logger::info("THEIA_SINGLE_PASS set — two-pass Hi-Z bypassed (single draw pass)");
        }
    }

    // Create 1-element dummy buffers for tile light slots (fallback when LightCuller hasn't run).
    constexpr VkDeviceSize kDummySize = sizeof(std::uint32_t) * 128; // >= kMaxLightsPerTile
    auto dummyCounts = Buffer::create(ctx,
                                      kDummySize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      "theia.dummyTileCounts");
    auto dummyIndices = Buffer::create(ctx,
                                       kDummySize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                       "theia.dummyTileIndices");
    if (dummyCounts && dummyIndices) {
        m_dummyTileCounts = std::move(*dummyCounts);
        m_dummyTileIndices = std::move(*dummyIndices);
    }

    // Identity instance list [0,1,...,kMaxInstances-1] for binding 10 fallback when
    // GpuCullPass fails to initialize. Written once; GPU reads it as a pass-through index map.
    {
        const std::uint32_t kIdCount = GpuCullPass::kMaxInstances;
        std::vector<std::uint32_t> identity(kIdCount);
        std::iota(identity.begin(), identity.end(), 0u);
        auto identBuf = Buffer::create(ctx,
                                       static_cast<VkDeviceSize>(kIdCount) * sizeof(std::uint32_t),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VMA_MEMORY_USAGE_CPU_TO_GPU,
                                       "theia.identityInstanceList");
        if (identBuf) {
            identBuf->uploadData(identity.data(), static_cast<VkDeviceSize>(kIdCount) * sizeof(std::uint32_t), 0);
            m_identityInstanceList = std::move(*identBuf);
        }
    }

    m_initialized = true;
    m_hdrFirstUse = true;
    const char* drawPath = m_dgcLayout != VK_NULL_HANDLE   ? "DGC"
                           : m_gpuCullPass.isInitialized() ? "indirect"
                                                           : "CPU-count";
    Logger::info("GPU-driven forward renderer initialized ({}x{}) [{}]", config.width, config.height, drawPath);
    return true;
}

void ForwardRenderer::shutdown() {
    if (!m_ctx) {
        return;
    }

    vkDeviceWaitIdle(m_ctx->device);

    if (m_graphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx->device, m_graphicsPipeline, nullptr);
        m_graphicsPipeline = VK_NULL_HANDLE;
    }
    if (m_graphicsPipelineTransparent != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx->device, m_graphicsPipelineTransparent, nullptr);
        m_graphicsPipelineTransparent = VK_NULL_HANDLE;
    }
    if (m_skyPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx->device, m_skyPipeline, nullptr);
        m_skyPipeline = VK_NULL_HANDLE;
    }
    if (m_skyPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_ctx->device, m_skyPipelineLayout, nullptr);
        m_skyPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_ctx->device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_meshSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
    }
    if (m_matSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
    }
    if (m_iblSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
    }
    if (m_textureSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
    }
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }

    m_iblDiffuseInfo = {};
    m_iblSpecularInfo = {};
    m_sheenLutInfo = {};
    m_brdfLutInfo = {};
    m_iblEnvSamplerInfo = {};
    m_iblEnvRawInfo = {};
    m_envMarginalCdf = VK_NULL_HANDLE;
    m_envConditionalCdf = VK_NULL_HANDLE;
    m_envImportanceWidth = 0;
    m_envImportanceHeight = 0;

    m_dummyTileCounts = {};
    m_dummyTileIndices = {};
    m_identityInstanceList = {};

    if (m_dgcPreprocessBuf != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_ctx->allocator, m_dgcPreprocessBuf, m_dgcPreprocessAlloc);
        m_dgcPreprocessBuf = VK_NULL_HANDLE;
        m_dgcPreprocessAlloc = VK_NULL_HANDLE;
        m_dgcPreprocessAddr = 0;
        m_dgcPreprocessSize = 0;
    }

    if (m_dgcLayout != VK_NULL_HANDLE) {
        vkDestroyIndirectCommandsLayoutEXT(m_ctx->device, m_dgcLayout, nullptr);
        m_dgcLayout = VK_NULL_HANDLE;
    }

    m_meshSet = VK_NULL_HANDLE;
    m_matSet = VK_NULL_HANDLE;
    m_iblSet = VK_NULL_HANDLE;
    m_textureSet = VK_NULL_HANDLE;
    m_texturesBoundFor = nullptr;
    m_hiZPass.shutdown();
    m_gpuCullPass.shutdown();
    m_meshletVisibility[0] = {};
    m_meshletVisibility[1] = {};
    m_visBuiltFor = nullptr;
    m_visMeshletCount = 0;
    m_visFrame = 0;
    m_depthTarget = {};
    m_gbufferTarget = {};
    m_giBufferTarget = {};
    m_ctx = nullptr;

    m_initialized = false;
}

bool ForwardRenderer::resize(std::uint32_t width,
                             std::uint32_t height,
                             VkImage hdrImage,
                             VkImageView hdrImageView) noexcept {
    if (!m_ctx) {
        return false;
    }
    m_config.width = width;
    m_config.height = height;
    m_config.hdrImage = hdrImage;
    m_config.hdrImageView = hdrImageView;
    m_depthTarget = {};
    m_gbufferTarget = {};
    m_giBufferTarget = {};
    m_hdrFirstUse = true;
    // Recreate the Hi-Z pyramid at the new resolution. Visibility buffers are
    // resolution-independent (one entry per meshlet) and are kept across resize.
    (void)m_hiZPass.initialize(*m_ctx, width, height);
    return createDepthTarget();
}

void ForwardRenderer::setTileBuffers(VkBuffer tileLightCounts,
                                     VkBuffer tileLightIndices,
                                     std::uint32_t tilesX,
                                     std::uint32_t tilesY) {
    m_tileLightCountsBuf = tileLightCounts;
    m_tileLightIndicesBuf = tileLightIndices;
    m_tilesX = tilesX;
    m_tilesY = tilesY;
}

void ForwardRenderer::setIbl(const IblResources& res, VkImageView rawEnvView, float envUnitNits) {
    if (!m_ctx || m_iblSet == VK_NULL_HANDLE) {
        return;
    }

    m_iblDiffuseInfo =
        VkDescriptorImageInfo{VK_NULL_HANDLE, res.diffuseIrrad.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_iblSpecularInfo =
        VkDescriptorImageInfo{VK_NULL_HANDLE, res.specularMipped.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_sheenLutInfo =
        VkDescriptorImageInfo{VK_NULL_HANDLE, res.sheenLut.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_brdfLutInfo = VkDescriptorImageInfo{VK_NULL_HANDLE, res.brdfLut.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_iblEnvSamplerInfo = VkDescriptorImageInfo{res.envSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    // Raw env panorama for the sky background. Fall back to the specular view (a valid
    // SAMPLED_IMAGE) when no raw env is supplied, so the descriptor stays valid.
    const VkImageView envView = (rawEnvView != VK_NULL_HANDLE) ? rawEnvView : res.specularMipped.view();
    m_iblEnvRawInfo = VkDescriptorImageInfo{VK_NULL_HANDLE, envView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_envUnitNits = envUnitNits;

    const std::array<VkWriteDescriptorSet, 6> writes{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             0,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_iblDiffuseInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             1,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_iblSpecularInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             2,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_sheenLutInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             5,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_brdfLutInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             3,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLER,
                             &m_iblEnvSamplerInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             4,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_iblEnvRawInfo,
                             nullptr,
                             nullptr},
    };
    vkUpdateDescriptorSets(m_ctx->device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool ForwardRenderer::createDepthTarget() {
    VkExtent2D extent{m_config.width, m_config.height};
    // SAMPLED_BIT so SSR can read depth after the forward pass.
    auto depthResult = Image::create(*m_ctx,
                                     extent,
                                     VK_FORMAT_D32_SFLOAT,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                     VK_IMAGE_ASPECT_DEPTH_BIT,
                                     "theia.depth");
    if (!depthResult) {
        Logger::error("Failed to create depth target");
        return false;
    }
    m_depthTarget = std::move(*depthResult);

    // GBuffer: RGBA16F — view-space normal (xyz * 0.5 + 0.5) + roughness (w).
    // SAMPLED_BIT so SSR compute can sample it; STORAGE_BIT reserved for future post-process.
    auto gbufResult = Image::create(*m_ctx,
                                    extent,
                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    "theia.gbuffer");
    if (!gbufResult) {
        Logger::error("Failed to create GBuffer target");
        return false;
    }
    m_gbufferTarget = std::move(*gbufResult);

    // GI GBuffer: RGBA32F — world-space primary hit position (xyz) + material index (w,
    // encoded as asfloat(materialIdx + 1); 0 = background). RGBA32F keeps the integer
    // material index bit-exact and the world position at full precision for ray seeding.
    // SAMPLED_BIT so the GI compute stage can fetch it via texelFetch (Texture2D.Load).
    auto giBufResult = Image::create(*m_ctx,
                                     extent,
                                     VK_FORMAT_R32G32B32A32_SFLOAT,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     "theia.gibuffer");
    if (!giBufResult) {
        Logger::error("Failed to create GI GBuffer target");
        return false;
    }
    m_giBufferTarget = std::move(*giBufResult);
    return true;
}

bool ForwardRenderer::ensureVisibilityBuffers() {
    if (m_scene == nullptr) {
        return false;
    }
    if (m_visBuiltFor == m_scene && m_meshletVisibility[0].handle() != VK_NULL_HANDLE) {
        return true;
    }
    const std::uint32_t meshletCount = std::max(1u, m_scene->meshletCount());
    const VkDeviceSize size = static_cast<VkDeviceSize>(meshletCount) * sizeof(std::uint32_t);
    for (auto& buf : m_meshletVisibility) {
        auto created = Buffer::create(*m_ctx,
                                      size,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      "theia.meshletVisibility");
        if (!created) {
            Logger::error("Failed to create meshlet visibility buffer");
            return false;
        }
        buf = std::move(*created);
    }
    m_visMeshletCount = meshletCount;
    m_visBuiltFor = m_scene;
    m_visFrame = 0;
    // Both buffers hold undefined VMA memory; the first frame clears both (see recordFrame).
    m_visClearPrev = true;
    return true;
}

void ForwardRenderer::drawOpaque(VkCommandBuffer cmd,
                                 const MeshPushConstants& pcBase,
                                 std::uint32_t cullPhase,
                                 std::uint32_t hiZMipCount) {
    const std::uint32_t instanceCount = m_scene->instanceCount();
    if (instanceCount == 0 || vkCmdDrawMeshTasksEXT == nullptr) {
        return;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    const std::array<VkDescriptorSet, 4> descSets{m_meshSet, m_matSet, m_iblSet, m_textureSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 4, descSets.data(), 0, nullptr);
    auto pc = pcBase;
    pc.presentationParams.y = 0.0f; // opaque routing in the task shader
    pc.cullPhase = cullPhase;
    pc.hiZMipCount = hiZMipCount;
    vkCmdPushConstants(cmd,
                       m_pipelineLayout,
                       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(MeshPushConstants),
                       &pc);
    if (m_gpuCullPass.isInitialized() && m_dgcLayout != VK_NULL_HANDLE) {
        // GD6 DGC path: one GPU-generated DRAW_MESH_TASKS command via VK_EXT_device_generated_commands.
        // indirectAddress points at indirectDrawBuf = {visibleCount, 1, 1} (written by GpuCullPass),
        // so a single sequence dispatches `visibleCount` task workgroups. The task shader indexes
        // compactInstanceList[gid.x] — identical semantics to the GD3 fallback, but the command
        // itself is GPU-resident and consumed through the modern device-generated-commands path.
        const VkGeneratedCommandsPipelineInfoEXT dgcPipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,
            .pipeline = m_graphicsPipeline,
        };
        const VkGeneratedCommandsInfoEXT dgcInfo{
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,
            .pNext = &dgcPipelineInfo,
            .shaderStages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .indirectExecutionSet = VK_NULL_HANDLE,
            .indirectCommandsLayout = m_dgcLayout,
            .indirectAddress = m_gpuCullPass.indirectDrawAddress(),
            .indirectAddressSize = 12u, // one VkDrawMeshTasksIndirectCommandEXT
            .preprocessAddress = m_dgcPreprocessAddr,
            .preprocessSize = m_dgcPreprocessSize,
            .maxSequenceCount = 1u,    // single GPU-generated draw
            .sequenceCountAddress = 0, // fixed count of 1
            .maxDrawCount = 1,
        };
        vkCmdExecuteGeneratedCommandsEXT(cmd, VK_FALSE, &dgcInfo);
    } else if (m_gpuCullPass.isInitialized() && vkCmdDrawMeshTasksIndirectEXT != nullptr) {
        // GD3 fallback: single indirect draw; task shader gid.x = visible instance position.
        vkCmdDrawMeshTasksIndirectEXT(cmd,
                                      m_gpuCullPass.indirectDrawBuffer(),
                                      0,
                                      1,    // exactly one indirect command entry
                                      12u); // stride = sizeof(VkDrawMeshTasksIndirectCommandEXT)
    } else {
        vkCmdDrawMeshTasksEXT(cmd, instanceCount, 1, 1);
    }
}

bool ForwardRenderer::createPipeline() {
    auto loadShaderModule = [this](const char* filename) -> VkShaderModule {
        auto module = harmonia::createShaderModule(m_ctx->device, shaderPath(filename));
        if (!module) {
            Logger::error("Failed to load shader: {}", filename);
            return VK_NULL_HANDLE;
        }
        return *module;
    };

    VkShaderModule taskModule = loadShaderModule("forward_render.task.spv");
    VkShaderModule meshModule = loadShaderModule("forward_render.mesh.spv");
    VkShaderModule fragModule = loadShaderModule("forward_render.frag.spv");
    if (!taskModule || !meshModule || !fragModule) {
        if (taskModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        }
        if (meshModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        }
        if (fragModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        }
        return false;
    }

    constexpr VkShaderStageFlags kTaskAndMeshStages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    constexpr VkShaderStageFlags kTaskMeshFragStages =
        VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(ForwardRenderer::MeshPushConstants),
    };

    // Set 0: geometry buffers + material table for task-shader bucketing.
    // Bindings 7-9 drive two-pass Hi-Z occlusion culling in the mesh shader:
    //   7 = previous-frame per-meshlet visibility (read), 8 = current-frame visibility
    //   (written), 9 = current-frame Hi-Z max-depth pyramid (sampled).
    // Binding 10: compactInstanceList from GpuCullPass — written by the cull compute
    //   shader each frame; task shader reads it via SV_DrawID to get the instance index.
    constexpr VkShaderStageFlags kTaskStage = VK_SHADER_STAGE_TASK_BIT_EXT;
    const std::array<VkDescriptorSetLayoutBinding, 12> meshBindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskMeshFragStages, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskMeshFragStages, nullptr},
        VkDescriptorSetLayoutBinding{2,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     1,
                                     VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                     nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskAndMeshStages, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskStage, nullptr},
        VkDescriptorSetLayoutBinding{7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskStage, nullptr},
        VkDescriptorSetLayoutBinding{11,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     1,
                                     VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                     nullptr}, // instance transforms
    };
    constexpr VkDescriptorBindingFlags kUAB = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    const std::array<VkDescriptorBindingFlags, 12> meshBindingFlags{
        kUAB, kUAB, kUAB, kUAB, kUAB, kUAB, kUAB, kUAB, kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo meshBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(meshBindingFlags.size()),
        .pBindingFlags = meshBindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo meshSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &meshBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<std::uint32_t>(meshBindings.size()),
        .pBindings = meshBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &meshSetLayoutInfo, nullptr, &m_meshSetLayout) != VK_SUCCESS) {
        Logger::error("Failed to create mesh descriptor set layout");
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // Set 1: material/lighting buffers (materials, lights, emissive triangles, tile lists — fragment stage)
    // Binding 5 = scene TLAS for ray-traced shadows (ray query in the fragment shader).
    const std::array<VkDescriptorSetLayoutBinding, 6> matBindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{
            3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // tileLightCounts
        VkDescriptorSetLayoutBinding{
            4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // tileLightIndices
        VkDescriptorSetLayoutBinding{
            5, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // scene TLAS
    };
    const std::array<VkDescriptorBindingFlags, 6> matBindingFlags{kUAB, kUAB, kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo matBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(matBindingFlags.size()),
        .pBindingFlags = matBindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo matSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &matBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<std::uint32_t>(matBindings.size()),
        .pBindings = matBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &matSetLayoutInfo, nullptr, &m_matSetLayout) != VK_SUCCESS) {
        Logger::error("Failed to create material descriptor set layout");
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // Set 2: IBL textures + sampler + env importance CDFs (fragment stage)
    // binding 0: t_iblDiffuse, 1: t_iblSpecular, 2: t_sheenLut, 3: s_iblLinear, 4: t_envRaw, 5: t_brdfLut
    // binding 6: envMarginalCdf, 7: envConditionalCdf
    const std::array<VkDescriptorSetLayoutBinding, 8> iblBindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    const std::array<VkDescriptorBindingFlags, 8> iblBindingFlags{kUAB, kUAB, kUAB, kUAB, kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo iblBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(iblBindingFlags.size()),
        .pBindingFlags = iblBindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo iblSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &iblBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<std::uint32_t>(iblBindings.size()),
        .pBindings = iblBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &iblSetLayoutInfo, nullptr, &m_iblSetLayout) != VK_SUCCESS) {
        Logger::error("Failed to create IBL descriptor set layout");
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // Set 3: bindless material textures (variable array of combined image samplers, fragment stage).
    // Uses partially-bound descriptors so only the slots actually populated per scene must be valid.
    const VkDescriptorSetLayoutBinding textureBinding{
        0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    const VkDescriptorBindingFlags textureBindingFlags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    const VkDescriptorSetLayoutBindingFlagsCreateInfo textureBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = 1,
        .pBindingFlags = &textureBindingFlags,
    };
    const VkDescriptorSetLayoutCreateInfo textureSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &textureBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = 1,
        .pBindings = &textureBinding,
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &textureSetLayoutInfo, nullptr, &m_textureSetLayout) != VK_SUCCESS) {
        Logger::error("Failed to create bindless texture descriptor set layout");
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // Single pool for all four sets. Storage-buffer count: set 0 (mesh) 11 +
    // set 1 (mat) 5 + set 2 (ibl env CDFs 6/7) 2 = 18 total. Set 3 (bindless
    // textures) uses COMBINED_IMAGE_SAMPLER, no storage-buffer slot.
    const std::array<VkDescriptorPoolSize, 5> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 18},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 4,
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        Logger::error("Failed to create descriptor pool");
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    const std::array<VkDescriptorSetLayout, 4> setLayouts{
        m_meshSetLayout, m_matSetLayout, m_iblSetLayout, m_textureSetLayout};
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = static_cast<std::uint32_t>(setLayouts.size()),
        .pSetLayouts = setLayouts.data(),
    };
    std::array<VkDescriptorSet, 4> sets{};
    if (vkAllocateDescriptorSets(m_ctx->device, &allocInfo, sets.data()) != VK_SUCCESS) {
        Logger::error("Failed to allocate descriptor sets");
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }
    m_meshSet = sets[0];
    m_matSet = sets[1];
    m_iblSet = sets[2];
    m_textureSet = sets[3];

    const std::array<VkDescriptorSetLayout, 4> pipelineSetLayouts{
        m_meshSetLayout, m_matSetLayout, m_iblSetLayout, m_textureSetLayout};
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(pipelineSetLayouts.size()),
        .pSetLayouts = pipelineSetLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        Logger::error("Failed to create graphics pipeline layout");
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    const std::array<VkPipelineShaderStageCreateInfo, 3> graphicsStages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
            .module = taskModule,
            .pName = "main",
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
            .module = meshModule,
            .pName = "main",
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragModule,
            .pName = "main",
        },
    };

    const VkPipelineViewportStateCreateInfo viewport{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencilTransparent{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineColorBlendAttachmentState colorAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    // GBuffer (target 1) blend state — write all, no blend
    const VkPipelineColorBlendAttachmentState gbufferAttachment = colorAttachment;
    // GI GBuffer (target 2) blend state — write all, no blend (opaque pass only)
    const VkPipelineColorBlendAttachmentState giBufferAttachment = colorAttachment;
    const std::array<VkPipelineColorBlendAttachmentState, 3> colorAttachments{
        colorAttachment, gbufferAttachment, giBufferAttachment};
    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<std::uint32_t>(colorAttachments.size()),
        .pAttachments = colorAttachments.data(),
    };
    const VkPipelineColorBlendAttachmentState transparentGbufferAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask = 0,
    };
    // Transparent pass keeps the opaque-written view-normal GBuffer (target 1) intact
    // (mask it out) but DOES write the GI GBuffer (target 2: world position + material
    // index) so the shared GI compute path integrator processes transparent surfaces
    // (refraction/reflection/env) exactly like opaque ones — one unified quality pipeline.
    const std::array<VkPipelineColorBlendAttachmentState, 3> transparentColorAttachments{
        colorAttachment, transparentGbufferAttachment, giBufferAttachment};
    const VkPipelineColorBlendStateCreateInfo transparentColorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<std::uint32_t>(transparentColorAttachments.size()),
        .pAttachments = transparentColorAttachments.data(),
    };
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    const std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };
    constexpr VkFormat kGbufferFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr VkFormat kGiBufferFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    const std::array<VkFormat, 3> colorFormats{m_config.outputFormat, kGbufferFormat, kGiBufferFormat};
    const VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = static_cast<std::uint32_t>(colorFormats.size()),
        .pColorAttachmentFormats = colorFormats.data(),
        .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
    };

    const VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering,
        .stageCount = static_cast<std::uint32_t>(graphicsStages.size()),
        .pStages = graphicsStages.data(),
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = nullptr,
        .pViewportState = &viewport,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = m_pipelineLayout,
    };
    if (vkCreateGraphicsPipelines(m_ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) !=
        VK_SUCCESS) {
        Logger::error("Failed to create graphics pipeline");
        vkDestroyPipelineLayout(m_ctx->device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    const VkGraphicsPipelineCreateInfo transparentPipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering,
        .stageCount = static_cast<std::uint32_t>(graphicsStages.size()),
        .pStages = graphicsStages.data(),
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = nullptr,
        .pViewportState = &viewport,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depthStencilTransparent,
        .pColorBlendState = &transparentColorBlend,
        .pDynamicState = &dynamicState,
        .layout = m_pipelineLayout,
    };
    if (vkCreateGraphicsPipelines(
            m_ctx->device, VK_NULL_HANDLE, 1, &transparentPipelineInfo, nullptr, &m_graphicsPipelineTransparent) !=
        VK_SUCCESS) {
        Logger::error("Failed to create transparent graphics pipeline");
        vkDestroyPipeline(m_ctx->device, m_graphicsPipeline, nullptr);
        m_graphicsPipeline = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(m_ctx->device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // ── Sky/background pipeline ───────────────────────────────────────────────
    // Fullscreen triangle that samples the environment panorama (or outputs black).
    // Reuses the IBL descriptor set layout (set 0 == specular prefilter + sampler).
    {
        VkShaderModule skyVert = loadShaderModule("sky.vert.spv");
        VkShaderModule skyFrag = loadShaderModule("sky.frag.spv");
        if (skyVert == VK_NULL_HANDLE || skyFrag == VK_NULL_HANDLE) {
            Logger::error("Failed to load sky shader modules");
            if (skyVert)
                vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
            if (skyFrag)
                vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
            vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
            return false;
        }

        const VkPushConstantRange skyPcRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(SkyPushConstants),
        };
        const VkPipelineLayoutCreateInfo skyLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_iblSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &skyPcRange,
        };
        if (vkCreatePipelineLayout(m_ctx->device, &skyLayoutInfo, nullptr, &m_skyPipelineLayout) != VK_SUCCESS) {
            Logger::error("Failed to create sky pipeline layout");
            vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
            vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
            vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
            return false;
        }

        const std::array<VkPipelineShaderStageCreateInfo, 2> skyStages{
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = skyVert,
                .pName = "main",
            },
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = skyFrag,
                .pName = "main",
            },
        };
        const VkPipelineInputAssemblyStateCreateInfo skyInputAsm{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        // No depth test/write — drawn first; geometry overwrites where present.
        const VkPipelineDepthStencilStateCreateInfo skyDepth{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_FALSE,
            .depthWriteEnable = VK_FALSE,
            .depthCompareOp = VK_COMPARE_OP_ALWAYS,
        };
        // Color 0 (HDR) writes RGBA; color 1 (GBuffer) is left untouched (keeps clear).
        const VkPipelineColorBlendAttachmentState skyHdrBlend{
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
        };
        const VkPipelineColorBlendAttachmentState skyGbufBlend{.blendEnable = VK_FALSE, .colorWriteMask = 0};
        const std::array<VkPipelineColorBlendAttachmentState, 3> skyBlends{skyHdrBlend, skyGbufBlend, skyGbufBlend};
        const VkPipelineColorBlendStateCreateInfo skyColorBlend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = static_cast<std::uint32_t>(skyBlends.size()),
            .pAttachments = skyBlends.data(),
        };

        const VkGraphicsPipelineCreateInfo skyPipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering,
            .stageCount = static_cast<std::uint32_t>(skyStages.size()),
            .pStages = skyStages.data(),
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &skyInputAsm,
            .pViewportState = &viewport,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &skyDepth,
            .pColorBlendState = &skyColorBlend,
            .pDynamicState = &dynamicState,
            .layout = m_skyPipelineLayout,
        };
        const VkResult skyRes =
            vkCreateGraphicsPipelines(m_ctx->device, VK_NULL_HANDLE, 1, &skyPipelineInfo, nullptr, &m_skyPipeline);
        vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
        vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
        if (skyRes != VK_SUCCESS) {
            Logger::error("Failed to create sky pipeline");
            vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
            return false;
        }
    }

    vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
    vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
    vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
    return true;
}

void ForwardRenderer::recordFrame(VkCommandBuffer cmd) {
    if (!m_scene || m_config.hdrImage == VK_NULL_HANDLE) {
        return;
    }

    // Ensure the two-pass Hi-Z visibility buffers exist for the current scene.
    const bool visReady = ensureVisibilityBuffers();

    const std::array imageBarriers{
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = m_hdrFirstUse ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = m_hdrFirstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_config.hdrImage,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        },
        // Depth: always discard (UNDEFINED) — depth is fully cleared each frame.
        // SAMPLED_BIT added so SSR can read it after the pass.
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_depthTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
        },
        // GBuffer: always discard (UNDEFINED) — fully written each frame.
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_gbufferTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        },
        // GI GBuffer: always discard (UNDEFINED) — fully written each frame by the opaque pass.
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_giBufferTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        },
    };
    const VkDependencyInfo hdrDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &hdrDep);
    m_hdrFirstUse = false;

    // Create buffer info structures for descriptor updates
    VkDescriptorBufferInfo vertexBufferInfo{
        .buffer = m_scene->vertexBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo instanceBufferInfo{
        .buffer = m_scene->instanceBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo instanceTransformInfo{
        .buffer = m_scene->instanceTransformBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo indexBufferInfo{
        .buffer = m_scene->indexBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo meshletBufferInfo{
        .buffer = m_scene->meshletBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo meshletVertexBufferInfo{
        .buffer = m_scene->meshletVertexBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo meshletTriangleBufferInfo{
        .buffer = m_scene->meshletTriangleBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    VkDescriptorBufferInfo materialBufferInfo{
        .buffer = m_scene->materialBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo lightBufferInfo{
        .buffer = m_scene->lightBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo emissiveTriangleBufferInfo{
        .buffer = m_scene->emissiveTriangleBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    const VkBuffer cdfFallbackBuffer = (m_dummyTileCounts.handle() != VK_NULL_HANDLE)
                                           ? m_dummyTileCounts.handle()
                                           : m_scene->materialBuffer().handle();
    VkDescriptorBufferInfo envMarginalCdfInfo{
        .buffer = (m_envMarginalCdf != VK_NULL_HANDLE) ? m_envMarginalCdf : cdfFallbackBuffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo envConditionalCdfInfo{
        .buffer = (m_envConditionalCdf != VK_NULL_HANDLE) ? m_envConditionalCdf : cdfFallbackBuffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    // Tile light lists: use real buffers if LightCuller ran, else dummy fallback.
    const VkBuffer tileCntBuf = (m_tileLightCountsBuf != VK_NULL_HANDLE && m_dummyTileCounts.handle() != VK_NULL_HANDLE)
                                    ? m_tileLightCountsBuf
                                    : m_dummyTileCounts.handle();
    const VkBuffer tileIdxBuf =
        (m_tileLightIndicesBuf != VK_NULL_HANDLE && m_dummyTileIndices.handle() != VK_NULL_HANDLE)
            ? m_tileLightIndicesBuf
            : m_dummyTileIndices.handle();
    const bool hasTileData = (tileCntBuf != VK_NULL_HANDLE && tileIdxBuf != VK_NULL_HANDLE);
    VkDescriptorBufferInfo tileLightCountsInfo{
        .buffer = hasTileData ? tileCntBuf : m_dummyTileCounts.handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo tileLightIndicesInfo{
        .buffer = hasTileData ? tileIdxBuf : m_dummyTileIndices.handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    const VkAccelerationStructureKHR sceneTlas = m_scene->tlas();
    const VkWriteDescriptorSetAccelerationStructureKHR tlasWriteInfo{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &sceneTlas,
    };

    // Two-pass Hi-Z bindings (set 0, bindings 7-9). prev = last frame's visibility,
    // curr = this frame's (written by the mesh shader). Buffers ping-pong per frame. If the
    // visibility buffers are unavailable (allocation failed), fall back to a valid scene
    // buffer so descriptors stay valid; the mesh shader runs cullPhase 0 (draw-all) and never
    // reads/writes them in that path.
    const VkBuffer visFallbackBuf = m_scene->meshletBuffer().handle();
    const VkBuffer prevVisBuf = m_meshletVisibility[m_visFrame].handle();
    const VkBuffer currVisBuf = m_meshletVisibility[m_visFrame ^ 1u].handle();
    VkDescriptorBufferInfo prevVisInfo{
        .buffer = (prevVisBuf != VK_NULL_HANDLE) ? prevVisBuf : visFallbackBuf,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo currVisInfo{
        .buffer = (currVisBuf != VK_NULL_HANDLE) ? currVisBuf : visFallbackBuf,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorImageInfo hiZInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = m_hiZPass.sampledView(),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    // Compact instance list from GpuCullPass. When GpuCullPass is not initialized the
    // identity list [0,1,...,N-1] is bound so the task shader's compactInstanceList[gid.x]
    // returns gid.x — equivalent to the CPU-count direct draw path.
    const VkBuffer identFallback =
        m_identityInstanceList.isValid() ? m_identityInstanceList.handle() : m_scene->instanceBuffer().handle();
    const VkBuffer compactInstBuf =
        m_gpuCullPass.isInitialized() ? m_gpuCullPass.compactInstanceListBuffer() : identFallback;
    VkDescriptorBufferInfo compactInstInfo{compactInstBuf, 0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 20> writes{
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &vertexBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &instanceBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &indexBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &meshletBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &meshletVertexBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 5,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &meshletTriangleBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 6,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &materialBufferInfo,
        },
        // Set 1: materials / lights / emissive triangles
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &materialBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &lightBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &emissiveTriangleBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &tileLightCountsInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &tileLightIndicesInfo,
        },
        // Set 1, binding 5: scene TLAS for ray-traced shadows.
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = &tlasWriteInfo,
            .dstSet = m_matSet,
            .dstBinding = 5,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        },
        // Set 2: env importance CDF buffers for transparent-path stochastic env sampling.
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_iblSet,
            .dstBinding = 6,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &envMarginalCdfInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_iblSet,
            .dstBinding = 7,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &envConditionalCdfInfo,
        },
        // Set 0, bindings 7-9: two-pass Hi-Z occlusion culling resources.
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 7,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &prevVisInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 8,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &currVisInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 9,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = &hiZInfo,
        },
        // Set 0, binding 10: compact visible instance list (GpuCullPass output).
        // Task shader reads compactInstanceList[gid.x] to get the instance index.
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 10,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &compactInstInfo,
        },
        // Set 0, binding 11: per-instance object→world transforms (mesh shader).
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 11,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &instanceTransformInfo,
        },
    };
    vkUpdateDescriptorSets(m_ctx->device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Set 3: bindless material textures. Written only when the scene changes (textures are
    // immutable per scene). Each scene texture carries its own view + sampler.
    if (m_scene != m_texturesBoundFor) {
        const auto& sceneTextures = m_scene->textures();
        const std::uint32_t texCount = std::min(static_cast<std::uint32_t>(sceneTextures.size()), kMaxBindlessTextures);
        if (texCount > 0) {
            std::vector<VkDescriptorImageInfo> imageInfos(texCount);
            for (std::uint32_t i = 0; i < texCount; ++i) {
                imageInfos[i] = VkDescriptorImageInfo{
                    .sampler = sceneTextures[i].sampler(),
                    .imageView = sceneTextures[i].image().view(),
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                };
            }
            const VkWriteDescriptorSet texWrite{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = m_textureSet,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = texCount,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = imageInfos.data(),
            };
            vkUpdateDescriptorSets(m_ctx->device, 1, &texWrite, 0, nullptr);
        }
        m_texturesBoundFor = m_scene;
    }

    // Begin dynamic rendering (Vulkan 1.4 - no render pass!)
    VkClearColorValue clearCol{{0.0f, 0.0f, 0.0f, 1.0f}};
    // GBuffer clear: encoded normal (0.5,0.5,0.5) = view-space (0,0,0), roughness=1 → background
    VkClearColorValue clearGbuf{{0.5f, 0.5f, 0.5f, 1.0f}};
    // GI GBuffer clears to all-zero: w=0 (asfloat) is the background sentinel the GI
    // stage tests to skip pixels with no opaque primary surface.
    VkClearColorValue clearGi{{0.0f, 0.0f, 0.0f, 0.0f}};
    VkClearDepthStencilValue clearDepth{1.0f, 0};

    // Camera matrices + push constants (needed before rendering so the two Hi-Z passes and
    // the Hi-Z occlusion test share the same projection).
    sm::float4x4 proj = sm::perspective(sm::radians(m_camera.vfovDeg),
                                        static_cast<float>(m_config.width) / static_cast<float>(m_config.height),
                                        m_camera.nearPlane,
                                        m_camera.farPlane);
    proj[1][1] *= -1.0f;
    if (m_cameraJitterEnabled) {
        proj = applyProjectionJitter(proj, cameraJitterNdc(m_frameSampleIndex, m_config.width, m_config.height));
    }
    const sm::float4x4 view = sm::lookAt(m_camera.position, m_camera.target, m_camera.up);
    const sm::float4x4 viewProj = proj * view;
    // exposure = 1 / (1.2 * 2^EV100) — matches Hyperion's PhysicalCamera calc
    const float exposure = m_camera.physical.exposure();
    // NDC projection scales for the mesh-shader Hi-Z bounding-sphere footprint estimate.
    const float projScaleX = std::abs(proj[0][0]);
    const float projScaleY = std::abs(proj[1][1]);

    const ForwardRenderer::MeshPushConstants pcBase{
        .viewProj = viewProj,
        .view = view,
        .cameraPos = sm::float4(m_camera.position, 1.0f),
        .exposure = exposure,
        .lightCount = m_scene ? m_scene->lightCount() : 0u,
        .emissiveTriangleCount = m_scene ? m_scene->emissiveTriangleCount() : 0u,
        .tilesX = m_tilesX,
        .tilesY = m_tilesY,
        .screenWidth = m_config.width,
        .screenHeight = m_config.height,
        .transparentMaxDepth = m_transparentMaxDepth,
        .frameSampleIndex = m_frameSampleIndex,
        .rngBaseSeed = m_rngBaseSeed,
        .rngFlags = (m_deterministicReplay ? 0x1u : 0u) | (m_rngDebug != 0u ? 0x2u : 0u),
        .cullPhase = 0u,
        .envImportanceWidth = m_envImportanceWidth,
        .envImportanceHeight = m_envImportanceHeight,
        .hiZMipCount = 0u,
        .giEnabled = m_giEnabled,
        // Ray-traced sun shadow: direction toward the dominant IBL light + strength.
        // shadowParams: x = ray tMin (scene-scale bias from camera near plane), y = sky ambient
        // floor, z = env_unit_nits, w = |proj[0][0]| (Hi-Z footprint x scale).
        .sunDirection = sm::float4(m_sunDir, m_hasEnv ? m_sunStrength : 0.0f),
        .shadowParams = sm::float4(std::max(m_camera.nearPlane, 1e-4f), 0.35f, m_envUnitNits, projScaleX),
        // presentationParams.w = |proj[1][1]| (Hi-Z footprint y scale).
        .presentationParams = sm::float4(m_indirectAmbientStrength, 0.0f, m_debugRayHitMode, projScaleY),
    };

    // Attachment builders — CLEAR for the first pass, LOAD for the second (preserving pass-1
    // color/depth/GBuffer so the two Hi-Z passes composite into the same targets).
    auto colorAtt = [&](VkImageView v, VkAttachmentLoadOp op, VkClearColorValue c) {
        return VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = v,
            .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .loadOp = op,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{.color = c},
        };
    };
    auto depthAtt = [&](VkAttachmentLoadOp op) {
        return VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = m_depthTarget.view(),
            .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .loadOp = op,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{.depthStencil = clearDepth},
        };
    };
    auto beginRendering = [&](VkCommandBuffer c, VkAttachmentLoadOp op) {
        const std::array<VkRenderingAttachmentInfo, 3> atts{
            colorAtt(m_config.hdrImageView, op, clearCol),
            colorAtt(m_gbufferTarget.view(), op, clearGbuf),
            colorAtt(m_giBufferTarget.view(), op, clearGi),
        };
        const VkRenderingAttachmentInfo depth = depthAtt(op);
        const VkRenderingInfo info{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea{{0, 0}, {m_config.width, m_config.height}},
            .layerCount = 1,
            .colorAttachmentCount = static_cast<std::uint32_t>(atts.size()),
            .pColorAttachments = atts.data(),
            .pDepthAttachment = &depth,
        };
        vkCmdBeginRendering(c, &info);
    };

    // Draw the environment sky background (only when the scene has an env map). Fills every
    // pixel at the far plane with no depth write; opaque geometry then overwrites it.
    auto drawSky = [&](VkCommandBuffer c) {
        if (!m_hasEnv || m_skyPipeline == VK_NULL_HANDLE) {
            return;
        }
        sm::float4x4 skyProj = sm::perspective(sm::radians(m_camera.vfovDeg),
                                               static_cast<float>(m_config.width) / static_cast<float>(m_config.height),
                                               m_camera.nearPlane,
                                               m_camera.farPlane);
        skyProj[1][1] *= -1.0f;
        if (m_cameraJitterEnabled) {
            skyProj =
                applyProjectionJitter(skyProj, cameraJitterNdc(m_frameSampleIndex, m_config.width, m_config.height));
        }
        const sm::float4x4 skyView = sm::lookAt(m_camera.position, m_camera.target, m_camera.up);
        const sm::float4x4 skyViewProj = skyProj * skyView;
        const SkyPushConstants skyPc{
            .invViewProj = sm::inverse(skyViewProj),
            .cameraPos = sm::float4(m_camera.position, 1.0f),
            .exposure = m_camera.physical.exposure(),
            .hasEnv = 1u,
            .envScale = m_envUnitNits,
            ._pad1 = 0u,
        };
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyPipeline);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyPipelineLayout, 0, 1, &m_iblSet, 0, nullptr);
        vkCmdPushConstants(c,
                           m_skyPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(skyPc),
                           &skyPc);
        vkCmdDraw(c, 3, 1, 0, 0);
    };

    const std::array<VkDescriptorSet, 4> descSets{m_meshSet, m_matSet, m_iblSet, m_textureSet};
    auto bindMeshSets = [&](VkCommandBuffer c) {
        vkCmdBindDescriptorSets(
            c, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 4, descSets.data(), 0, nullptr);
    };
    auto drawTransparent = [&](VkCommandBuffer c) {
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineTransparent);
        bindMeshSets(c);
        auto pc = pcBase;
        pc.presentationParams.y = 1.0f; // transparent routing in the task shader
        pc.cullPhase = 0u;              // transparent geometry is not occlusion-culled
        pc.hiZMipCount = 0u;
        vkCmdPushConstants(c,
                           m_pipelineLayout,
                           VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(MeshPushConstants),
                           &pc);
        const std::uint32_t instCnt = m_scene->instanceCount();
        if (m_gpuCullPass.isInitialized() && m_dgcLayout != VK_NULL_HANDLE) {
            const VkGeneratedCommandsPipelineInfoEXT dgcPipelineInfo{
                .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,
                .pipeline = m_graphicsPipelineTransparent,
            };
            const VkGeneratedCommandsInfoEXT dgcInfo{
                .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,
                .pNext = &dgcPipelineInfo,
                .shaderStages =
                    VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                .indirectExecutionSet = VK_NULL_HANDLE,
                .indirectCommandsLayout = m_dgcLayout,
                .indirectAddress = m_gpuCullPass.indirectDrawAddress(),
                .indirectAddressSize = 12u, // one VkDrawMeshTasksIndirectCommandEXT
                .preprocessAddress = m_dgcPreprocessAddr,
                .preprocessSize = m_dgcPreprocessSize,
                .maxSequenceCount = 1u,    // single GPU-generated draw
                .sequenceCountAddress = 0, // fixed count of 1
                .maxDrawCount = 1,
            };
            vkCmdExecuteGeneratedCommandsEXT(c, VK_FALSE, &dgcInfo);
        } else if (m_gpuCullPass.isInitialized() && vkCmdDrawMeshTasksIndirectEXT != nullptr) {
            vkCmdDrawMeshTasksIndirectEXT(c, m_gpuCullPass.indirectDrawBuffer(), 0, 1, 12u);
        } else {
            vkCmdDrawMeshTasksEXT(c, instCnt, 1, 1);
        }
    };

    // Set dynamic state (shared by both passes).
    VkViewport vp{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(m_config.width),
        .height = static_cast<float>(m_config.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{{0, 0}, {m_config.width, m_config.height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const std::uint32_t instanceCount = m_scene->instanceCount();
    const bool canDraw = (instanceCount > 0) && (vkCmdDrawMeshTasksEXT != nullptr);

    // GPU-driven frustum cull (must run outside dynamic rendering — compute cannot run
    // inside a render pass). Dispatch before the Hi-Z passes and barrier results.
    if (canDraw && m_gpuCullPass.isInitialized()) {
        m_gpuCullPass.dispatch(
            cmd, m_scene->instanceBuffer().handle(), m_scene->instanceBoundsBuffer().handle(), instanceCount, viewProj);
        // Barrier: compute writes → task shader reads (compactInstanceList) +
        //          indirect command reads (indirectDrawBuf).
        // The indirectDrawBuf doubles as the DGC sequenceCountAddress, which is consumed at
        // the COMMAND_PREPROCESS stage by vkCmdExecuteGeneratedCommandsEXT (implicit
        // preprocessing) — not at DRAW_INDIRECT. Cover both stages so the count is fully
        // written by the cull compute before DGC preprocessing reads it; omitting
        // COMMAND_PREPROCESS lets the driver read a partial sequence count, dropping the
        // instances near the tail of the compact list (far geometry) non-obviously.
        const std::array cullBarriers{
            VkBufferMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                .buffer = m_gpuCullPass.compactInstanceListBuffer(),
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            },
            VkBufferMemoryBarrier2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT,
                .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT,
                .buffer = m_gpuCullPass.indirectDrawBuffer(),
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            },
        };
        const VkDependencyInfo cullDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = static_cast<std::uint32_t>(cullBarriers.size()),
            .pBufferMemoryBarriers = cullBarriers.data(),
        };
        vkCmdPipelineBarrier2(cmd, &cullDep);
    }

    const bool twoPass = canDraw && visReady && !m_forceSinglePass;
    const std::uint32_t hiZMip =
        (twoPass && m_hiZPass.isInitialized() && m_hiZTestEnabled && !m_hiZDebugDisabled) ? m_hiZPass.mipLevels() : 0u;

    if (!twoPass) {
        // Fallback: single pass drawing all meshlets (original behavior). Also covers the
        // vkCmdDrawMeshTasksEXT-missing case (draws nothing but keeps attachments cleared).
        if (instanceCount > 0 && vkCmdDrawMeshTasksEXT == nullptr) {
            Logger::error("vkCmdDrawMeshTasksEXT is NULL - mesh shader extension not loaded!");
        }
        m_hiZPass.prepareForSampling(cmd);
        beginRendering(cmd, VK_ATTACHMENT_LOAD_OP_CLEAR);
        drawSky(cmd);
        if (canDraw) {
            drawOpaque(cmd, pcBase, /*cullPhase*/ 0u, /*hiZMipCount*/ 0u);
            drawTransparent(cmd);
        }
        vkCmdEndRendering(cmd);
        m_hdrFirstUse = false;
        return;
    }

    // ---- Two-pass Hi-Z occlusion culling ----
    // Ensure the Hi-Z image is sampleable in both passes (first-use layout transition).
    m_hiZPass.prepareForSampling(cmd);

    // Reset the current-frame visibility (and previous on the first frame for a scene).
    vkCmdFillBuffer(cmd, currVisBuf, 0, VK_WHOLE_SIZE, 0u);
    if (m_visClearPrev) {
        vkCmdFillBuffer(cmd, prevVisBuf, 0, VK_WHOLE_SIZE, 0u);
        m_visClearPrev = false;
    }
    const std::array visFillBarriers{
        VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .buffer = currVisBuf,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        },
        VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .buffer = prevVisBuf,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        },
    };
    const VkDependencyInfo visFillDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = static_cast<std::uint32_t>(visFillBarriers.size()),
        .pBufferMemoryBarriers = visFillBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &visFillDep);

    // Pass 1: redraw only meshlets visible last frame → current-frame initial depth.
    beginRendering(cmd, VK_ATTACHMENT_LOAD_OP_CLEAR);
    drawSky(cmd);
    drawOpaque(cmd, pcBase, /*cullPhase*/ 1u, /*hiZMipCount*/ 0u);
    vkCmdEndRendering(cmd);

    // Build the current-frame Hi-Z pyramid from pass-1 depth (skipped when the test is off).
    if (hiZMip > 0u) {
        const VkImageMemoryBarrier2 depthToRead{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_depthTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
        };
        const VkDependencyInfo depthReadDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &depthToRead,
        };
        vkCmdPipelineBarrier2(cmd, &depthReadDep);

        m_hiZPass.build(cmd, m_depthTarget.view());

        // Restore depth to a write attachment for pass 2 (contents preserved for LOAD).
        const VkImageMemoryBarrier2 depthToAttach{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_depthTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
        };
        const VkDependencyInfo depthAttachDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &depthToAttach,
        };
        vkCmdPipelineBarrier2(cmd, &depthAttachDep);
    }

    // Make pass-1 visibility writes available to pass-2 visibility writes (WAW safety).
    const VkBufferMemoryBarrier2 visRWBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .buffer = currVisBuf,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    const VkDependencyInfo visRWDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &visRWBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &visRWDep);

    // Pass 2: remaining meshlets, occlusion-tested against the current-frame Hi-Z, then the
    // transparent pass — composited over pass-1 targets (LOAD, no clear).
    beginRendering(cmd, VK_ATTACHMENT_LOAD_OP_LOAD);
    drawOpaque(cmd, pcBase, /*cullPhase*/ 2u, hiZMip);
    drawTransparent(cmd);
    vkCmdEndRendering(cmd);

    // Swap ping-pong: this frame's visibility becomes next frame's "previous".
    m_visFrame ^= 1u;
    m_hdrFirstUse = false;
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
