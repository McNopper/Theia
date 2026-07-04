#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "theia/renderer/GiPass.hpp"

#include <algorithm>
#include <array>
#include <vector>
#include <harmonia/core/Logger.hpp>
#include <harmonia/core/ShaderModule.hpp>
#include <theia/renderer/ShaderPath.hpp>
#include <theia/scene/Scene.hpp>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

namespace {

[[nodiscard]] VkImageMemoryBarrier2 imgBarrier(VkImage image,
                                               VkImageLayout oldLayout,
                                               VkImageLayout newLayout,
                                               VkPipelineStageFlags2 srcStage,
                                               VkAccessFlags2 srcAccess,
                                               VkPipelineStageFlags2 dstStage,
                                               VkAccessFlags2 dstAccess) {
    return VkImageMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
}

} // namespace

GiPass::~GiPass() {
    shutdown();
}

bool GiPass::initialize(const DeviceContext& ctx, const Config& cfg, const char* giSpv) {
    m_ctx = &ctx;
    m_cfg = cfg;
    m_hdrFirstUse = true;
    m_boundScene = nullptr;
    m_texturesBoundFor = nullptr;
    m_boundEnvMapView = VK_NULL_HANDLE;
    m_boundEnvSampler = VK_NULL_HANDLE;
    m_boundEnvMarginalCdf = VK_NULL_HANDLE;
    m_boundEnvConditionalCdf = VK_NULL_HANDLE;
    m_boundGradientVarianceView = VK_NULL_HANDLE;
    m_dummyGradientReady = false;

    // A3(b): 1×1 R32G32F placeholder for binding 15 when no A-SVGF gradient/variance
    // guide is available. Keeps the descriptor valid; hasGradientVariance=0 gates reads.
    auto dummyGrad = Image::create(ctx,
                                   {1U, 1U},
                                   VK_FORMAT_R32G32_SFLOAT,
                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                   "theia.gi.dummyGradientVariance");
    if (!dummyGrad) {
        Logger::error("GiPass: failed to create dummy gradient/variance image: VkResult {}",
                      static_cast<int>(dummyGrad.error()));
        return false;
    }
    m_dummyGradientVariance = std::move(*dummyGrad);

    if (!createDescriptors()) {
        return false;
    }
    if (!createPipeline(giSpv)) {
        return false;
    }
    return true;
}

void GiPass::shutdown() {
    if (m_ctx == nullptr) {
        return;
    }
    const VkDevice device = m_ctx->device;
    m_dummyGradientVariance = {};
    m_dummyGradientReady = false;
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
        m_set = VK_NULL_HANDLE;
    }
    if (m_setLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_setLayout, nullptr);
        m_setLayout = VK_NULL_HANDLE;
    }
    m_boundScene = nullptr;
    m_texturesBoundFor = nullptr;
    m_boundEnvMapView = VK_NULL_HANDLE;
    m_boundEnvSampler = VK_NULL_HANDLE;
    m_boundEnvMarginalCdf = VK_NULL_HANDLE;
    m_boundEnvConditionalCdf = VK_NULL_HANDLE;
    m_boundGradientVarianceView = VK_NULL_HANDLE;
}

bool GiPass::createDescriptors() {
    const std::array<VkDescriptorSetLayoutBinding, 16> bindings{{
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // TLAS
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},              // hdr (RW)
        {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},              // giBuffer
        {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},              // gbuffer
        {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},              // envMap
        {5, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                    // env sampler
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // materials
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // vertices
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // instances
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // indices
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},            // emissive tris
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},            // marginal CDF
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},            // conditional CDF
        {13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // textures
        {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},            // emissive power CDF
        {15, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},             // A3(b) gradient/variance
    }};

    constexpr VkDescriptorBindingFlags kUpdateAfterBind = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    constexpr VkDescriptorBindingFlags kTextureFlags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    const std::array<VkDescriptorBindingFlags, 16> bindingFlags{
        kUpdateAfterBind, kUpdateAfterBind, kUpdateAfterBind, kUpdateAfterBind, kUpdateAfterBind, kUpdateAfterBind,
        kUpdateAfterBind, kUpdateAfterBind, kUpdateAfterBind, kUpdateAfterBind, kUpdateAfterBind, kUpdateAfterBind,
        kUpdateAfterBind, kTextureFlags, kUpdateAfterBind, kUpdateAfterBind};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };

    const VkDescriptorSetLayoutCreateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &bindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &setInfo, nullptr, &m_setLayout) != VK_SUCCESS) {
        Logger::error("GiPass: failed to create descriptor set layout");
        return false;
    }

    const std::array<VkDescriptorPoolSize, 6> poolSizes{{
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures},
    }};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &m_pool) != VK_SUCCESS) {
        Logger::error("GiPass: failed to create descriptor pool");
        return false;
    }

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_setLayout,
    };
    if (vkAllocateDescriptorSets(m_ctx->device, &allocInfo, &m_set) != VK_SUCCESS) {
        Logger::error("GiPass: failed to allocate descriptor set");
        return false;
    }
    return true;
}

bool GiPass::createPipeline(const char* giSpv) {
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(GiPushConstants),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_setLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        Logger::error("GiPass: failed to create pipeline layout");
        return false;
    }

    auto module = harmonia::createShaderModule(m_ctx->device, shaderPath(giSpv));
    if (!module) {
        Logger::error("GiPass: cannot load shader: {}", giSpv);
        return false;
    }

    const VkComputePipelineCreateInfo pipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = *module,
                  .pName = "main"},
        .layout = m_pipelineLayout,
    };
    const VkResult res = vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_pipeline);
    vkDestroyShaderModule(m_ctx->device, *module, nullptr);
    if (res != VK_SUCCESS) {
        Logger::error("GiPass: failed to create compute pipeline");
        return false;
    }
    return true;
}

void GiPass::updateDescriptors(const FrameParams& params) {
    const Scene* scene = params.scene;

    VkAccelerationStructureKHR tlas = scene->tlas();
    const VkWriteDescriptorSetAccelerationStructureKHR tlasWriteInfo{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlas,
    };

    const VkDescriptorImageInfo hdrInfo{
        .imageView = m_cfg.hdrView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo giBufferInfo{
        .imageView = m_cfg.giBufferView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo gbufferInfo{
        .imageView = m_cfg.gbufferView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo envInfo{
        .imageView = params.envMapView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo envSamplerInfo{.sampler = params.envSampler};

    // Env CDF buffers may be absent (no env map); bind the material buffer as a harmless
    // placeholder so the descriptor stays valid — the shader gates all reads on hasEnvMap.
    const VkBuffer cdfFallback = scene->materialBuffer().handle();
    const VkBuffer marginalCdf = (params.envMarginalCdf != VK_NULL_HANDLE) ? params.envMarginalCdf : cdfFallback;
    const VkBuffer conditionalCdf = (params.envConditionalCdf != VK_NULL_HANDLE) ? params.envConditionalCdf : cdfFallback;

    const VkDescriptorBufferInfo materialsInfo{scene->materialBuffer().handle(), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo verticesInfo{scene->vertexBuffer().handle(), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo instancesInfo{scene->instanceBuffer().handle(), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo indicesInfo{scene->indexBuffer().handle(), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo emissiveInfo{scene->emissiveTriangleBuffer().handle(), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo emissiveCdfInfo{scene->emissiveCdfBuffer().handle(), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo marginalInfo{marginalCdf, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo conditionalInfo{conditionalCdf, 0, VK_WHOLE_SIZE};

    // A3(b): A-SVGF gradient/variance guide, or the 1×1 dummy when unavailable.
    // Both live in VK_IMAGE_LAYOUT_GENERAL (the real guide is a denoiser storage image).
    const VkDescriptorImageInfo gradientVarianceInfo{
        .imageView = (params.gradientVarianceView != VK_NULL_HANDLE) ? params.gradientVarianceView
                                                                     : m_dummyGradientVariance.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    const std::array<VkWriteDescriptorSet, 15> writes{{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &tlasWriteInfo, m_set, 0, 0, 1,
         VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 1, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 2, 0, 1,
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &giBufferInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 3, 0, 1,
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gbufferInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 4, 0, 1,
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &envInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 5, 0, 1,
         VK_DESCRIPTOR_TYPE_SAMPLER, &envSamplerInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 6, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialsInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 7, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &verticesInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 8, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instancesInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 9, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &indicesInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 10, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &emissiveInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 11, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &marginalInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 12, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &conditionalInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 14, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &emissiveCdfInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_set, 15, 0, 1,
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gradientVarianceInfo, nullptr, nullptr},
    }};

    std::vector<VkWriteDescriptorSet> allWrites(writes.begin(), writes.end());
    if (m_texturesBoundFor != scene) {
        const auto& sceneTextures = scene->textures();
        const uint32_t texCount = std::min(static_cast<uint32_t>(sceneTextures.size()), kMaxBindlessTextures);
        if (texCount > 0) {
            std::vector<VkDescriptorImageInfo> imageInfos(texCount);
            for (uint32_t i = 0; i < texCount; ++i) {
                imageInfos[i] = VkDescriptorImageInfo{
                    .sampler = sceneTextures[i].sampler(),
                    .imageView = sceneTextures[i].image().view(),
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                };
            }
            allWrites.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = m_set,
                .dstBinding = 13,
                .dstArrayElement = 0,
                .descriptorCount = texCount,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = imageInfos.data(),
            });
        }
        m_texturesBoundFor = scene;
    }
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(allWrites.size()), allWrites.data(), 0, nullptr);
}

bool GiPass::descriptorsDirty(const FrameParams& params) const {
    return m_boundScene != params.scene || m_boundEnvMapView != params.envMapView || m_boundEnvSampler != params.envSampler ||
           m_boundEnvMarginalCdf != params.envMarginalCdf || m_boundEnvConditionalCdf != params.envConditionalCdf ||
           m_boundGradientVarianceView != params.gradientVarianceView;
}

void GiPass::record(VkCommandBuffer cmd, const FrameParams& params) {
    if (m_pipeline == VK_NULL_HANDLE || params.scene == nullptr || m_cfg.hdrImage == VK_NULL_HANDLE) {
        return;
    }

    if (descriptorsDirty(params)) {
        updateDescriptors(params);
        m_boundScene = params.scene;
        m_boundEnvMapView = params.envMapView;
        m_boundEnvSampler = params.envSampler;
        m_boundEnvMarginalCdf = params.envMarginalCdf;
        m_boundEnvConditionalCdf = params.envConditionalCdf;
        m_boundGradientVarianceView = params.gradientVarianceView;
    }

    // One-time layout transition for the dummy gradient/variance placeholder: it is never
    // written, but the descriptor declares GENERAL, so move it out of UNDEFINED once.
    if (!m_dummyGradientReady && m_dummyGradientVariance.isValid()) {
        m_dummyGradientVariance.transition(cmd,
                                           VK_IMAGE_LAYOUT_UNDEFINED,
                                           VK_IMAGE_LAYOUT_GENERAL,
                                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                           VK_ACCESS_2_NONE,
                                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                           VK_ACCESS_2_SHADER_READ_BIT);
        m_dummyGradientReady = true;
    }

    // ---- Barriers: forward attachments -> compute inputs/outputs ----
    const std::array<VkImageMemoryBarrier2, 3> preBarriers{{
        imgBarrier(m_cfg.giBufferImage,
                   VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT),
        imgBarrier(m_cfg.gbufferImage,
                   VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT),
        imgBarrier(m_cfg.hdrImage,
                   VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT),
    }};
    const VkDependencyInfo preDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(preBarriers.size()),
        .pImageMemoryBarriers = preBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &preDep);
    m_hdrFirstUse = false;

    const GiPushConstants pc{
        .view = params.viewTransposed,
        .cameraPos = glm::vec4(params.cameraPos, 1.0f),
        .exposure = params.exposure,
        .frameSampleIndex = params.frameSampleIndex,
        .rngBaseSeed = params.rngBaseSeed,
        .emissiveTriangleCount = params.scene->emissiveTriangleCount(),
        .envImportanceWidth = params.envImportanceWidth,
        .envImportanceHeight = params.envImportanceHeight,
        .hasEnvMap = params.hasEnvMap ? 1u : 0u,
        .envLuminanceScale = params.envLuminanceScale,
        .maxDepth = std::max(1u, params.maxDepth),
        .screenWidth = m_cfg.width,
        .screenHeight = m_cfg.height,
        ._pad0 = 0u,
        .a3RegularizationEnabled = params.useA3Regularization ? 1u : 0u,
        .a3ChromaticImportanceEnabled = params.useA3ChromaticImportance ? 1u : 0u,
        .hasGradientVariance = (params.gradientVarianceView != VK_NULL_HANDLE) ? 1u : 0u,
        ._pad1 = 0u,
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_set, 0, nullptr);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    const uint32_t gx = (m_cfg.width + 7u) / 8u;
    const uint32_t gy = (m_cfg.height + 7u) / 8u;
    vkCmdDispatch(cmd, gx, gy, 1);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
