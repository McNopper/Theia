#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "theia/renderer/GiPass.hpp"

#include <array>
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
}

bool GiPass::createDescriptors() {
    const std::array<VkDescriptorSetLayoutBinding, 13> bindings{{
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
    }};

    const VkDescriptorSetLayoutCreateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &setInfo, nullptr, &m_setLayout) != VK_SUCCESS) {
        Logger::error("GiPass: failed to create descriptor set layout");
        return false;
    }

    const std::array<VkDescriptorPoolSize, 5> poolSizes{{
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7},
    }};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
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
    const VkDescriptorBufferInfo marginalInfo{marginalCdf, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo conditionalInfo{conditionalCdf, 0, VK_WHOLE_SIZE};

    const std::array<VkWriteDescriptorSet, 13> writes{{
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
    }};
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void GiPass::record(VkCommandBuffer cmd, const FrameParams& params) {
    if (m_pipeline == VK_NULL_HANDLE || params.scene == nullptr || m_cfg.hdrImage == VK_NULL_HANDLE) {
        return;
    }

    updateDescriptors(params);

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
