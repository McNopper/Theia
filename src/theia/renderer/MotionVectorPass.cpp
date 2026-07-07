#include "theia/renderer/MotionVectorPass.hpp"

#include <array>

#include "harmonia/core/Barrier.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "theia/renderer/ShaderPath.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

MotionVectorPass::~MotionVectorPass() {
    shutdown();
}

bool MotionVectorPass::initialize(const DeviceContext& ctx, const Config& cfg, const char* spvName) {
    shutdown();
    m_ctx = &ctx;
    m_cfg = cfg;
    m_firstUse = true;

    if (cfg.width == 0 || cfg.height == 0 || cfg.giBufferImage == VK_NULL_HANDLE ||
        cfg.giBufferView == VK_NULL_HANDLE) {
        Logger::error("MotionVectorPass: invalid Config");
        return false;
    }

    if (!createImage()) {
        return false;
    }
    if (!createPipeline(spvName)) {
        return false;
    }
    return true;
}

void MotionVectorPass::shutdown() {
    if (m_ctx == nullptr) {
        return;
    }
    m_motionVectorImage = {};
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
    m_cfg = {};
    m_firstUse = true;
}

bool MotionVectorPass::createImage() noexcept {
    auto img = Image::create(*m_ctx,
                              {m_cfg.width, m_cfg.height},
                              VK_FORMAT_R32G32_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT,
                              "theia.motionVectors");
    if (!img) {
        Logger::error("MotionVectorPass: failed to create motion vector image: VkResult {}",
                      static_cast<int>(img.error()));
        return false;
    }
    m_motionVectorImage = std::move(*img);
    return true;
}

bool MotionVectorPass::createPipeline(const char* spvName) noexcept {
    // Binding 0: GI G-buffer (sampled image — integer Load, no sampler needed)
    // Binding 1: motion vector output (storage image, R32G32F)
    // Binding 2: prevInstanceTransforms (storage buffer, one float4x4 per instance)
    const std::array bindings{
        VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
    };
    const VkDescriptorSetLayoutCreateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &setInfo, nullptr, &m_setLayout) != VK_SUCCESS) {
        Logger::error("MotionVectorPass: failed to create descriptor set layout");
        return false;
    }

    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(MotionVectorPC),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &m_setLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        Logger::error("MotionVectorPass: failed to create pipeline layout");
        return false;
    }

    auto module = harmonia::createShaderModule(m_ctx->device, shaderPath(spvName));
    if (!module) {
        Logger::error("MotionVectorPass: cannot load shader: {}", spvName);
        return false;
    }

    const VkComputePipelineCreateInfo pipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = *module,
                  .pName = "main"},
        .layout = m_pipelineLayout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };
    const VkResult res =
        vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_pipeline);
    vkDestroyShaderModule(m_ctx->device, *module, nullptr);
    if (res != VK_SUCCESS) {
        Logger::error("MotionVectorPass: failed to create compute pipeline");
        return false;
    }

    m_ctx->setDebugName(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(m_pipeline),
                        "theia.motionVectors.pipeline");
    return true;
}

void MotionVectorPass::record(VkCommandBuffer cmd, const FrameParams& params) noexcept {
    if (m_pipeline == VK_NULL_HANDLE || !m_motionVectorImage.isValid()) {
        return;
    }
    // Guard: prev-instance-transform buffer must be valid before we can push the descriptor.
    if (params.prevInstanceTransformBuffer == VK_NULL_HANDLE) {
        return;
    }

    // Transition the output image: UNDEFINED → GENERAL on first use.
    const VkImageMemoryBarrier2 preMvBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask  = m_firstUse ? VK_PIPELINE_STAGE_2_NONE
                                    : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = m_firstUse ? 0U : static_cast<VkAccessFlags2>(VK_ACCESS_2_SHADER_WRITE_BIT),
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .oldLayout     = m_firstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
        .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image         = m_motionVectorImage.handle(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    const VkDependencyInfo preDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &preMvBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &preDep);
    m_firstUse = false;

    // Push descriptors: binding 0 = giBuffer (sampled, SHADER_READ_ONLY_OPTIMAL after GiPass),
    //                   binding 1 = motion vector output (storage, GENERAL),
    //                   binding 2 = prevInstanceTransforms (storage buffer).
    const VkDescriptorImageInfo giInfo{
        .sampler     = VK_NULL_HANDLE,
        .imageView   = m_cfg.giBufferView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo mvInfo{
        .sampler     = VK_NULL_HANDLE,
        .imageView   = m_motionVectorImage.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorBufferInfo prevTransformInfo{
        .buffer = params.prevInstanceTransformBuffer,
        .offset = 0,
        .range  = VK_WHOLE_SIZE,
    };
    const std::array writes{
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = &giInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &mvInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &prevTransformInfo,
        },
    };

    const MotionVectorPC pc{
        .curViewProj    = params.curViewProj,
        .prevViewProj   = params.prevViewProj,
        .invCurViewProj = params.invCurViewProj,
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout,
                           0, static_cast<uint32_t>(writes.size()), writes.data());
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    const uint32_t gx = (m_cfg.width  + 7u) / 8u;
    const uint32_t gy = (m_cfg.height + 7u) / 8u;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Make motion-vector writes visible to subsequent COMPUTE_SHADER reads
    // (the denoiser's temporal accumulation pass reads gMotionVectors).
    const VkImageMemoryBarrier2 postMvBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image         = m_motionVectorImage.handle(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    const VkDependencyInfo postDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &postMvBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &postDep);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
