#include "theia/renderer/TaaPass.hpp"

#include <array>
#include <cstdint>

#include "harmonia/core/Barrier.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "theia/renderer/ShaderPath.hpp"
#include "theia/renderer/TaaPass.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

TaaPass::~TaaPass() {
    shutdown();
}

bool TaaPass::initialize(const DeviceContext& ctx, const Config& cfg, const char* spvName) {
    shutdown();
    m_ctx = &ctx;
    m_cfg = cfg;
    m_firstUse = true;
    m_frameCount = 0;

    if (cfg.width == 0 || cfg.height == 0 || cfg.hdrImage == VK_NULL_HANDLE || cfg.hdrView == VK_NULL_HANDLE ||
        cfg.motionVecView == VK_NULL_HANDLE) {
        Logger::error("TaaPass: invalid Config");
        return false;
    }

    if (!createImages())
        return false;
    if (!createSampler())
        return false;
    if (!createPipeline(spvName))
        return false;
    return true;
}

void TaaPass::shutdown() {
    if (m_ctx == nullptr)
        return;

    m_history = {};
    m_taaOutput = {};

    if (m_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_ctx->device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx->device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipeLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_ctx->device, m_pipeLayout, nullptr);
        m_pipeLayout = VK_NULL_HANDLE;
    }
    if (m_setLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_setLayout, nullptr);
        m_setLayout = VK_NULL_HANDLE;
    }
    m_ctx = nullptr;
    m_cfg = {};
    m_firstUse = true;
    m_frameCount = 0;
}

bool TaaPass::createImages() noexcept {
    // History buffer: read each frame as a sampled image; written by vkCmdCopyImage each frame.
    auto hist = Image::create(*m_ctx,
                              {m_cfg.width, m_cfg.height},
                              VK_FORMAT_R32G32B32A32_SFLOAT,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT,
                              "theia.taa.history");
    if (!hist) {
        Logger::error("TaaPass: failed to create history image: VkResult {}", static_cast<int>(hist.error()));
        return false;
    }
    m_history = std::move(*hist);

    // TAA output: written by compute dispatch, then copied to history and HDR each frame.
    auto out = Image::create(*m_ctx,
                             {m_cfg.width, m_cfg.height},
                             VK_FORMAT_R32G32B32A32_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             "theia.taa.output");
    if (!out) {
        Logger::error("TaaPass: failed to create TAA output image: VkResult {}", static_cast<int>(out.error()));
        return false;
    }
    m_taaOutput = std::move(*out);
    return true;
}

bool TaaPass::createSampler() noexcept {
    const VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .compareEnable = VK_FALSE,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    if (vkCreateSampler(m_ctx->device, &info, nullptr, &m_sampler) != VK_SUCCESS) {
        Logger::error("TaaPass: failed to create sampler");
        return false;
    }
    return true;
}

bool TaaPass::createPipeline(const char* spvName) noexcept {
    // binding 0: hdrCurrent   — COMBINED_IMAGE_SAMPLER (Texture2D<float4>)
    // binding 1: motionVectors — COMBINED_IMAGE_SAMPLER (Texture2D<float2>)
    // binding 2: history       — COMBINED_IMAGE_SAMPLER (Texture2D<float4>)
    // binding 3: taaOutput     — STORAGE_IMAGE          (RWTexture2D<float4>)
    const std::array bindings{
        VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
    };
    const VkDescriptorSetLayoutCreateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &setInfo, nullptr, &m_setLayout) != VK_SUCCESS) {
        Logger::error("TaaPass: failed to create descriptor set layout");
        return false;
    }

    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(TaaPushConstants),
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
    if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &m_pipeLayout) != VK_SUCCESS) {
        Logger::error("TaaPass: failed to create pipeline layout");
        return false;
    }

    auto module = harmonia::createShaderModule(m_ctx->device, shaderPath(spvName));
    if (!module) {
        Logger::error("TaaPass: cannot load shader: {}", spvName);
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
        .layout = m_pipeLayout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };
    const VkResult res = vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_pipeline);
    vkDestroyShaderModule(m_ctx->device, *module, nullptr);
    if (res != VK_SUCCESS) {
        Logger::error("TaaPass: failed to create compute pipeline");
        return false;
    }
    m_ctx->setDebugName(VK_OBJECT_TYPE_PIPELINE, m_pipeline, "theia.taa.pipeline");
    return true;
}

void TaaPass::record(VkCommandBuffer cmd, const FrameParams& params) noexcept {
    if (m_pipeline == VK_NULL_HANDLE || !m_history.isValid() || !m_taaOutput.isValid()) {
        return;
    }

    // On the first GPU use after (re-)initialization, transition both scratch images
    // from UNDEFINED to GENERAL.  Force shader firstFrame=1 so the empty history is
    // never blended in — the output is simply the current frame.
    const bool first = m_firstUse || params.firstFrame;
    if (m_firstUse) {
        const std::array<VkImageMemoryBarrier2, 2> initBarriers{{
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
             .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
             .srcAccessMask = VK_ACCESS_2_NONE,
             .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
             .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
             .newLayout = VK_IMAGE_LAYOUT_GENERAL,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .image = m_history.handle(),
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
             .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
             .srcAccessMask = VK_ACCESS_2_NONE,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
             .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
             .newLayout = VK_IMAGE_LAYOUT_GENERAL,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .image = m_taaOutput.handle(),
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        }};
        const VkDependencyInfo initDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<std::uint32_t>(initBarriers.size()),
            .pImageMemoryBarriers = initBarriers.data(),
        };
        vkCmdPipelineBarrier2(cmd, &initDep);
    }

    // Ensure GiPass's HDR writes (compute shader) are visible as shader reads for TAA.
    const VkImageMemoryBarrier2 hdrReadBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = m_cfg.hdrImage,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    const VkDependencyInfo preDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &hdrReadBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &preDep);

    // Descriptor writes for push-descriptor dispatch.
    const VkDescriptorImageInfo hdrInfo{
        .sampler = m_sampler,
        .imageView = m_cfg.hdrView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo mvInfo{
        .sampler = m_sampler,
        .imageView = m_cfg.motionVecView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo histInfo{
        .sampler = m_sampler,
        .imageView = m_history.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo outInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = m_taaOutput.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const std::array writes{
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &hdrInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &mvInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &histInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &outInfo,
        },
    };
    const TaaPushConstants pc{
        .screenWidth = m_cfg.width,
        .screenHeight = m_cfg.height,
        .alpha = params.alpha,
        .firstFrame = first ? 1u : 0u,
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdPushDescriptorSet(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeLayout, 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    const std::uint32_t gx = (m_cfg.width + 7u) / 8u;
    const std::uint32_t gy = (m_cfg.height + 7u) / 8u;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Make TAA output visible as a transfer source for the two copies that follow.
    const VkImageMemoryBarrier2 outReadBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = m_taaOutput.handle(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    const VkDependencyInfo postDispatchDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &outReadBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &postDispatchDep);

    // Copy taaOutput → history (so the next frame can blend against it)
    // and taaOutput → HDR (so the Harmonia denoiser receives the stabilised signal).
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {m_cfg.width, m_cfg.height, 1},
    };
    vkCmdCopyImage(
        cmd, m_taaOutput.handle(), VK_IMAGE_LAYOUT_GENERAL, m_history.handle(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    vkCmdCopyImage(
        cmd, m_taaOutput.handle(), VK_IMAGE_LAYOUT_GENERAL, m_cfg.hdrImage, VK_IMAGE_LAYOUT_GENERAL, 1, &region);

    // Make history visible for next frame's compute read; make HDR visible for the denoiser.
    const std::array<VkImageMemoryBarrier2, 2> postCopyBarriers{{
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = m_history.handle(),
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = m_cfg.hdrImage,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
    }};
    const VkDependencyInfo postCopyDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<std::uint32_t>(postCopyBarriers.size()),
        .pImageMemoryBarriers = postCopyBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &postCopyDep);

    m_firstUse = false;
    ++m_frameCount;
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
