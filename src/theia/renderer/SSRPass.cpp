#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "theia/renderer/SSRPass.hpp"

#include <algorithm>
#include <array>
#include <harmonia/core/Logger.hpp>
#include <harmonia/core/ShaderModule.hpp>
#include <theia/renderer/ShaderPath.hpp>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

static constexpr float kRoughnessMax = 0.55f; // must match ssr.comp.slang

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] VkImageView createMipView(const DeviceContext& ctx, const Image& image, uint32_t mipLevel) {
    const VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image.handle(),
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = image.format(),
        .components{
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange{
            VK_IMAGE_ASPECT_COLOR_BIT,
            mipLevel,
            1,
            0,
            1,
        },
    };

    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create mip image view {}", mipLevel);
        return VK_NULL_HANDLE;
    }
    return view;
}

[[nodiscard]] VkShaderModule loadSpv(VkDevice device, const char* filename) {
    auto module = harmonia::createShaderModule(device, shaderPath(filename));
    if (!module) {
        Logger::error("SSRPass: cannot load shader: {}", filename);
        return VK_NULL_HANDLE;
    }
    return *module;
}

[[nodiscard]] VkImageMemoryBarrier2 imgBarrier(VkImage image,
                                               VkImageLayout oldLayout,
                                               VkImageLayout newLayout,
                                               VkPipelineStageFlags2 srcStage,
                                               VkAccessFlags2 srcAccess,
                                               VkPipelineStageFlags2 dstStage,
                                               VkAccessFlags2 dstAccess,
                                               VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
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
        .subresourceRange = {aspect, 0, 1, 0, 1},
    };
}

void pipelineBarrier(VkCommandBuffer cmd, std::initializer_list<VkImageMemoryBarrier2> barriers) {
    const VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers = barriers.begin(),
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

SSRPass::~SSRPass() {
    shutdown();
}

bool SSRPass::initialize(const DeviceContext& ctx,
                         const Config& cfg,
                         const char* ssrSpv,
                         const char* compositeSpv,
                         const char* ssaoSpv,
                         const char* ssaoBlurSpv) {
    m_ctx = &ctx;
    m_cfg = cfg;
    m_ssrResultFirstUse = true;
    m_depthPyramidFirstUse = true;
    m_ssgiStrength = cfg.ssgiStrength;

    // Create SSR result image: RGBA16F, STORAGE for compute write + SAMPLED for composite read
    auto res = Image::create(ctx,
                             {cfg.width, cfg.height},
                             VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             "theia.ssrResult");
    if (!res) {
        Logger::error("SSRPass: failed to create SSR result image");
        return false;
    }
    m_ssrResult = std::move(*res);

    // Create AO result image: R16F, STORAGE for SSAO write + SAMPLED for blur read
    auto aoRes = Image::create(ctx,
                               {cfg.width, cfg.height},
                               VK_FORMAT_R16_SFLOAT,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               "theia.ssaoResult");
    if (!aoRes) {
        Logger::error("SSRPass: failed to create SSAO result image");
        return false;
    }
    m_ssaoResult = std::move(*aoRes);

    uint32_t mipW = cfg.width;
    uint32_t mipH = cfg.height;
    m_depthPyramidMipCount = 1;
    while (mipW > 1u || mipH > 1u) {
        mipW = (mipW > 1u) ? (mipW >> 1) : 1u;
        mipH = (mipH > 1u) ? (mipH >> 1) : 1u;
        ++m_depthPyramidMipCount;
    }
    auto pyramid = Image::create(ctx,
                                 {cfg.width, cfg.height},
                                 VK_FORMAT_R32_SFLOAT,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                 "theia.depthPyramid",
                                 m_depthPyramidMipCount);
    if (!pyramid) {
        Logger::error("SSRPass: failed to create depth pyramid image");
        return false;
    }
    m_depthPyramid = std::move(*pyramid);
    m_depthPyramidMipViews.clear();
    m_depthPyramidMipViews.reserve(m_depthPyramidMipCount);
    for (uint32_t mip = 0; mip < m_depthPyramidMipCount; ++mip) {
        auto view = createMipView(ctx, m_depthPyramid, mip);
        if (view == VK_NULL_HANDLE) {
            return false;
        }
        m_depthPyramidMipViews.push_back(view);
    }

    if (!createSamplers()) {
        return false;
    }
    if (!createDescriptors()) {
        return false;
    }
    if (!createPipelines(ssrSpv, compositeSpv, ssaoSpv, ssaoBlurSpv, "ssgi.comp.spv")) {
        return false;
    }
    updateDescriptors();

    Logger::info("SSRPass initialized ({}x{})", cfg.width, cfg.height);
    return true;
}

void SSRPass::shutdown() {
    if (!m_ctx) {
        return;
    }
    vkDeviceWaitIdle(m_ctx->device);

    auto destroy = [&](auto& handle, auto fn) {
        if (handle != VK_NULL_HANDLE) {
            fn(m_ctx->device, handle, nullptr);
            handle = VK_NULL_HANDLE;
        }
    };

    destroy(m_depthPyramidPipeline, vkDestroyPipeline);
    destroy(m_depthPyramidLayout, vkDestroyPipelineLayout);
    destroy(m_depthPyramidSetLayout, vkDestroyDescriptorSetLayout);
    destroy(m_depthPyramidPool, vkDestroyDescriptorPool);

    destroy(m_ssrPipeline, vkDestroyPipeline);
    destroy(m_ssrLayout, vkDestroyPipelineLayout);
    destroy(m_ssrSetLayout, vkDestroyDescriptorSetLayout);
    destroy(m_ssrPool, vkDestroyDescriptorPool);

    destroy(m_compositePipeline, vkDestroyPipeline);
    destroy(m_compositeLayout, vkDestroyPipelineLayout);
    destroy(m_compositeSetLayout, vkDestroyDescriptorSetLayout);
    destroy(m_compositePool, vkDestroyDescriptorPool);

    destroy(m_ssaoPipeline, vkDestroyPipeline);
    destroy(m_ssaoLayout, vkDestroyPipelineLayout);
    destroy(m_ssaoSetLayout, vkDestroyDescriptorSetLayout);
    destroy(m_ssaoPool, vkDestroyDescriptorPool);

    destroy(m_ssaoBlurPipeline, vkDestroyPipeline);
    destroy(m_ssaoBlurLayout, vkDestroyPipelineLayout);
    destroy(m_ssaoBlurSetLayout, vkDestroyDescriptorSetLayout);
    destroy(m_ssaoBlurPool, vkDestroyDescriptorPool);

    destroy(m_ssgiPipeline, vkDestroyPipeline);
    destroy(m_ssgiLayout, vkDestroyPipelineLayout);
    destroy(m_ssgiSetLayout, vkDestroyDescriptorSetLayout);
    destroy(m_ssgiPool, vkDestroyDescriptorPool);

    destroy(m_samplerNearest, vkDestroySampler);
    destroy(m_samplerLinear, vkDestroySampler);

    for (VkImageView view : m_depthPyramidMipViews) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_ctx->device, view, nullptr);
        }
    }
    m_depthPyramidMipViews.clear();

    m_ssrResult = {};
    m_ssaoResult = {};
    m_depthPyramid = {};
    m_ssgiStrength = 0.0f;
    m_ctx = nullptr;
}

void SSRPass::dispatch(VkCommandBuffer cmd, const glm::mat4& proj, const glm::mat4& invProj, bool afterGi) {
    const VkImageLayout ssrOldLayout = m_ssrResultFirstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    m_ssrResultFirstUse = false;
    const VkImageLayout hdrOldLayout = afterGi ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    const VkImageLayout gbufferOldLayout = afterGi ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                   : VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;

    // ---- Barrier: transition all inputs and outputs for compute ----
    // depth    : ATTACHMENT_OPTIMAL → DEPTH_STENCIL_READ_ONLY (sampled by SSR)
    // gbuffer  : ATTACHMENT_OPTIMAL/GENERAL → SHADER_READ_ONLY (sampled by SSR + composite)
    // hdr      : ATTACHMENT_OPTIMAL/GENERAL → GENERAL          (sampled by SSR, written by composite)
    // ssrResult: UNDEFINED/GENERAL  → GENERAL                 (written by SSR)
    const VkImageMemoryBarrier2 depthBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = m_cfg.depthImage,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    const std::array<VkImageMemoryBarrier2, 4> preBarriers{{
        depthBarrier,
        imgBarrier(m_cfg.gbufferImage,
                   gbufferOldLayout,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   afterGi ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   afterGi ? VK_ACCESS_2_SHADER_READ_BIT : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT),
        imgBarrier(m_cfg.hdrImage,
                   hdrOldLayout,
                   VK_IMAGE_LAYOUT_GENERAL,
                   afterGi ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   afterGi ? VK_ACCESS_2_SHADER_WRITE_BIT : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT),
        imgBarrier(m_ssrResult.handle(),
                   ssrOldLayout,
                   VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_NONE,
                   0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_WRITE_BIT),
    }};
    const VkDependencyInfo preDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(preBarriers.size()),
        .pImageMemoryBarriers = preBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &preDep);

    // ---- Depth pyramid generation (min depth hierarchy for SSR) ----
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_depthPyramidPipeline);
    for (uint32_t mip = 0; mip < m_depthPyramidMipCount; ++mip) {
        const VkImageLayout outOldLayout = m_depthPyramidFirstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        m_depthPyramid.transition(cmd,
                                   outOldLayout,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT,
                                   mip,
                                   1);

        const DepthPyramidPushConstants dpc{mip == 0 ? 0u : (mip - 1u), 0u, 0u, 0u};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_depthPyramidLayout, 0, 1, &m_depthPyramidSets[mip], 0, nullptr);
        vkCmdPushConstants(cmd, m_depthPyramidLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dpc), &dpc);
        const uint32_t mipWidth = std::max(1u, m_cfg.width >> mip);
        const uint32_t mipHeight = std::max(1u, m_cfg.height >> mip);
        vkCmdDispatch(cmd, (mipWidth + 7u) / 8u, (mipHeight + 7u) / 8u, 1);

        m_depthPyramid.transition(cmd,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT,
                                   mip,
                                   1);
    }
    m_depthPyramidFirstUse = false;

    // ---- SSR ray march dispatch ----
    const SSRPushConstants pc{proj, invProj};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssrPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssrLayout, 0, 1, &m_ssrSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_ssrLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    const uint32_t gx = (m_cfg.width + 7) / 8;
    const uint32_t gy = (m_cfg.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // ---- Barrier: SSR write → composite read ----
    pipelineBarrier(cmd,
                    {
                        imgBarrier(m_ssrResult.handle(),
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT),
                    });

    // ---- Composite dispatch ----
    const CompositePushConstants cpc{m_cfg.ssrStrength, kRoughnessMax, 0.0f, 0.0f};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositeLayout, 0, 1, &m_compositeSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_compositeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cpc), &cpc);
    vkCmdDispatch(cmd, gx, gy, 1);

    if (m_ssgiStrength > 0.0f) {
        // SSGI reads and writes the same HDR buffer in-place, so make the
        // prior composite writes visible to sampled reads before dispatch.
        pipelineBarrier(cmd,
                        {
                            imgBarrier(m_cfg.hdrImage,
                                       VK_IMAGE_LAYOUT_GENERAL,
                                       VK_IMAGE_LAYOUT_GENERAL,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_2_SHADER_WRITE_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT),
                        });
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssgiPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssgiLayout, 0, 1, &m_ssgiSet, 0, nullptr);
        vkCmdDispatch(cmd, gx, gy, 1);
    }

    // ---- SSAO / contact-shadow dispatch ----
    // Writes a raw (noisy) ambient-occlusion factor to m_ssaoResult. A following bilateral
    // blur denoises it and multiplies it into the composited HDR, giving the soft contact
    // shadows that IBL-only scenes lack without the per-pixel grain of unfiltered SSAO.
    const VkImageLayout aoOldLayout = m_ssaoResultFirstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    m_ssaoResultFirstUse = false;
    pipelineBarrier(cmd,
                    {
                        imgBarrier(m_ssaoResult.handle(),
                                   aoOldLayout,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_NONE,
                                   0,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT),
                    });
    const SSAOPushConstants spc{proj, invProj, 0.15f, m_ssaoStrength, 0.025f, 1.5f};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoLayout, 0, 1, &m_ssaoSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_ssaoLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(spc), &spc);
    vkCmdDispatch(cmd, gx, gy, 1);

    // ---- SSAO bilateral-blur + apply dispatch ----
    // Barriers: AO write → read; composite's HDR write → blur read/write.
    pipelineBarrier(cmd,
                    {
                        imgBarrier(m_ssaoResult.handle(),
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT),
                        imgBarrier(m_cfg.hdrImage,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT),
                    });
    const SSAOBlurPushConstants bpc{invProj, {1.0f / float(m_cfg.width), 1.0f / float(m_cfg.height)}, 0.0f, 0.0f};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoBlurPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoBlurLayout, 0, 1, &m_ssaoBlurSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_ssaoBlurLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bpc), &bpc);
    vkCmdDispatch(cmd, gx, gy, 1);
    // HDR stays in GENERAL — ToneMapper reads it as GENERAL. ✓
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

bool SSRPass::createSamplers() {
    const VkSamplerCreateInfo nearestInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    if (vkCreateSampler(m_ctx->device, &nearestInfo, nullptr, &m_samplerNearest) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create nearest sampler");
        return false;
    }

    const VkSamplerCreateInfo linearInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    if (vkCreateSampler(m_ctx->device, &linearInfo, nullptr, &m_samplerLinear) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create linear sampler");
        return false;
    }
    return true;
}

bool SSRPass::createDescriptors() {
    // --- Depth pyramid set: 0=source depth/sampled, 1=destination mip storage ---
    const std::array<VkDescriptorSetLayoutBinding, 2> depthPyramidBindings{{
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    }};
    const VkDescriptorSetLayoutCreateInfo depthPyramidSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(depthPyramidBindings.size()),
        .pBindings = depthPyramidBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &depthPyramidSetInfo, nullptr, &m_depthPyramidSetLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create depth pyramid descriptor set layout");
        return false;
    }
    const std::array<VkDescriptorPoolSize, 2> depthPyramidPoolSizes{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_depthPyramidMipCount},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, m_depthPyramidMipCount},
    }};
    const VkDescriptorPoolCreateInfo depthPyramidPoolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = m_depthPyramidMipCount,
        .poolSizeCount = static_cast<uint32_t>(depthPyramidPoolSizes.size()),
        .pPoolSizes = depthPyramidPoolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &depthPyramidPoolInfo, nullptr, &m_depthPyramidPool) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create depth pyramid descriptor pool");
        return false;
    }
    std::vector<VkDescriptorSetLayout> depthLayouts(m_depthPyramidMipCount, m_depthPyramidSetLayout);
    const VkDescriptorSetAllocateInfo depthAllocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_depthPyramidPool,
        .descriptorSetCount = static_cast<uint32_t>(depthLayouts.size()),
        .pSetLayouts = depthLayouts.data(),
    };
    m_depthPyramidSets.resize(m_depthPyramidMipCount, VK_NULL_HANDLE);
    if (vkAllocateDescriptorSets(m_ctx->device, &depthAllocInfo, m_depthPyramidSets.data()) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to allocate depth pyramid descriptor sets");
        return false;
    }

    // --- SSR ray march set: binding 0=depth, 1=gbuffer, 2=hdr, 3=ssrOut, 4=nearestSampler, 5=linearSampler ---
    const std::array<VkDescriptorSetLayoutBinding, 6> ssrBindings{{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // depth
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // gbuffer
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // hdr
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},          // ssrOut
        {4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // nearest
        {5, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // linear
    }};
    constexpr VkDescriptorBindingFlags kUAB = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    const std::array<VkDescriptorBindingFlags, 6> ssrBindingFlags{kUAB, kUAB, kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo ssrBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(ssrBindingFlags.size()),
        .pBindingFlags = ssrBindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo ssrSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &ssrBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<uint32_t>(ssrBindings.size()),
        .pBindings = ssrBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &ssrSetInfo, nullptr, &m_ssrSetLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSR descriptor set layout");
        return false;
    }
    const std::array<VkDescriptorPoolSize, 3> ssrPoolSizes{{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 2},
    }};
    const VkDescriptorPoolCreateInfo ssrPoolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(ssrPoolSizes.size()),
        .pPoolSizes = ssrPoolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &ssrPoolInfo, nullptr, &m_ssrPool) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSR descriptor pool");
        return false;
    }
    const VkDescriptorSetAllocateInfo ssrAllocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_ssrPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_ssrSetLayout,
    };
    if (vkAllocateDescriptorSets(m_ctx->device, &ssrAllocInfo, &m_ssrSet) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to allocate SSR descriptor set");
        return false;
    }

    // --- Composite set: binding 0=ssrResult(sampled), 1=gbuffer(sampled), 2=hdr(storage), 3=linearSampler ---
    const std::array<VkDescriptorSetLayoutBinding, 4> compBindings{{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // ssrResult
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // gbuffer
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},          // hdr
        {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // linear
    }};
    const std::array<VkDescriptorBindingFlags, 4> compBindingFlags{kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo compBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(compBindingFlags.size()),
        .pBindingFlags = compBindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo compSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &compBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<uint32_t>(compBindings.size()),
        .pBindings = compBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &compSetInfo, nullptr, &m_compositeSetLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create composite descriptor set layout");
        return false;
    }
    const std::array<VkDescriptorPoolSize, 3> compPoolSizes{{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
    }};
    const VkDescriptorPoolCreateInfo compPoolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(compPoolSizes.size()),
        .pPoolSizes = compPoolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &compPoolInfo, nullptr, &m_compositePool) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create composite descriptor pool");
        return false;
    }
    const VkDescriptorSetAllocateInfo compAllocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_compositePool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_compositeSetLayout,
    };
    if (vkAllocateDescriptorSets(m_ctx->device, &compAllocInfo, &m_compositeSet) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to allocate composite descriptor set");
        return false;
    }

    // --- SSAO set: 0=depth(sampler), 1=gbuffer(sampler), 2=ao(storage write), 3=nearest, 4=linear ---
    const std::array<VkDescriptorSetLayoutBinding, 5> ssaoBindings{{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // depth
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // gbuffer
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},          // ao (write)
        {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // nearest
        {4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // linear
    }};
    const std::array<VkDescriptorBindingFlags, 5> ssaoBindingFlags{kUAB, kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo ssaoBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(ssaoBindingFlags.size()),
        .pBindingFlags = ssaoBindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo ssaoSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &ssaoBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<uint32_t>(ssaoBindings.size()),
        .pBindings = ssaoBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &ssaoSetInfo, nullptr, &m_ssaoSetLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSAO descriptor set layout");
        return false;
    }
    const std::array<VkDescriptorPoolSize, 3> ssaoPoolSizes{{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 2},
    }};
    const VkDescriptorPoolCreateInfo ssaoPoolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(ssaoPoolSizes.size()),
        .pPoolSizes = ssaoPoolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &ssaoPoolInfo, nullptr, &m_ssaoPool) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSAO descriptor pool");
        return false;
    }
    const VkDescriptorSetAllocateInfo ssaoAllocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_ssaoPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_ssaoSetLayout,
    };
    if (vkAllocateDescriptorSets(m_ctx->device, &ssaoAllocInfo, &m_ssaoSet) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to allocate SSAO descriptor set");
        return false;
    }

    // --- SSAO blur set: 0=ao(sampler), 1=depth(sampler), 2=hdr(storage rw), 3=nearest, 4=linear ---
    const std::array<VkDescriptorSetLayoutBinding, 5> blurBindings{{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // ao
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // depth
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},          // hdr (rw)
        {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // nearest
        {4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // linear
    }};
    const std::array<VkDescriptorBindingFlags, 5> blurBindingFlags{kUAB, kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo blurBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(blurBindingFlags.size()),
        .pBindingFlags = blurBindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo blurSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &blurBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<uint32_t>(blurBindings.size()),
        .pBindings = blurBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &blurSetInfo, nullptr, &m_ssaoBlurSetLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSAO blur descriptor set layout");
        return false;
    }
    const std::array<VkDescriptorPoolSize, 3> blurPoolSizes{{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 2},
    }};
    const VkDescriptorPoolCreateInfo blurPoolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(blurPoolSizes.size()),
        .pPoolSizes = blurPoolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &blurPoolInfo, nullptr, &m_ssaoBlurPool) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSAO blur descriptor pool");
        return false;
    }
    const VkDescriptorSetAllocateInfo blurAllocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_ssaoBlurPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_ssaoBlurSetLayout,
    };
    if (vkAllocateDescriptorSets(m_ctx->device, &blurAllocInfo, &m_ssaoBlurSet) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to allocate SSAO blur descriptor set");
        return false;
    }

    // --- SSGI set: 0=depth(sampled), 1=gbuffer(sampled), 2=hdr(sampled), 3=hdr(storage), 4=nearest, 5=linear ---
    const std::array<VkDescriptorSetLayoutBinding, 6> ssgiBindings{{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // depth
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // gbuffer
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // hdr sampled
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},          // hdr storage
        {4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // nearest
        {5, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // linear
    }};
    const std::array<VkDescriptorBindingFlags, 6> ssgiBindingFlags{kUAB, kUAB, kUAB, kUAB, kUAB, kUAB};
    const VkDescriptorSetLayoutBindingFlagsCreateInfo ssgiBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(ssgiBindingFlags.size()),
        .pBindingFlags = ssgiBindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo ssgiSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &ssgiBindingFlagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<uint32_t>(ssgiBindings.size()),
        .pBindings = ssgiBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &ssgiSetInfo, nullptr, &m_ssgiSetLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSGI descriptor set layout");
        return false;
    }
    const std::array<VkDescriptorPoolSize, 3> ssgiPoolSizes{{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 2},
    }};
    const VkDescriptorPoolCreateInfo ssgiPoolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(ssgiPoolSizes.size()),
        .pPoolSizes = ssgiPoolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &ssgiPoolInfo, nullptr, &m_ssgiPool) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSGI descriptor pool");
        return false;
    }
    const VkDescriptorSetAllocateInfo ssgiAllocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_ssgiPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_ssgiSetLayout,
    };
    if (vkAllocateDescriptorSets(m_ctx->device, &ssgiAllocInfo, &m_ssgiSet) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to allocate SSGI descriptor set");
        return false;
    }

    return true;
}

bool SSRPass::createPipelines(const char* ssrSpv,
                              const char* compositeSpv,
                              const char* ssaoSpv,
                              const char* ssaoBlurSpv,
                              const char* ssgiSpv) {
    // --- Depth pyramid pipeline ---
    const VkPushConstantRange depthPyramidPCRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(DepthPyramidPushConstants),
    };
    const VkPipelineLayoutCreateInfo depthPyramidLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_depthPyramidSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &depthPyramidPCRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &depthPyramidLayoutInfo, nullptr, &m_depthPyramidLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create depth pyramid pipeline layout");
        return false;
    }

    VkShaderModule depthPyramidModule = loadSpv(m_ctx->device, "depth_pyramid.comp.spv");
    if (!depthPyramidModule) {
        return false;
    }

    const VkComputePipelineCreateInfo depthPyramidPipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = depthPyramidModule,
                  .pName = "main"},
        .layout = m_depthPyramidLayout,
    };
    const VkResult depthPyramidRes = vkCreateComputePipelines(
        m_ctx->device, VK_NULL_HANDLE, 1, &depthPyramidPipeInfo, nullptr, &m_depthPyramidPipeline);
    vkDestroyShaderModule(m_ctx->device, depthPyramidModule, nullptr);
    if (depthPyramidRes != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create depth pyramid compute pipeline");
        return false;
    }

    // --- SSR pipeline ---
    const VkPushConstantRange ssrPCRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(SSRPushConstants),
    };
    const VkPipelineLayoutCreateInfo ssrLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_ssrSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &ssrPCRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &ssrLayoutInfo, nullptr, &m_ssrLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSR pipeline layout");
        return false;
    }

    VkShaderModule ssrModule = loadSpv(m_ctx->device, ssrSpv);
    if (!ssrModule) {
        return false;
    }

    const VkComputePipelineCreateInfo ssrPipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = ssrModule,
                  .pName = "main"},
        .layout = m_ssrLayout,
    };
    const VkResult ssrRes =
        vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &ssrPipeInfo, nullptr, &m_ssrPipeline);
    vkDestroyShaderModule(m_ctx->device, ssrModule, nullptr);
    if (ssrRes != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSR compute pipeline");
        return false;
    }

    // --- Composite pipeline ---
    const VkPushConstantRange compPCRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(CompositePushConstants),
    };
    const VkPipelineLayoutCreateInfo compLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_compositeSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &compPCRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &compLayoutInfo, nullptr, &m_compositeLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create composite pipeline layout");
        return false;
    }

    VkShaderModule compModule = loadSpv(m_ctx->device, compositeSpv);
    if (!compModule) {
        return false;
    }

    const VkComputePipelineCreateInfo compPipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = compModule,
                  .pName = "main"},
        .layout = m_compositeLayout,
    };
    const VkResult compRes =
        vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &compPipeInfo, nullptr, &m_compositePipeline);
    vkDestroyShaderModule(m_ctx->device, compModule, nullptr);
    if (compRes != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create composite compute pipeline");
        return false;
    }

    // --- SSAO pipeline ---
    const VkPushConstantRange ssaoPCRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(SSAOPushConstants),
    };
    const VkPipelineLayoutCreateInfo ssaoLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_ssaoSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &ssaoPCRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &ssaoLayoutInfo, nullptr, &m_ssaoLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSAO pipeline layout");
        return false;
    }
    VkShaderModule ssaoModule = loadSpv(m_ctx->device, ssaoSpv);
    if (!ssaoModule) {
        return false;
    }
    const VkComputePipelineCreateInfo ssaoPipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = ssaoModule,
                  .pName = "main"},
        .layout = m_ssaoLayout,
    };
    const VkResult ssaoRes =
        vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &ssaoPipeInfo, nullptr, &m_ssaoPipeline);
    vkDestroyShaderModule(m_ctx->device, ssaoModule, nullptr);
    if (ssaoRes != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSAO compute pipeline");
        return false;
    }

    // --- SSAO blur pipeline ---
    const VkPushConstantRange blurPCRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(SSAOBlurPushConstants),
    };
    const VkPipelineLayoutCreateInfo blurLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_ssaoBlurSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &blurPCRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &blurLayoutInfo, nullptr, &m_ssaoBlurLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSAO blur pipeline layout");
        return false;
    }
    VkShaderModule blurModule = loadSpv(m_ctx->device, ssaoBlurSpv);
    if (!blurModule) {
        return false;
    }
    const VkComputePipelineCreateInfo blurPipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = blurModule,
                  .pName = "main"},
        .layout = m_ssaoBlurLayout,
    };
    const VkResult blurRes =
        vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &blurPipeInfo, nullptr, &m_ssaoBlurPipeline);
    vkDestroyShaderModule(m_ctx->device, blurModule, nullptr);
    if (blurRes != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSAO blur compute pipeline");
        return false;
    }

    // --- SSGI pipeline ---
    const VkPipelineLayoutCreateInfo ssgiLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_ssgiSetLayout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &ssgiLayoutInfo, nullptr, &m_ssgiLayout) != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSGI pipeline layout");
        return false;
    }
    VkShaderModule ssgiModule = loadSpv(m_ctx->device, ssgiSpv);
    if (!ssgiModule) {
        return false;
    }
    const VkComputePipelineCreateInfo ssgiPipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = ssgiModule,
                  .pName = "main"},
        .layout = m_ssgiLayout,
    };
    const VkResult ssgiRes =
        vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &ssgiPipeInfo, nullptr, &m_ssgiPipeline);
    vkDestroyShaderModule(m_ctx->device, ssgiModule, nullptr);
    if (ssgiRes != VK_SUCCESS) {
        Logger::error("SSRPass: failed to create SSGI compute pipeline");
        return false;
    }

    return true;
}

void SSRPass::updateDescriptors() {
    // Depth pyramid descriptor writes (source depth / previous mip, destination mip storage).
    const VkDescriptorImageInfo depthPyramidSrcDepthInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = m_cfg.depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    };
    for (uint32_t mip = 0; mip < m_depthPyramidMipCount; ++mip) {
        const VkDescriptorImageInfo depthPyramidSrcInfo = (mip == 0)
                                                               ? depthPyramidSrcDepthInfo
                                                               : VkDescriptorImageInfo{
                                                                     .sampler = VK_NULL_HANDLE,
                                                                     .imageView = m_depthPyramidMipViews[mip - 1],
                                                                     .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                 };
        const VkDescriptorImageInfo depthPyramidDstInfo{
            .imageView = m_depthPyramidMipViews[mip],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const std::array<VkWriteDescriptorSet, 2> depthPyramidWrites{{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             nullptr,
             m_depthPyramidSets[mip],
             0,
             0,
             1,
             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
             &depthPyramidSrcInfo,
             nullptr,
             nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
             nullptr,
             m_depthPyramidSets[mip],
             1,
             0,
             1,
             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             &depthPyramidDstInfo,
             nullptr,
             nullptr},
        }};
        vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(depthPyramidWrites.size()), depthPyramidWrites.data(), 0, nullptr);
    }

    // SSR pass descriptor writes
    const VkDescriptorImageInfo depthInfo{
        .sampler = m_samplerNearest,
        .imageView = m_depthPyramid.view(),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo gbufferInfo{
        .sampler = m_samplerLinear,
        .imageView = m_cfg.gbufferView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo hdrSampledInfo{
        .sampler = m_samplerLinear,
        .imageView = m_cfg.hdrView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo ssrOutInfo{
        .imageView = m_ssrResult.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo nearestSamplerInfo{.sampler = m_samplerNearest};
    const VkDescriptorImageInfo linearSamplerInfo{.sampler = m_samplerLinear};

    const std::array<VkWriteDescriptorSet, 6> ssrWrites{{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssrSet,
         0,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &depthInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssrSet,
         1,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &gbufferInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssrSet,
         2,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &hdrSampledInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssrSet,
         3,
         0,
         1,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         &ssrOutInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssrSet,
         4,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &nearestSamplerInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssrSet,
         5,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &linearSamplerInfo,
         nullptr,
         nullptr},
    }};
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(ssrWrites.size()), ssrWrites.data(), 0, nullptr);

    // Composite pass descriptor writes
    const VkDescriptorImageInfo ssrResultSampledInfo{
        .sampler = m_samplerLinear,
        .imageView = m_ssrResult.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo hdrStorageInfo{
        .imageView = m_cfg.hdrView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    const std::array<VkWriteDescriptorSet, 4> compWrites{{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_compositeSet,
         0,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &ssrResultSampledInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_compositeSet,
         1,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &gbufferInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_compositeSet,
         2,
         0,
         1,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         &hdrStorageInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_compositeSet,
         3,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &linearSamplerInfo,
         nullptr,
         nullptr},
    }};
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(compWrites.size()), compWrites.data(), 0, nullptr);

    // SSAO pass descriptor writes (depth + gbuffer sampled, AO storage out, two samplers).
    const VkDescriptorImageInfo ssaoDepthInfo{
        .sampler = m_samplerNearest,
        .imageView = m_cfg.depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo ssaoOutInfo{
        .imageView = m_ssaoResult.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const std::array<VkWriteDescriptorSet, 5> ssaoWrites{{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoSet,
         0,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &ssaoDepthInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoSet,
         1,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &gbufferInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoSet,
         2,
         0,
         1,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         &ssaoOutInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoSet,
         3,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &nearestSamplerInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoSet,
         4,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &linearSamplerInfo,
         nullptr,
         nullptr},
    }};
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(ssaoWrites.size()), ssaoWrites.data(), 0, nullptr);

    // SSAO blur pass descriptor writes (AO + depth sampled, hdr storage rw, two samplers).
    const VkDescriptorImageInfo blurAoInfo{
        .sampler = m_samplerLinear,
        .imageView = m_ssaoResult.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo blurDepthInfo{
        .sampler = m_samplerNearest,
        .imageView = m_cfg.depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo blurHdrStorageInfo{
        .imageView = m_cfg.hdrView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const std::array<VkWriteDescriptorSet, 5> blurWrites{{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoBlurSet,
         0,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &blurAoInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoBlurSet,
         1,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &blurDepthInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoBlurSet,
         2,
         0,
         1,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         &blurHdrStorageInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoBlurSet,
         3,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &nearestSamplerInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssaoBlurSet,
         4,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &linearSamplerInfo,
         nullptr,
         nullptr},
    }};
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(blurWrites.size()), blurWrites.data(), 0, nullptr);

    // SSGI pass descriptor writes (depth + gbuffer sampled, HDR sampled + storage, two samplers).
    const VkDescriptorImageInfo ssgiDepthInfo{
        .sampler = m_samplerNearest,
        .imageView = m_cfg.depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo ssgiGbufferInfo{
        .sampler = m_samplerLinear,
        .imageView = m_cfg.gbufferView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo ssgiHdrSampledInfo{
        .sampler = m_samplerLinear,
        .imageView = m_cfg.hdrView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo ssgiHdrStorageInfo{
        .imageView = m_cfg.hdrView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const std::array<VkWriteDescriptorSet, 6> ssgiWrites{{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssgiSet,
         0,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &ssgiDepthInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssgiSet,
         1,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &ssgiGbufferInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssgiSet,
         2,
         0,
         1,
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         &ssgiHdrSampledInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssgiSet,
         3,
         0,
         1,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         &ssgiHdrStorageInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssgiSet,
         4,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &nearestSamplerInfo,
         nullptr,
         nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         nullptr,
         m_ssgiSet,
         5,
         0,
         1,
         VK_DESCRIPTOR_TYPE_SAMPLER,
         &linearSamplerInfo,
         nullptr,
         nullptr},
    }};
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(ssgiWrites.size()), ssgiWrites.data(), 0, nullptr);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
