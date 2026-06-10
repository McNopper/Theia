#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "theia/renderer/SSRPass.hpp"

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

static constexpr float kRoughnessMax = 0.45f; // must match ssr.comp.slang

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

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

    if (!createSamplers()) {
        return false;
    }
    if (!createDescriptors()) {
        return false;
    }
    if (!createPipelines(ssrSpv, compositeSpv, ssaoSpv, ssaoBlurSpv)) {
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

    destroy(m_samplerNearest, vkDestroySampler);
    destroy(m_samplerLinear, vkDestroySampler);

    m_ssrResult = {};
    m_ssaoResult = {};
    m_ctx = nullptr;
}

void SSRPass::dispatch(VkCommandBuffer cmd, const glm::mat4& proj, const glm::mat4& invProj) {
    const VkImageLayout ssrOldLayout = m_ssrResultFirstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    m_ssrResultFirstUse = false;

    // ---- Barrier: transition all inputs and outputs for compute ----
    // depth    : ATTACHMENT_OPTIMAL → DEPTH_STENCIL_READ_ONLY (sampled by SSR)
    // gbuffer  : ATTACHMENT_OPTIMAL → SHADER_READ_ONLY        (sampled by SSR + composite)
    // hdr      : ATTACHMENT_OPTIMAL → GENERAL                 (sampled by SSR, written by composite)
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
    // --- SSR ray march set: binding 0=depth, 1=gbuffer, 2=hdr, 3=ssrOut, 4=nearestSampler, 5=linearSampler ---
    const std::array<VkDescriptorSetLayoutBinding, 6> ssrBindings{{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // depth
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // gbuffer
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // hdr
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},          // ssrOut
        {4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // nearest
        {5, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},                // linear
    }};
    const VkDescriptorSetLayoutCreateInfo ssrSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
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
    const VkDescriptorSetLayoutCreateInfo compSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
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
    const VkDescriptorSetLayoutCreateInfo ssaoSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
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
    const VkDescriptorSetLayoutCreateInfo blurSetInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
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

    return true;
}

bool SSRPass::createPipelines(const char* ssrSpv,
                              const char* compositeSpv,
                              const char* ssaoSpv,
                              const char* ssaoBlurSpv) {
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

    return true;
}

void SSRPass::updateDescriptors() {
    // SSR pass descriptor writes
    const VkDescriptorImageInfo depthInfo{
        .sampler = m_samplerNearest,
        .imageView = m_cfg.depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
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
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
