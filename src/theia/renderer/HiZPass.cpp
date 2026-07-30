#include "theia/renderer/HiZPass.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "harmonia/core/Logger.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "theia/renderer/ShaderPath.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

namespace {

[[nodiscard]] std::uint32_t mipCountFor(std::uint32_t w, std::uint32_t h) noexcept {
    const std::uint32_t maxDim = std::max(w, h);
    std::uint32_t levels = 1;
    while ((maxDim >> (levels - 1)) > 1u) {
        ++levels;
    }
    return levels;
}

[[nodiscard]] std::uint32_t mipExtent(std::uint32_t base, std::uint32_t level) noexcept {
    return std::max(1u, base >> level);
}

} // namespace

HiZPass::~HiZPass() {
    shutdown();
}

bool HiZPass::initialize(const harmonia::DeviceContext& ctx, std::uint32_t width, std::uint32_t height, const char* spvName) {
    shutdown();
    m_ctx = &ctx;
    m_width = width;
    m_height = height;
    m_firstUse = true;

    if (width == 0 || height == 0) {
        harmonia::Logger::error("HiZPass: invalid extent {}x{}", width, height);
        m_ctx = nullptr;
        return false;
    }
    m_mipLevels = mipCountFor(width, height);

    if (!createImage()) {
        shutdown();
        return false;
    }
    if (!createPipeline(spvName)) {
        // Backward-compatible: keep the image (so callers can still bind a valid Hi-Z
        // descriptor) but report failure via isInitialized() == false.
        harmonia::Logger::warn("HiZPass: pipeline unavailable — occlusion culling disabled");
        return false;
    }
    harmonia::Logger::info("HiZPass: initialized {}x{} with {} mip levels", width, height, m_mipLevels);
    return true;
}

void HiZPass::shutdown() {
    if (m_ctx == nullptr) {
        return;
    }
    m_mipViews.clear();
    m_image = {};
    m_pipeline.reset();
    m_pipelineLayout.reset();
    m_setLayout.reset();
    m_ctx = nullptr;
    m_width = 0;
    m_height = 0;
    m_mipLevels = 1;
    m_firstUse = true;
}

bool HiZPass::createImage() noexcept {
    auto img = harmonia::Image::create(*m_ctx,
                             {m_width, m_height},
                             VK_FORMAT_R32_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             "theia.hiZ",
                             m_mipLevels);
    if (!img) {
        harmonia::Logger::error("HiZPass: failed to create Hi-Z image: VkResult {}", static_cast<int>(img.error()));
        return false;
    }
    m_image = std::move(*img);

    // One single-mip storage view per level (used as reduction source/destination).
    m_mipViews.clear();
    m_mipViews.reserve(m_mipLevels);
    for (std::uint32_t level = 0; level < m_mipLevels; ++level) {
        const VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = m_image.handle(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R32_SFLOAT,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1},
        };
        VkImageView view{};
        if (vkCreateImageView(m_ctx->device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
            harmonia::Logger::error("HiZPass: failed to create mip view {}", level);
            return false;
        }
        m_mipViews.emplace_back(m_ctx->device, view);
    }
    return true;
}

bool HiZPass::createPipeline(const char* spvName) noexcept {
    const std::array bindings{
        VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, // srcDepth (level 0)
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, // srcMip (level L-1)
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, // dstMip (level L)
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    {
        VkDescriptorSetLayout setLayout{};
        if (vkCreateDescriptorSetLayout(m_ctx->device, &setInfo, nullptr, &setLayout) != VK_SUCCESS) {
            harmonia::Logger::error("HiZPass: failed to create descriptor set layout");
            return false;
        }
        m_setLayout = harmonia::UniqueDescriptorSetLayout{m_ctx->device, setLayout};
    }

    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(HiZPC),
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_setLayout.ptr(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    {
        VkPipelineLayout pipelineLayout{};
        if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            harmonia::Logger::error("HiZPass: failed to create pipeline layout");
            return false;
        }
        m_pipelineLayout = harmonia::UniquePipelineLayout{m_ctx->device, pipelineLayout};
    }

    auto module = harmonia::createShaderModule(m_ctx->device, shaderPath(spvName));
    if (!module) {
        harmonia::Logger::error("HiZPass: cannot load shader: {}", spvName);
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
    VkPipeline pipeline{};
    const VkResult res = vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline);
    vkDestroyShaderModule(m_ctx->device, *module, nullptr);
    if (res != VK_SUCCESS) {
        harmonia::Logger::error("HiZPass: failed to create compute pipeline");
        return false;
    }
    m_pipeline = harmonia::UniquePipeline{m_ctx->device, pipeline};
    return true;
}

void HiZPass::prepareForSampling(VkCommandBuffer cmd) noexcept {
    if (!m_image.isValid() || !m_firstUse) {
        return;
    }
    m_firstUse = false;
    // UNDEFINED -> SHADER_READ_ONLY_OPTIMAL for all mips. Pass 1 never samples these
    // (hiZMipCount = 0), so the undefined contents are harmless; this only satisfies the
    // layout the mesh shader's Hi-Z descriptor declares.
    m_image.transition(cmd,
                       VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_PIPELINE_STAGE_2_NONE,
                       0,
                       VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void HiZPass::build(VkCommandBuffer cmd, VkImageView depthView) noexcept {
    if (m_pipeline == VK_NULL_HANDLE || !m_image.isValid() || depthView == VK_NULL_HANDLE) {
        return;
    }

    // Hi-Z: SHADER_READ_ONLY_OPTIMAL -> GENERAL (all mips) for compute writes.
    m_image.transition(cmd,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

    const VkDescriptorImageInfo depthInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = depthView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    for (std::uint32_t level = 0; level < m_mipLevels; ++level) {
        const std::uint32_t dstW = mipExtent(m_width, level);
        const std::uint32_t dstH = mipExtent(m_height, level);
        const std::uint32_t srcLevel = (level == 0) ? 0 : (level - 1);
        const std::uint32_t srcW = (level == 0) ? m_width : mipExtent(m_width, level - 1);
        const std::uint32_t srcH = (level == 0) ? m_height : mipExtent(m_height, level - 1);

        const VkDescriptorImageInfo srcMipInfo{
            .imageView = m_mipViews[srcLevel],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorImageInfo dstMipInfo{
            .imageView = m_mipViews[level],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const std::array writes{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .pImageInfo = &depthInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &srcMipInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &dstMipInfo,
            },
        };
        vkCmdPushDescriptorSet(cmd,
                               VK_PIPELINE_BIND_POINT_COMPUTE,
                               m_pipelineLayout,
                               0,
                               static_cast<std::uint32_t>(writes.size()),
                               writes.data());

        const HiZPC pc{
            .srcSize = {srcW, srcH},
            .dstSize = {dstW, dstH},
            .copyMode = (level == 0) ? 1u : 0u,
            ._pad = 0u,
        };
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        vkCmdDispatch(cmd, (dstW + 7u) / 8u, (dstH + 7u) / 8u, 1);

        // Make this level's writes visible to the next level's reads (skip after last).
        if (level + 1 < m_mipLevels) {
            const VkImageMemoryBarrier2 levelBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = m_image.handle(),
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, m_mipLevels, 0, 1},
            };
            const VkDependencyInfo dep{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &levelBarrier,
            };
            vkCmdPipelineBarrier2(cmd, &dep);
        }
    }

    // Hi-Z: GENERAL -> SHADER_READ_ONLY_OPTIMAL for pass-2 mesh-shader sampling.
    m_image.transition(cmd,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
