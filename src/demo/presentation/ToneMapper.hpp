#pragma once

#include <volk/volk.h>

#include <expected>
#include <filesystem>

#include "demo/presentation/OutputColorSpace.hpp"
#include "hyperion/DeviceContext.hpp"

class ToneMapper {
  public:
    /// Create a graphics-pipeline tone mapper using dynamic rendering.
    /// swapchainFormat must match the current swapchain image format.
    [[nodiscard]] static std::expected<ToneMapper, VkResult> create(const DeviceContext& ctx,
                                                                    VkPipelineLayout pipelineLayout,
                                                                    VkFormat swapchainFormat,
                                                                    const std::filesystem::path& vertSpvPath,
                                                                    const std::filesystem::path& fragSpvPath);

    ToneMapper() = default;
    ToneMapper(const ToneMapper&) = delete;
    ToneMapper& operator=(const ToneMapper&) = delete;
    ToneMapper(ToneMapper&& other) noexcept;
    ToneMapper& operator=(ToneMapper&& other) noexcept;
    ~ToneMapper();

    /// Record the tone-mapping draw into cmd.
    /// The swapchain image must already be in VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL.
    /// colorSpace must match the swapchain's active OutputColorSpace.
    void record(VkCommandBuffer cmd,
                VkImageView hdrView,
                VkImageView swapchainView,
                VkExtent2D extent,
                OutputColorSpace colorSpace) const noexcept;

  private:
    void destroy() noexcept;

    VkDevice m_device{};
    VkPipeline m_pipeline{};
    VkPipelineLayout m_pipelineLayout{};
    VkFormat m_attachmentFormat{VK_FORMAT_UNDEFINED};
};
