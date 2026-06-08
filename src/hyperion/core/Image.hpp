#pragma once

#include <expected>
#include <string_view>

#include "hyperion/DeviceContext.hpp"

class Image {
  public:
    [[nodiscard]] static std::expected<Image, VkResult> create(const DeviceContext& ctx,
                                                               VkExtent2D extent,
                                                               VkFormat format,
                                                               VkImageUsageFlags usage,
                                                               VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                                                               std::string_view debugName = "",
                                                               uint32_t mipLevels = 1U);

    Image() = default;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    ~Image();

    void transition(VkCommandBuffer cmd,
                    VkImageLayout oldLayout,
                    VkImageLayout newLayout,
                    VkPipelineStageFlags2 srcStage,
                    VkAccessFlags2 srcAccess,
                    VkPipelineStageFlags2 dstStage,
                    VkAccessFlags2 dstAccess,
                    uint32_t baseMipLevel = 0U,
                    uint32_t levelCount = 0U) const noexcept;

    [[nodiscard]] VkImage handle() const noexcept { return m_image; }
    [[nodiscard]] VkImageView view() const noexcept { return m_view; }
    [[nodiscard]] VkFormat format() const noexcept { return m_format; }
    [[nodiscard]] VkExtent2D extent() const noexcept { return m_extent; }
    [[nodiscard]] uint32_t mipLevels() const noexcept { return m_mipLevels; }
    [[nodiscard]] bool isValid() const noexcept { return m_image != VK_NULL_HANDLE; }

  private:
    void destroy() noexcept;

    VkImage m_image = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkExtent2D m_extent{};
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags m_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t m_mipLevels = 1U;
};
