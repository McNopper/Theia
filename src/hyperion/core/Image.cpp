#include "hyperion/core/Image.hpp"

#include <string>

std::expected<Image, VkResult> Image::create(const DeviceContext& ctx,
                                             VkExtent2D extent,
                                             VkFormat format,
                                             VkImageUsageFlags usage,
                                             VkImageAspectFlags aspect,
                                             std::string_view debugName,
                                             uint32_t mipLevels) {
    if (!ctx.isValid() || ctx.allocator == VK_NULL_HANDLE || extent.width == 0U || extent.height == 0U ||
        mipLevels == 0U) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    const VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0U,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = VkExtent3D{extent.width, extent.height, 1U},
        .mipLevels = mipLevels,
        .arrayLayers = 1U,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0U,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VmaAllocationCreateInfo allocationInfo{
        .flags = 0U,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0U,
        .preferredFlags = 0U,
        .memoryTypeBits = 0U,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.0F,
    };

    Image image;
    const VkResult imageResult =
        vmaCreateImage(ctx.allocator, &imageInfo, &allocationInfo, &image.m_image, &image.m_allocation, nullptr);
    if (imageResult != VK_SUCCESS) {
        return std::unexpected(imageResult);
    }

    const VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0U,
        .image = image.m_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components =
            VkComponentMapping{
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
        .subresourceRange =
            VkImageSubresourceRange{
                .aspectMask = aspect,
                .baseMipLevel = 0U,
                .levelCount = mipLevels,
                .baseArrayLayer = 0U,
                .layerCount = 1U,
            },
    };

    const VkResult viewResult = vkCreateImageView(ctx.device, &viewInfo, nullptr, &image.m_view);
    if (viewResult != VK_SUCCESS) {
        vmaDestroyImage(ctx.allocator, image.m_image, image.m_allocation);
        return std::unexpected(viewResult);
    }

    image.m_allocator = ctx.allocator;
    image.m_device = ctx.device;
    image.m_extent = extent;
    image.m_format = format;
    image.m_aspect = aspect;
    image.m_mipLevels = mipLevels;

    if (!debugName.empty()) {
        const std::string baseName(debugName);
        ctx.setDebugName(VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(image.m_image), baseName.c_str());
        ctx.setDebugName(
            VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<uint64_t>(image.m_view), (baseName + " View").c_str());
    }

    return image;
}

Image::Image(Image&& other) noexcept
    : m_image(other.m_image),
      m_view(other.m_view),
      m_allocation(other.m_allocation),
      m_allocator(other.m_allocator),
      m_device(other.m_device),
      m_extent(other.m_extent),
      m_format(other.m_format),
      m_aspect(other.m_aspect),
      m_mipLevels(other.m_mipLevels) {
    other.m_image = VK_NULL_HANDLE;
    other.m_view = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_allocator = VK_NULL_HANDLE;
    other.m_device = VK_NULL_HANDLE;
    other.m_extent = {};
    other.m_format = VK_FORMAT_UNDEFINED;
    other.m_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    other.m_mipLevels = 1U;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        destroy();
        m_image = other.m_image;
        m_view = other.m_view;
        m_allocation = other.m_allocation;
        m_allocator = other.m_allocator;
        m_device = other.m_device;
        m_extent = other.m_extent;
        m_format = other.m_format;
        m_aspect = other.m_aspect;
        m_mipLevels = other.m_mipLevels;

        other.m_image = VK_NULL_HANDLE;
        other.m_view = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_allocator = VK_NULL_HANDLE;
        other.m_device = VK_NULL_HANDLE;
        other.m_extent = {};
        other.m_format = VK_FORMAT_UNDEFINED;
        other.m_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        other.m_mipLevels = 1U;
    }
    return *this;
}

Image::~Image() {
    destroy();
}

void Image::transition(VkCommandBuffer cmd,
                       VkImageLayout oldLayout,
                       VkImageLayout newLayout,
                       VkPipelineStageFlags2 srcStage,
                       VkAccessFlags2 srcAccess,
                       VkPipelineStageFlags2 dstStage,
                       VkAccessFlags2 dstAccess,
                       uint32_t baseMipLevel,
                       uint32_t levelCount) const noexcept {
    if (cmd == VK_NULL_HANDLE || m_image == VK_NULL_HANDLE) {
        return;
    }

    const uint32_t resolvedLevelCount = (levelCount == 0U) ? (m_mipLevels - baseMipLevel) : levelCount;
    const VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = m_image,
        .subresourceRange =
            VkImageSubresourceRange{
                .aspectMask = m_aspect,
                .baseMipLevel = baseMipLevel,
                .levelCount = resolvedLevelCount,
                .baseArrayLayer = 0U,
                .layerCount = 1U,
            },
    };
    const VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0U,
        .memoryBarrierCount = 0U,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0U,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 1U,
        .pImageMemoryBarriers = &barrier,
    };

    vkCmdPipelineBarrier2(cmd, &dependencyInfo);
}

void Image::destroy() noexcept {
    if (m_device != VK_NULL_HANDLE && m_view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_view, nullptr);
    }
    if (m_allocator != VK_NULL_HANDLE && m_image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_image, m_allocation);
    }

    m_image = VK_NULL_HANDLE;
    m_view = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
    m_allocator = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_extent = {};
    m_format = VK_FORMAT_UNDEFINED;
    m_aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    m_mipLevels = 1U;
}
