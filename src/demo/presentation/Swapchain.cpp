#include "demo/presentation/Swapchain.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace {
[[nodiscard]] VkExtent2D clampExtent(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D requested) noexcept {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    return VkExtent2D{
        .width = std::clamp(requested.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        .height = std::clamp(requested.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
}

[[nodiscard]] VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) noexcept {
    // FIFO is guaranteed by spec and most stable across drivers
    for (VkPresentModeKHR preferred : {VK_PRESENT_MODE_FIFO_KHR}) {
        if (std::find(modes.begin(), modes.end(), preferred) != modes.end()) {
            return preferred;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

[[nodiscard]] uint32_t chooseImageCount(const VkSurfaceCapabilitiesKHR& capabilities) noexcept {
    uint32_t desired = std::max(2U, capabilities.minImageCount + 1U);
    if (capabilities.maxImageCount > 0U) {
        desired = std::min(desired, capabilities.maxImageCount);
    }
    return desired;
}
} // namespace

std::expected<Swapchain, VkResult> Swapchain::create(const DeviceContext& ctx,
                                                     VkSurfaceKHR surface,
                                                     VkExtent2D extent,
                                                     bool preferHDR,
                                                     VkSwapchainKHR oldSwapchain) {
    if (!ctx.isValid() || surface == VK_NULL_HANDLE || extent.width == 0U || extent.height == 0U) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    Swapchain swapchain;
    swapchain.m_ctx = &ctx;
    swapchain.m_surface = surface;
    swapchain.m_physicalDevice = ctx.physicalDevice;
    swapchain.m_preferHDR = preferHDR;

    VkSurfaceCapabilitiesKHR capabilities{};
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physicalDevice, surface, &capabilities);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    uint32_t formatCount = 0;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice, surface, &formatCount, nullptr);
    if (result != VK_SUCCESS || formatCount == 0U) {
        return std::unexpected(result == VK_SUCCESS ? VK_ERROR_FORMAT_NOT_SUPPORTED : result);
    }
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice, surface, &formatCount, formats.data());
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    uint32_t presentModeCount = 0;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice, surface, &presentModeCount, nullptr);
    if (result != VK_SUCCESS || presentModeCount == 0U) {
        return std::unexpected(result == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : result);
    }
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    result =
        vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice, surface, &presentModeCount, presentModes.data());
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    const auto findSurfaceFormat = [&](VkFormat format, VkColorSpaceKHR colorSpace) -> const VkSurfaceFormatKHR* {
        for (const VkSurfaceFormatKHR& candidate : formats) {
            if (candidate.format == format && candidate.colorSpace == colorSpace) {
                return &candidate;
            }
        }
        return nullptr;
    };

    const VkSurfaceFormatKHR* chosenFormat = nullptr;
    if (preferHDR) {
        // Priority: HDR10 PQ → HLG → scRGB → Display P3 → SDR linear
        if (!chosenFormat)
            chosenFormat = findSurfaceFormat(VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT);
        if (!chosenFormat)
            chosenFormat = findSurfaceFormat(VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_HLG_EXT);
        if (!chosenFormat)
            chosenFormat = findSurfaceFormat(VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT);
        if (!chosenFormat)
            chosenFormat = findSurfaceFormat(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT);
        if (!chosenFormat)
            chosenFormat = findSurfaceFormat(VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_BT709_LINEAR_EXT);
    }
    // SDR fallback — always available
    if (!chosenFormat)
        chosenFormat = findSurfaceFormat(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
    if (!chosenFormat)
        chosenFormat = &formats.front();

    swapchain.m_format = chosenFormat->format;
    swapchain.m_colorSpace = chosenFormat->colorSpace;
    swapchain.m_extent = clampExtent(capabilities, extent);

    // COLOR_ATTACHMENT_BIT is always supported on swapchain images (Vulkan spec §34.2.2).
    // TRANSFER_DST_BIT is always supported when the surface supports presentation.
    const VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    const VkSwapchainCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface,
        .minImageCount = chooseImageCount(capabilities),
        .imageFormat = swapchain.m_format,
        .imageColorSpace = swapchain.m_colorSpace,
        .imageExtent = swapchain.m_extent,
        .imageArrayLayers = 1,
        .imageUsage = imageUsage,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = choosePresentMode(presentModes),
        .clipped = VK_TRUE,
        .oldSwapchain = oldSwapchain,
    };
    result = vkCreateSwapchainKHR(ctx.device, &createInfo, nullptr, &swapchain.m_swapchain);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    uint32_t imageCount = 0;
    result = vkGetSwapchainImagesKHR(ctx.device, swapchain.m_swapchain, &imageCount, nullptr);
    if (result != VK_SUCCESS || imageCount == 0U) {
        return std::unexpected(result == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : result);
    }
    swapchain.m_images.resize(imageCount);
    result = vkGetSwapchainImagesKHR(ctx.device, swapchain.m_swapchain, &imageCount, swapchain.m_images.data());
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    swapchain.m_views.reserve(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i) {
        const VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = swapchain.m_images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = swapchain.m_format,
            .components =
                VkComponentMapping{
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                },
            .subresourceRange =
                VkImageSubresourceRange{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        VkImageView view = VK_NULL_HANDLE;
        result = vkCreateImageView(ctx.device, &viewInfo, nullptr, &view);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }
        swapchain.m_views.push_back(view);
    }

    return swapchain;
}

Swapchain::Swapchain(Swapchain&& other) noexcept
    : m_ctx(other.m_ctx),
      m_surface(std::exchange(other.m_surface, VK_NULL_HANDLE)),
      m_physicalDevice(std::exchange(other.m_physicalDevice, VK_NULL_HANDLE)),
      m_swapchain(std::exchange(other.m_swapchain, VK_NULL_HANDLE)),
      m_format(other.m_format),
      m_colorSpace(other.m_colorSpace),
      m_extent(other.m_extent),
      m_preferHDR(other.m_preferHDR),
      m_images(std::move(other.m_images)),
      m_views(std::move(other.m_views)) {
    other.m_ctx = nullptr;
    other.m_format = VK_FORMAT_UNDEFINED;
    other.m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    other.m_extent = {};
    other.m_preferHDR = true;
}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept {
    if (this != &other) {
        destroy();
        m_ctx = other.m_ctx;
        m_surface = std::exchange(other.m_surface, VK_NULL_HANDLE);
        m_physicalDevice = std::exchange(other.m_physicalDevice, VK_NULL_HANDLE);
        m_swapchain = std::exchange(other.m_swapchain, VK_NULL_HANDLE);
        m_format = other.m_format;
        m_colorSpace = other.m_colorSpace;
        m_extent = other.m_extent;
        m_preferHDR = other.m_preferHDR;
        m_images = std::move(other.m_images);
        m_views = std::move(other.m_views);
        other.m_ctx = nullptr;
        other.m_format = VK_FORMAT_UNDEFINED;
        other.m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        other.m_extent = {};
        other.m_preferHDR = true;
    }
    return *this;
}

Swapchain::~Swapchain() {
    destroy();
}

VkResult Swapchain::acquireNextImage(VkSemaphore signalSemaphore, uint32_t& outIndex) {
    if (m_ctx == nullptr || m_swapchain == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    return vkAcquireNextImageKHR(m_ctx->device, m_swapchain, UINT64_MAX, signalSemaphore, VK_NULL_HANDLE, &outIndex);
}

VkResult Swapchain::present(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore) {
    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = waitSemaphore != VK_NULL_HANDLE ? 1U : 0U,
        .pWaitSemaphores = waitSemaphore != VK_NULL_HANDLE ? &waitSemaphore : nullptr,
        .swapchainCount = 1U,
        .pSwapchains = &m_swapchain,
        .pImageIndices = &imageIndex,
        .pResults = nullptr,
    };
    return vkQueuePresentKHR(queue, &presentInfo);
}

VkResult Swapchain::recreate(VkExtent2D newExtent) {
    if (m_ctx == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Pass the old swapchain so the driver can retire it and reuse resources.
    auto recreated = create(*m_ctx, m_surface, newExtent, m_preferHDR, m_swapchain);
    if (!recreated) {
        return recreated.error();
    }

    // Retire the old swapchain now that the new one is ready.
    const VkDevice device = m_ctx->device;
    for (VkImageView view : m_views) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, nullptr);
        }
    }
    m_views.clear();
    m_images.clear();
    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }

    const DeviceContext* savedCtx = recreated->m_ctx;
    *this = std::move(*recreated);
    if (m_ctx == nullptr) {
        m_ctx = savedCtx;
    }
    return VK_SUCCESS;
}

VkSwapchainKHR Swapchain::handle() const noexcept {
    return m_swapchain;
}

VkFormat Swapchain::format() const noexcept {
    return m_format;
}

VkColorSpaceKHR Swapchain::colorSpace() const noexcept {
    return m_colorSpace;
}

VkExtent2D Swapchain::extent() const noexcept {
    return m_extent;
}

uint32_t Swapchain::imageCount() const noexcept {
    return static_cast<uint32_t>(m_images.size());
}

VkImage Swapchain::image(uint32_t i) const noexcept {
    return i < m_images.size() ? m_images[i] : VK_NULL_HANDLE;
}

VkImageView Swapchain::imageView(uint32_t i) const noexcept {
    return i < m_views.size() ? m_views[i] : VK_NULL_HANDLE;
}

OutputColorSpace Swapchain::outputColorSpace() const noexcept {
    switch (m_colorSpace) {
    case VK_COLOR_SPACE_HDR10_ST2084_EXT:
        return OutputColorSpace::eHDR10;
    case VK_COLOR_SPACE_HDR10_HLG_EXT:
        return OutputColorSpace::eHLG;
    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
        return OutputColorSpace::eScRGB;
    case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
        return OutputColorSpace::eDisplayP3;
    case VK_COLOR_SPACE_BT709_LINEAR_EXT:
        return OutputColorSpace::eSDRLinear;
    default:
        return OutputColorSpace::eSDR;
    }
}

void Swapchain::destroy() noexcept {
    if (m_ctx != nullptr) {
        for (VkImageView view : m_views) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(m_ctx->device, view, nullptr);
            }
        }
        if (m_swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(m_ctx->device, m_swapchain, nullptr);
        }
    }

    m_views.clear();
    m_images.clear();
    m_swapchain = VK_NULL_HANDLE;
    m_surface = VK_NULL_HANDLE;
    m_physicalDevice = VK_NULL_HANDLE;
    m_format = VK_FORMAT_UNDEFINED;
    m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    m_extent = {};
    m_ctx = nullptr;
}
