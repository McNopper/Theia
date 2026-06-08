#pragma once

#include <volk/volk.h>

#include <expected>
#include <vector>

#include "demo/presentation/OutputColorSpace.hpp"
#include "hyperion/DeviceContext.hpp"

class Swapchain {
  public:
    [[nodiscard]] static std::expected<Swapchain, VkResult> create(const DeviceContext& ctx,
                                                                   VkSurfaceKHR surface,
                                                                   VkExtent2D extent,
                                                                   bool preferHDR = true,
                                                                   VkSwapchainKHR oldSwapchain = VK_NULL_HANDLE);

    Swapchain() = default;
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&& other) noexcept;
    Swapchain& operator=(Swapchain&& other) noexcept;
    ~Swapchain();

    VkResult acquireNextImage(VkSemaphore signalSemaphore, uint32_t& outIndex);
    VkResult present(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore);
    VkResult recreate(VkExtent2D newExtent);

    [[nodiscard]] VkSwapchainKHR handle() const noexcept;
    [[nodiscard]] VkFormat format() const noexcept;
    [[nodiscard]] VkColorSpaceKHR colorSpace() const noexcept;
    [[nodiscard]] VkExtent2D extent() const noexcept;
    [[nodiscard]] uint32_t imageCount() const noexcept;
    [[nodiscard]] VkImage image(uint32_t i) const noexcept;
    [[nodiscard]] VkImageView imageView(uint32_t i) const noexcept;

    /// Returns the OutputColorSpace that was negotiated with the display.
    /// Use this to drive the ToneMapper each frame.
    [[nodiscard]] OutputColorSpace outputColorSpace() const noexcept;

  private:
    void destroy() noexcept;

    const DeviceContext* m_ctx{};
    VkSurfaceKHR m_surface{};
    VkPhysicalDevice m_physicalDevice{};
    VkSwapchainKHR m_swapchain{};
    VkFormat m_format{VK_FORMAT_UNDEFINED};
    VkColorSpaceKHR m_colorSpace{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    VkExtent2D m_extent{};
    bool m_preferHDR = true;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_views;
};
