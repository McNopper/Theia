#include "demo/presentation/Surface.hpp"

#include <SDL3/SDL_vulkan.h>

#include <utility>

std::expected<Surface, VkResult> Surface::create(VkInstance instance, SDL_Window* window) {
    if (instance == VK_NULL_HANDLE || window == nullptr) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    Surface surface;
    surface.m_instance = instance;
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface.m_surface)) {
        return std::unexpected(VK_ERROR_SURFACE_LOST_KHR);
    }
    return surface;
}

Surface::Surface(Surface&& other) noexcept
    : m_instance(std::exchange(other.m_instance, VK_NULL_HANDLE)),
      m_surface(std::exchange(other.m_surface, VK_NULL_HANDLE)) {}

Surface& Surface::operator=(Surface&& other) noexcept {
    if (this != &other) {
        destroy();
        m_instance = std::exchange(other.m_instance, VK_NULL_HANDLE);
        m_surface = std::exchange(other.m_surface, VK_NULL_HANDLE);
    }
    return *this;
}

Surface::~Surface() {
    destroy();
}

VkSurfaceKHR Surface::handle() const noexcept {
    return m_surface;
}

void Surface::destroy() noexcept {
    if (m_instance != VK_NULL_HANDLE && m_surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    }
    m_instance = VK_NULL_HANDLE;
    m_surface = VK_NULL_HANDLE;
}
