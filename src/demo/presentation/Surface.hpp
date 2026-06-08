#pragma once

#include <volk/volk.h>

#include <SDL3/SDL.h>

#include <expected>

class Surface {
  public:
    [[nodiscard]] static std::expected<Surface, VkResult> create(VkInstance instance, SDL_Window* window);

    Surface() = default;
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;
    Surface(Surface&& other) noexcept;
    Surface& operator=(Surface&& other) noexcept;
    ~Surface();

    [[nodiscard]] VkSurfaceKHR handle() const noexcept;

  private:
    void destroy() noexcept;

    VkInstance m_instance{};
    VkSurfaceKHR m_surface{};
};
