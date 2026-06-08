#pragma once

#include <volk/volk.h>

#include <SDL3/SDL.h>

#include <expected>
#include <string>

#include "demo/vulkan_init/DebugUtils.hpp"
#include "demo/vulkan_init/PhysicalDevice.hpp"
#include "hyperion/DeviceContext.hpp"

class Context {
  public:
    struct Config {
        std::string appName = "Hyperion";
        uint32_t appVersion = VK_MAKE_VERSION(1, 0, 0);
        bool enableValidation = true;
        SDL_Window* window = nullptr;
    };

    [[nodiscard]] static std::expected<Context, VkResult> create(Config config);

    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;
    ~Context();

    [[nodiscard]] const DeviceContext& deviceContext() const noexcept;
    [[nodiscard]] VkInstance instance() const noexcept;
    [[nodiscard]] VkSurfaceKHR surface() const noexcept;
    [[nodiscard]] const PhysicalDeviceInfo& physicalDeviceInfo() const noexcept;

  private:
    void destroy() noexcept;

    VkInstance m_instance{};
    VkSurfaceKHR m_surface{};
    DebugUtils m_debugUtils{};
    PhysicalDeviceInfo m_physicalDeviceInfo{};
    DeviceContext m_deviceContext{};
    bool m_validationEnabled = false;
};
