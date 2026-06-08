#pragma once

#include <volk/volk.h>

#include <expected>

class DebugUtils {
  public:
    [[nodiscard]] static std::expected<DebugUtils, VkResult> create(VkInstance instance);

    DebugUtils() = default;
    DebugUtils(const DebugUtils&) = delete;
    DebugUtils& operator=(const DebugUtils&) = delete;
    DebugUtils(DebugUtils&& other) noexcept;
    DebugUtils& operator=(DebugUtils&& other) noexcept;
    ~DebugUtils();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                        VkDebugUtilsMessageTypeFlagsEXT types,
                                                        const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                        void* userData);

  private:
    void destroy() noexcept;

    VkInstance m_instance{};
    VkDebugUtilsMessengerEXT m_messenger{};
};
