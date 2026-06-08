#include "demo/vulkan_init/DebugUtils.hpp"

#include <string_view>
#include <utility>

#include "hyperion/core/Logger.hpp"

namespace {
[[nodiscard]] const char* severityLabel(VkDebugUtilsMessageSeverityFlagBitsEXT severity) noexcept {
    switch (severity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        return "warning";
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        return "error";
    default:
        return "message";
    }
}

[[nodiscard]] const char* typeLabel(VkDebugUtilsMessageTypeFlagsEXT types) noexcept {
    if ((types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) != 0U) {
        return "validation";
    }
    if ((types & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) != 0U) {
        return "performance";
    }
    return "general";
}
} // namespace

std::expected<DebugUtils, VkResult> DebugUtils::create(VkInstance instance) {
    if (instance == VK_NULL_HANDLE || vkCreateDebugUtilsMessengerEXT == nullptr) {
        return std::unexpected(VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    const VkDebugUtilsMessengerCreateInfoEXT createInfo{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = &DebugUtils::debugCallback,
        .pUserData = nullptr,
    };

    DebugUtils debugUtils;
    debugUtils.m_instance = instance;
    const VkResult result = vkCreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugUtils.m_messenger);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    return debugUtils;
}

DebugUtils::DebugUtils(DebugUtils&& other) noexcept
    : m_instance(std::exchange(other.m_instance, VK_NULL_HANDLE)),
      m_messenger(std::exchange(other.m_messenger, VK_NULL_HANDLE)) {}

DebugUtils& DebugUtils::operator=(DebugUtils&& other) noexcept {
    if (this != &other) {
        destroy();
        m_instance = std::exchange(other.m_instance, VK_NULL_HANDLE);
        m_messenger = std::exchange(other.m_messenger, VK_NULL_HANDLE);
    }
    return *this;
}

DebugUtils::~DebugUtils() {
    destroy();
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtils::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                         VkDebugUtilsMessageTypeFlagsEXT types,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                         void* /*userData*/) {
    const std::string_view message = callbackData != nullptr && callbackData->pMessage != nullptr
                                         ? callbackData->pMessage
                                         : std::string_view{"(no message)"};

    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        Logger::error("Vulkan {} {}: {}", severityLabel(severity), typeLabel(types), message);
    } else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        Logger::warn("Vulkan {} {}: {}", severityLabel(severity), typeLabel(types), message);
    }
    return VK_FALSE;
}

void DebugUtils::destroy() noexcept {
    if (m_instance != VK_NULL_HANDLE && m_messenger != VK_NULL_HANDLE && vkDestroyDebugUtilsMessengerEXT != nullptr) {
        vkDestroyDebugUtilsMessengerEXT(m_instance, m_messenger, nullptr);
    }
    m_instance = VK_NULL_HANDLE;
    m_messenger = VK_NULL_HANDLE;
}
