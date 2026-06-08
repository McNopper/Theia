#include "demo/vulkan_init/PhysicalDevice.hpp"

#include <array>
#include <string_view>
#include <vector>

namespace {
constexpr std::array kRequiredExtensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
};

[[nodiscard]] int scoreDevice(VkPhysicalDeviceProperties properties) noexcept {
    int score = 0;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
    } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 250;
    }
    score += static_cast<int>(properties.limits.maxImageDimension2D);
    return score;
}
} // namespace

std::expected<PhysicalDeviceInfo, VkResult> PhysicalDevice::select(VkInstance instance, VkSurfaceKHR surface) {
    uint32_t deviceCount = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }
    if (deviceCount == 0U) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    bool foundCompatible = false;
    int bestScore = -1;
    PhysicalDeviceInfo bestInfo{};

    for (VkPhysicalDevice device : devices) {
        if (!hasRequiredExtensions(device) || !hasRayTracingSupport(device)) {
            continue;
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        if (queueFamilyCount == 0U) {
            continue;
        }

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        uint32_t graphicsFamily = VK_QUEUE_FAMILY_IGNORED;
        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0U) {
                continue;
            }

            VkBool32 presentSupported = VK_FALSE;
            result = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupported);
            if (result != VK_SUCCESS) {
                return std::unexpected(result);
            }
            if (presentSupported == VK_TRUE) {
                graphicsFamily = i;
                break;
            }
        }
        if (graphicsFamily == VK_QUEUE_FAMILY_IGNORED) {
            continue;
        }

        PhysicalDeviceInfo info{};
        info.device = device;
        info.graphicsFamily = graphicsFamily;
        info.rtProps = {};
        info.rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        info.properties = {};
        info.properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        info.properties.pNext = &info.rtProps;
        vkGetPhysicalDeviceProperties2(device, &info.properties);
        vkGetPhysicalDeviceMemoryProperties(device, &info.memProperties);

        const int score = scoreDevice(info.properties.properties) + 500;
        if (!foundCompatible || score > bestScore) {
            foundCompatible = true;
            bestScore = score;
            bestInfo = info;
        }
    }

    if (!foundCompatible) {
        return std::unexpected(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    return bestInfo;
}

bool PhysicalDevice::hasRayTracingSupport(VkPhysicalDevice device) {
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
    rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.pNext = &rtFeatures;
    VkPhysicalDeviceVulkan14Features features14{};
    features14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    features14.pNext = &asFeatures;
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.pNext = &features14;
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features12;
    vkGetPhysicalDeviceFeatures2(device, &features2);

    return asFeatures.accelerationStructure == VK_TRUE && rtFeatures.rayTracingPipeline == VK_TRUE &&
           features12.bufferDeviceAddress == VK_TRUE && features12.descriptorIndexing == VK_TRUE &&
           features12.runtimeDescriptorArray == VK_TRUE && features12.descriptorBindingPartiallyBound == VK_TRUE &&
           features12.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE &&
           features12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE &&
           features13.dynamicRendering == VK_TRUE && features13.synchronization2 == VK_TRUE &&
           features13.maintenance4 == VK_TRUE && features14.pushDescriptor == VK_TRUE;
}

bool PhysicalDevice::hasRequiredExtensions(VkPhysicalDevice device) {
    uint32_t extensionCount = 0;
    VkResult result = vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    if (result != VK_SUCCESS || extensionCount == 0U) {
        return false;
    }

    std::vector<VkExtensionProperties> extensions(extensionCount);
    result = vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());
    if (result != VK_SUCCESS) {
        return false;
    }

    for (const char* required : kRequiredExtensions) {
        bool found = false;
        for (const VkExtensionProperties& extension : extensions) {
            if (std::string_view(extension.extensionName) == required) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}
