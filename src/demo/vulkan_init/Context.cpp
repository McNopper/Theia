#include "demo/vulkan_init/Context.hpp"

#include <SDL3/SDL_vulkan.h>

#include <array>
#include <string_view>
#include <utility>
#include <vector>
#include <vma/vk_mem_alloc.h>

namespace {
[[nodiscard]] bool validationEnabled(bool requested) noexcept {
    return requested;
}

[[nodiscard]] bool hasValidationLayer() {
    uint32_t layerCount = 0;
    if (vkEnumerateInstanceLayerProperties(&layerCount, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkLayerProperties> layers(layerCount);
    if (vkEnumerateInstanceLayerProperties(&layerCount, layers.data()) != VK_SUCCESS) {
        return false;
    }
    for (const VkLayerProperties& layer : layers) {
        if (std::string_view(layer.layerName) == "VK_LAYER_KHRONOS_validation") {
            return true;
        }
    }
    return false;
}

[[nodiscard]] VkResult createSurface(VkInstance instance, SDL_Window* window, VkSurfaceKHR& surface) {
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
        return VK_ERROR_SURFACE_LOST_KHR;
    }
    return VK_SUCCESS;
}

// Verify required volk device globals were loaded.
[[nodiscard]] VkResult checkRayTracingFunctions() {
    return (vkCmdTraceRaysKHR != nullptr && vkCmdBuildAccelerationStructuresKHR != nullptr &&
            vkCreateAccelerationStructureKHR != nullptr && vkDestroyAccelerationStructureKHR != nullptr &&
            vkGetAccelerationStructureDeviceAddressKHR != nullptr && vkGetRayTracingShaderGroupHandlesKHR != nullptr &&
            vkCmdPushDescriptorSet != nullptr)
               ? VK_SUCCESS
               : VK_ERROR_FEATURE_NOT_PRESENT;
}

[[nodiscard]] VkResult createAllocator(const DeviceContext& ctx, VkInstance instance, VmaAllocator& allocator) {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    functions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    functions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    functions.vkAllocateMemory = vkAllocateMemory;
    functions.vkFreeMemory = vkFreeMemory;
    functions.vkMapMemory = vkMapMemory;
    functions.vkUnmapMemory = vkUnmapMemory;
    functions.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    functions.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    functions.vkBindBufferMemory = vkBindBufferMemory;
    functions.vkBindImageMemory = vkBindImageMemory;
    functions.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    functions.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    functions.vkCreateBuffer = vkCreateBuffer;
    functions.vkDestroyBuffer = vkDestroyBuffer;
    functions.vkCreateImage = vkCreateImage;
    functions.vkDestroyImage = vkDestroyImage;
    functions.vkCmdCopyBuffer = vkCmdCopyBuffer;
    functions.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2;
    functions.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2;
    functions.vkBindBufferMemory2KHR = vkBindBufferMemory2;
    functions.vkBindImageMemory2KHR = vkBindImageMemory2;
    functions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
    functions.vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements;
    functions.vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements;

    const VmaAllocatorCreateInfo createInfo{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = ctx.physicalDevice,
        .device = ctx.device,
        .preferredLargeHeapBlockSize = 0,
        .pAllocationCallbacks = nullptr,
        .pDeviceMemoryCallbacks = nullptr,
        .pHeapSizeLimit = nullptr,
        .pVulkanFunctions = &functions,
        .instance = instance,
        .vulkanApiVersion = VK_API_VERSION_1_4,
        .pTypeExternalMemoryHandleTypes = nullptr,
    };
    return vmaCreateAllocator(&createInfo, &allocator);
}

[[nodiscard]] VkResult createDevice(const PhysicalDeviceInfo& info, DeviceContext& ctx) {
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuerySupported{};
    rayQuerySupported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeaturesSupported{};
    rtFeaturesSupported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtFeaturesSupported.pNext = &rayQuerySupported;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeaturesSupported{};
    asFeaturesSupported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeaturesSupported.pNext = &rtFeaturesSupported;
    VkPhysicalDeviceMeshShaderFeaturesEXT meshFeaturesSupported{};
    meshFeaturesSupported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    meshFeaturesSupported.pNext = &asFeaturesSupported;
    VkPhysicalDeviceVulkan14Features features14Supported{};
    features14Supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    features14Supported.pNext = &meshFeaturesSupported;
    VkPhysicalDeviceVulkan13Features features13Supported{};
    features13Supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13Supported.pNext = &features14Supported;
    VkPhysicalDeviceVulkan12Features features12Supported{};
    features12Supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12Supported.pNext = &features13Supported;
    VkPhysicalDeviceVulkan11Features features11Supported{};
    features11Supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11Supported.pNext = &features12Supported;
    VkPhysicalDeviceFeatures2 supportedFeatures{};
    supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures.pNext = &features11Supported;
    vkGetPhysicalDeviceFeatures2(info.device, &supportedFeatures);

    if (features12Supported.bufferDeviceAddress != VK_TRUE || features12Supported.descriptorIndexing != VK_TRUE ||
        features12Supported.runtimeDescriptorArray != VK_TRUE ||
        features12Supported.descriptorBindingPartiallyBound != VK_TRUE ||
        features12Supported.descriptorBindingStorageBufferUpdateAfterBind != VK_TRUE ||
        features12Supported.descriptorBindingSampledImageUpdateAfterBind != VK_TRUE ||
        features12Supported.timelineSemaphore != VK_TRUE || features13Supported.dynamicRendering != VK_TRUE ||
        features13Supported.synchronization2 != VK_TRUE || features13Supported.maintenance4 != VK_TRUE ||
        features14Supported.pushDescriptor != VK_TRUE || asFeaturesSupported.accelerationStructure != VK_TRUE ||
        rtFeaturesSupported.rayTracingPipeline != VK_TRUE || meshFeaturesSupported.meshShader != VK_TRUE ||
        rayQuerySupported.rayQuery != VK_TRUE) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.rayQuery = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{};
    rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtFeatures.pNext = &rayQueryFeatures;
    rtFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.pNext = &rtFeatures;
    asFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceVulkan14Features features14{};
    features14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    features14.pNext = &asFeatures;
    features14.pushDescriptor = VK_TRUE;

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.pNext = &features14;
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;
    features13.maintenance4 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;
    features12.descriptorIndexing = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing =
        features12Supported.shaderSampledImageArrayNonUniformIndexing;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.pNext = &features12;
    features11.multiview = features11Supported.multiview;
    features11.shaderDrawParameters = features11Supported.shaderDrawParameters;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features11;
    features2.features.multiDrawIndirect = supportedFeatures.features.multiDrawIndirect;
    features2.features.samplerAnisotropy = supportedFeatures.features.samplerAnisotropy;
    features2.features.shaderInt64 = supportedFeatures.features.shaderInt64;

    // Enable mesh shader features
    VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{};
    meshFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    meshFeatures.pNext = &features2;
    meshFeatures.taskShader = VK_TRUE;
    meshFeatures.meshShader = VK_TRUE;

    constexpr float queuePriority = 1.0f;
    const VkDeviceQueueCreateInfo queueInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = info.graphicsFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    constexpr std::array deviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_EXT_MESH_SHADER_EXTENSION_NAME,
    };
    const VkDeviceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &meshFeatures,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = nullptr,
    };

    return vkCreateDevice(info.device, &createInfo, nullptr, &ctx.device);
}
} // namespace

std::expected<Context, VkResult> Context::create(Config config) {
    if (config.window == nullptr) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }
    VkResult result = volkInitialize();
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    Context context;
    context.m_validationEnabled = validationEnabled(config.enableValidation);

    Uint32 sdlExtensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    if (sdlExtensions == nullptr || sdlExtensionCount == 0U) {
        return std::unexpected(VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    std::vector<const char*> extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);

    // Probe available instance extensions so we can opt-in to HDR color spaces.
    // VK_EXT_swapchain_colorspace is required to use any non-sRGB VkColorSpaceKHR
    // (HDR10, HLG, scRGB, Display P3 …) in a swapchain; without it the validation
    // layer rejects vkCreateSwapchainKHR even if the driver enumerates those formats.
    {
        uint32_t extCount = 0;
        if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr) == VK_SUCCESS && extCount > 0) {
            std::vector<VkExtensionProperties> available(extCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &extCount, available.data());
            for (const VkExtensionProperties& ext : available) {
                if (std::string_view(ext.extensionName) == VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) {
                    extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
                    break;
                }
            }
        }
    }

    std::vector<const char*> layers;
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{
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

    if (context.m_validationEnabled) {
        if (!hasValidationLayer()) {
            return std::unexpected(VK_ERROR_LAYER_NOT_PRESENT);
        }
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = config.appName.c_str(),
        .applicationVersion = config.appVersion,
        .pEngineName = "Hyperion",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_4,
    };
    const VkInstanceCreateInfo instanceInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = context.m_validationEnabled ? &debugCreateInfo : nullptr,
        .flags = 0,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.empty() ? nullptr : layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    result = vkCreateInstance(&instanceInfo, nullptr, &context.m_instance);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    volkLoadInstance(context.m_instance);

    if (context.m_validationEnabled) {
        auto debugUtils = DebugUtils::create(context.m_instance);
        if (!debugUtils) {
            return std::unexpected(debugUtils.error());
        }
        context.m_debugUtils = std::move(*debugUtils);
    }

    result = createSurface(context.m_instance, config.window, context.m_surface);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    auto physical = PhysicalDevice::select(context.m_instance, context.m_surface);
    if (!physical) {
        return std::unexpected(physical.error());
    }
    context.m_physicalDeviceInfo = *physical;

    context.m_deviceContext.physicalDevice = context.m_physicalDeviceInfo.device;
    context.m_deviceContext.graphicsFamily = context.m_physicalDeviceInfo.graphicsFamily;
    result = createDevice(context.m_physicalDeviceInfo, context.m_deviceContext);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    volkLoadDevice(context.m_deviceContext.device);
    result = checkRayTracingFunctions();
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    vkGetDeviceQueue(context.m_deviceContext.device,
                     context.m_physicalDeviceInfo.graphicsFamily,
                     0,
                     &context.m_deviceContext.graphicsQueue);
    result = createAllocator(context.m_deviceContext, context.m_instance, context.m_deviceContext.allocator);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    return context;
}

Context::Context(Context&& other) noexcept
    : m_instance(std::exchange(other.m_instance, VK_NULL_HANDLE)),
      m_surface(std::exchange(other.m_surface, VK_NULL_HANDLE)),
      m_debugUtils(std::move(other.m_debugUtils)),
      m_physicalDeviceInfo(other.m_physicalDeviceInfo),
      m_deviceContext(other.m_deviceContext),
      m_validationEnabled(other.m_validationEnabled) {
    other.m_physicalDeviceInfo = {};
    other.m_deviceContext = {};
    other.m_validationEnabled = false;
}

Context& Context::operator=(Context&& other) noexcept {
    if (this != &other) {
        destroy();
        m_instance = std::exchange(other.m_instance, VK_NULL_HANDLE);
        m_surface = std::exchange(other.m_surface, VK_NULL_HANDLE);
        m_debugUtils = std::move(other.m_debugUtils);
        m_physicalDeviceInfo = other.m_physicalDeviceInfo;
        m_deviceContext = other.m_deviceContext;
        m_validationEnabled = other.m_validationEnabled;
        other.m_physicalDeviceInfo = {};
        other.m_deviceContext = {};
        other.m_validationEnabled = false;
    }
    return *this;
}

Context::~Context() {
    destroy();
}

const DeviceContext& Context::deviceContext() const noexcept {
    return m_deviceContext;
}

VkInstance Context::instance() const noexcept {
    return m_instance;
}

VkSurfaceKHR Context::surface() const noexcept {
    return m_surface;
}

const PhysicalDeviceInfo& Context::physicalDeviceInfo() const noexcept {
    return m_physicalDeviceInfo;
}

void Context::destroy() noexcept {
    if (m_deviceContext.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_deviceContext.device);
    }
    if (m_deviceContext.allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(m_deviceContext.allocator);
    }
    if (m_deviceContext.device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_deviceContext.device, nullptr);
    }
    if (m_instance != VK_NULL_HANDLE && m_surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    }

    m_deviceContext = {};
    m_physicalDeviceInfo = {};
    m_surface = VK_NULL_HANDLE;
    m_debugUtils = {};

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
    }
    m_instance = VK_NULL_HANDLE;
    m_validationEnabled = false;
}
