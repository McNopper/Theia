#pragma once

#include <volk/volk.h>

#include <cstdint>
#include <vma/vk_mem_alloc.h>

/// Minimal Vulkan device state shared across modules.
/// After volkLoadDevice() the codebase calls Vulkan entry points through volk globals,
/// so no per-context pfn fields are stored here.
struct DeviceContext {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsFamily = 0;

    [[nodiscard]] bool isValid() const noexcept { return device != VK_NULL_HANDLE; }

    void setDebugName(VkObjectType type, uint64_t handle, const char* name) const noexcept {
        if (device == VK_NULL_HANDLE || handle == 0U || name == nullptr || name[0] == '\0' ||
            vkSetDebugUtilsObjectNameEXT == nullptr) {
            return;
        }

        const VkDebugUtilsObjectNameInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = type,
            .objectHandle = handle,
            .pObjectName = name,
        };

        static_cast<void>(vkSetDebugUtilsObjectNameEXT(device, &info));
    }
};
