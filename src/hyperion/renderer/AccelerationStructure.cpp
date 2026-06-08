#include "hyperion/renderer/AccelerationStructure.hpp"

#include <string>
#include <utility>

AccelerationStructure::~AccelerationStructure() noexcept {
    reset();
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept
    : m_device(std::exchange(other.m_device, VK_NULL_HANDLE)),
      m_buffer(std::move(other.m_buffer)),
      m_handle(std::exchange(other.m_handle, VK_NULL_HANDLE)),
      m_deviceAddress(std::exchange(other.m_deviceAddress, 0)) {}

AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&& other) noexcept {
    if (this != &other) {
        reset();
        m_device = std::exchange(other.m_device, VK_NULL_HANDLE);
        m_buffer = std::move(other.m_buffer);
        m_handle = std::exchange(other.m_handle, VK_NULL_HANDLE);
        m_deviceAddress = std::exchange(other.m_deviceAddress, 0);
    }
    return *this;
}

std::expected<AccelerationStructure, VkResult> AccelerationStructure::create(const DeviceContext& ctx,
                                                                             VkAccelerationStructureTypeKHR type,
                                                                             VkDeviceSize size,
                                                                             std::string_view debugName) {
    auto storage = Buffer::create(ctx,
                                  size,
                                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  std::string(debugName).append(".buffer"));
    if (!storage) {
        return std::unexpected(storage.error());
    }

    const VkAccelerationStructureCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .createFlags = 0,
        .buffer = storage->handle(),
        .offset = 0,
        .size = size,
        .type = type,
        .deviceAddress = 0,
    };

    AccelerationStructure accelerationStructure;
    accelerationStructure.m_device = ctx.device;
    accelerationStructure.m_buffer = std::move(*storage);

    if (const VkResult result =
            vkCreateAccelerationStructureKHR(ctx.device, &createInfo, nullptr, &accelerationStructure.m_handle);
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    const VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .pNext = nullptr,
        .accelerationStructure = accelerationStructure.m_handle,
    };
    accelerationStructure.m_deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(ctx.device, &addressInfo);

    if (!debugName.empty()) {
        ctx.setDebugName(VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
                         reinterpret_cast<uint64_t>(accelerationStructure.m_handle),
                         std::string(debugName).c_str());
    }

    return accelerationStructure;
}

void AccelerationStructure::reset() noexcept {
    if (m_device != VK_NULL_HANDLE && m_handle != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(m_device, m_handle, nullptr);
        m_handle = VK_NULL_HANDLE;
    }
    m_device = VK_NULL_HANDLE;
    m_deviceAddress = 0;
}
