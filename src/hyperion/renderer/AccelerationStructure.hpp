#pragma once

#include <volk/volk.h>

#include <expected>
#include <string_view>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/Buffer.hpp"

class AccelerationStructure {
  public:
    AccelerationStructure() = default;
    ~AccelerationStructure() noexcept;

    AccelerationStructure(AccelerationStructure&& other) noexcept;
    AccelerationStructure& operator=(AccelerationStructure&& other) noexcept;

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    [[nodiscard]] static std::expected<AccelerationStructure, VkResult> create(const DeviceContext& ctx,
                                                                               VkAccelerationStructureTypeKHR type,
                                                                               VkDeviceSize size,
                                                                               std::string_view debugName = "");

    [[nodiscard]] VkAccelerationStructureKHR handle() const noexcept { return m_handle; }
    [[nodiscard]] VkDeviceAddress deviceAddress() const noexcept { return m_deviceAddress; }
    [[nodiscard]] const Buffer& buffer() const noexcept { return m_buffer; }

  private:
    void reset() noexcept;

    VkDevice m_device{VK_NULL_HANDLE};
    Buffer m_buffer{};
    VkAccelerationStructureKHR m_handle{VK_NULL_HANDLE};
    VkDeviceAddress m_deviceAddress{};
};
