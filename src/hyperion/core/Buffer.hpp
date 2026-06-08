#pragma once

#include <expected>
#include <string_view>

#include "hyperion/DeviceContext.hpp"

class Buffer {
  public:
    [[nodiscard]] static std::expected<Buffer, VkResult> create(const DeviceContext& ctx,
                                                                VkDeviceSize size,
                                                                VkBufferUsageFlags usage,
                                                                VmaMemoryUsage memUsage,
                                                                std::string_view debugName = "");

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    ~Buffer();

    [[nodiscard]] VkBuffer handle() const noexcept { return m_buffer; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return m_size; }
    [[nodiscard]] void* mappedData() const noexcept { return m_mapped; }
    [[nodiscard]] VkDeviceAddress deviceAddress() const noexcept;
    [[nodiscard]] bool isValid() const noexcept { return m_buffer != VK_NULL_HANDLE; }

    void uploadData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);

  private:
    void destroy() noexcept;

    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    uint32_t m_queueFamily = 0;
    void* m_mapped = nullptr;
};
