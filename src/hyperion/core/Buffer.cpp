#include "hyperion/core/Buffer.hpp"

#include <cstring>
#include <string>

#include "hyperion/core/Logger.hpp"

namespace {
[[nodiscard]] bool prefersHostAccess(VmaMemoryUsage usage) noexcept {
    switch (usage) {
    case VMA_MEMORY_USAGE_CPU_ONLY:
    case VMA_MEMORY_USAGE_CPU_TO_GPU:
    case VMA_MEMORY_USAGE_GPU_TO_CPU:
    case VMA_MEMORY_USAGE_CPU_COPY:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] VkResult createFence(VkDevice device, VkFence* fence) noexcept {
    const VkFenceCreateInfo fenceInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0U,
    };
    return vkCreateFence(device, &fenceInfo, nullptr, fence);
}
} // namespace

std::expected<Buffer, VkResult> Buffer::create(const DeviceContext& ctx,
                                               VkDeviceSize size,
                                               VkBufferUsageFlags usage,
                                               VmaMemoryUsage memUsage,
                                               std::string_view debugName) {
    if (!ctx.isValid() || ctx.allocator == VK_NULL_HANDLE || size == 0U) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0U,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0U,
        .pQueueFamilyIndices = nullptr,
    };

    VmaAllocationCreateFlags flags = 0U;
    if (prefersHostAccess(memUsage)) {
        flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    const VmaAllocationCreateInfo allocationInfo{
        .flags = flags,
        .usage = memUsage,
        .requiredFlags = 0U,
        .preferredFlags = 0U,
        .memoryTypeBits = 0U,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.0F,
    };

    Buffer buffer;
    VmaAllocationInfo vmaInfo{};
    const VkResult result =
        vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocationInfo, &buffer.m_buffer, &buffer.m_allocation, &vmaInfo);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    buffer.m_allocator = ctx.allocator;
    buffer.m_device = ctx.device;
    buffer.m_queue = ctx.graphicsQueue;
    buffer.m_size = size;
    buffer.m_queueFamily = ctx.graphicsFamily;
    buffer.m_mapped = vmaInfo.pMappedData;

    if (buffer.m_mapped == nullptr) {
        VkMemoryPropertyFlags memoryFlags = 0U;
        vmaGetAllocationMemoryProperties(ctx.allocator, buffer.m_allocation, &memoryFlags);
        if ((memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0U) {
            void* mapped = nullptr;
            const VkResult mapResult = vmaMapMemory(ctx.allocator, buffer.m_allocation, &mapped);
            if (mapResult != VK_SUCCESS) {
                vmaDestroyBuffer(ctx.allocator, buffer.m_buffer, buffer.m_allocation);
                return std::unexpected(mapResult);
            }
            buffer.m_mapped = mapped;
        }
    }

    if (!debugName.empty()) {
        const std::string name(debugName);
        ctx.setDebugName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(buffer.m_buffer), name.c_str());
    }

    return buffer;
}

Buffer::Buffer(Buffer&& other) noexcept
    : m_buffer(other.m_buffer),
      m_allocation(other.m_allocation),
      m_allocator(other.m_allocator),
      m_device(other.m_device),
      m_queue(other.m_queue),
      m_size(other.m_size),
      m_queueFamily(other.m_queueFamily),
      m_mapped(other.m_mapped) {
    other.m_buffer = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_allocator = VK_NULL_HANDLE;
    other.m_device = VK_NULL_HANDLE;
    other.m_queue = VK_NULL_HANDLE;
    other.m_size = 0U;
    other.m_queueFamily = 0U;
    other.m_mapped = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        destroy();
        m_buffer = other.m_buffer;
        m_allocation = other.m_allocation;
        m_allocator = other.m_allocator;
        m_device = other.m_device;
        m_queue = other.m_queue;
        m_size = other.m_size;
        m_queueFamily = other.m_queueFamily;
        m_mapped = other.m_mapped;

        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_allocator = VK_NULL_HANDLE;
        other.m_device = VK_NULL_HANDLE;
        other.m_queue = VK_NULL_HANDLE;
        other.m_size = 0U;
        other.m_queueFamily = 0U;
        other.m_mapped = nullptr;
    }
    return *this;
}

Buffer::~Buffer() {
    destroy();
}

VkDeviceAddress Buffer::deviceAddress() const noexcept {
    if (m_device == VK_NULL_HANDLE || m_buffer == VK_NULL_HANDLE) {
        return 0U;
    }

    const VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = m_buffer,
    };
    return vkGetBufferDeviceAddress(m_device, &addressInfo);
}

void Buffer::uploadData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    if (m_buffer == VK_NULL_HANDLE || m_allocator == VK_NULL_HANDLE || data == nullptr || size == 0U) {
        return;
    }
    if (offset > m_size || size > (m_size - offset)) {
        Logger::error("Buffer upload out of range: size={} offset={} capacity={}", size, offset, m_size);
        return;
    }

    if (m_mapped != nullptr) {
        auto* const dst = static_cast<unsigned char*>(m_mapped) + offset;
        std::memcpy(dst, data, static_cast<size_t>(size));
        static_cast<void>(vmaFlushAllocation(m_allocator, m_allocation, offset, size));
        return;
    }

    if (m_device == VK_NULL_HANDLE || m_queue == VK_NULL_HANDLE) {
        Logger::error("Buffer staging upload failed: missing device or queue");
        return;
    }

    const VkBufferCreateInfo stagingInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0U,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0U,
        .pQueueFamilyIndices = nullptr,
    };
    const VmaAllocationCreateInfo stagingAllocInfo{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = 0U,
        .preferredFlags = 0U,
        .memoryTypeBits = 0U,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.0F,
    };

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = VK_NULL_HANDLE;
    VmaAllocationInfo stagingAllocationInfo{};
    VkResult result = vmaCreateBuffer(
        m_allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocationInfo);
    if (result != VK_SUCCESS) {
        Logger::error("Failed to create staging buffer: VkResult {}", static_cast<int>(result));
        return;
    }

    std::memcpy(stagingAllocationInfo.pMappedData, data, static_cast<size_t>(size));
    static_cast<void>(vmaFlushAllocation(m_allocator, stagingAllocation, 0U, size));

    VkCommandPool commandPool = VK_NULL_HANDLE;
    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = m_queueFamily,
    };

    result = vkCreateCommandPool(m_device, &poolInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        Logger::error("Failed to create staging command pool: VkResult {}", static_cast<int>(result));
        vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
        return;
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1U,
    };
    result = vkAllocateCommandBuffers(m_device, &allocInfo, &cmd);
    if (result == VK_SUCCESS) {
        const VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        result = vkBeginCommandBuffer(cmd, &beginInfo);
    }
    if (result == VK_SUCCESS) {
        const VkBufferCopy copyRegion{
            .srcOffset = 0U,
            .dstOffset = offset,
            .size = size,
        };
        vkCmdCopyBuffer(cmd, stagingBuffer, m_buffer, 1U, &copyRegion);
        result = vkEndCommandBuffer(cmd);
    }

    VkFence fence = VK_NULL_HANDLE;
    if (result == VK_SUCCESS) {
        result = createFence(m_device, &fence);
    }
    if (result == VK_SUCCESS) {
        const VkCommandBufferSubmitInfo commandBufferInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = cmd,
            .deviceMask = 0U,
        };
        const VkSubmitInfo2 submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .flags = 0U,
            .waitSemaphoreInfoCount = 0U,
            .pWaitSemaphoreInfos = nullptr,
            .commandBufferInfoCount = 1U,
            .pCommandBufferInfos = &commandBufferInfo,
            .signalSemaphoreInfoCount = 0U,
            .pSignalSemaphoreInfos = nullptr,
        };
        result = vkQueueSubmit2(m_queue, 1U, &submitInfo, fence);
    }
    if (result == VK_SUCCESS) {
        result = vkWaitForFences(m_device, 1U, &fence, VK_TRUE, UINT64_MAX);
    }

    if (result != VK_SUCCESS) {
        Logger::error("Buffer staging upload failed: VkResult {}", static_cast<int>(result));
    }

    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(m_device, fence, nullptr);
    }
    if (cmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device, commandPool, 1U, &cmd);
    }
    vkDestroyCommandPool(m_device, commandPool, nullptr);
    vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
}

void Buffer::destroy() noexcept {
    // No explicit vmaUnmapMemory needed: all persistent maps use VMA_ALLOCATION_CREATE_MAPPED_BIT,
    // which VMA automatically unmaps when the allocation is destroyed via vmaDestroyBuffer.
    if (m_allocator != VK_NULL_HANDLE && m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
    }

    m_buffer = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
    m_allocator = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_queue = VK_NULL_HANDLE;
    m_size = 0U;
    m_queueFamily = 0U;
    m_mapped = nullptr;
}
