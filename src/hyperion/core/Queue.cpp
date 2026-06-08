#include "hyperion/core/Queue.hpp"

#include <vector>

Queue::Queue(VkQueue queue, uint32_t family) noexcept : m_queue(queue), m_family(family) {}

VkResult Queue::submit(std::span<const VkCommandBuffer> cmds,
                       std::span<const VkSemaphoreSubmitInfo> waits,
                       std::span<const VkSemaphoreSubmitInfo> signals,
                       VkFence fence) const noexcept {
    if (m_queue == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::vector<VkCommandBufferSubmitInfo> commandInfos;
    commandInfos.reserve(cmds.size());

    for (const VkCommandBuffer cmd : cmds) {
        commandInfos.push_back(VkCommandBufferSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = cmd,
            .deviceMask = 0U,
        });
    }

    const VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0U,
        .waitSemaphoreInfoCount = static_cast<uint32_t>(waits.size()),
        .pWaitSemaphoreInfos = waits.data(),
        .commandBufferInfoCount = static_cast<uint32_t>(commandInfos.size()),
        .pCommandBufferInfos = commandInfos.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signals.size()),
        .pSignalSemaphoreInfos = signals.data(),
    };

    return vkQueueSubmit2(m_queue, 1U, &submitInfo, fence);
}

VkQueue Queue::handle() const noexcept {
    return m_queue;
}

uint32_t Queue::family() const noexcept {
    return m_family;
}
