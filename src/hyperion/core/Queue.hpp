#pragma once

#include <volk/volk.h>

#include <span>

class Queue {
  public:
    explicit Queue(VkQueue queue, uint32_t family) noexcept;

    VkResult submit(std::span<const VkCommandBuffer> cmds,
                    std::span<const VkSemaphoreSubmitInfo> waits = {},
                    std::span<const VkSemaphoreSubmitInfo> signals = {},
                    VkFence fence = VK_NULL_HANDLE) const noexcept;

    [[nodiscard]] VkQueue handle() const noexcept;
    [[nodiscard]] uint32_t family() const noexcept;

  private:
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_family = 0;
};
