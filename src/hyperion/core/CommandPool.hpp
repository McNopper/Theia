#pragma once

#include <expected>

#include "hyperion/DeviceContext.hpp"

class CommandPool {
  public:
    [[nodiscard]] static std::expected<CommandPool, VkResult> create(const DeviceContext& ctx, uint32_t queueFamily);

    CommandPool() = default;
    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;
    CommandPool(CommandPool&& other) noexcept;
    CommandPool& operator=(CommandPool&& other) noexcept;
    ~CommandPool();

    [[nodiscard]] std::expected<VkCommandBuffer, VkResult> allocate() const;
    void free(VkCommandBuffer cmd) const noexcept;

    [[nodiscard]] std::expected<VkCommandBuffer, VkResult> beginOneShot() const;
    VkResult endOneShot(VkCommandBuffer cmd) const noexcept;

    [[nodiscard]] VkCommandPool handle() const noexcept { return m_pool; }

  private:
    void destroy() noexcept;

    VkDevice m_device = VK_NULL_HANDLE;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
};
