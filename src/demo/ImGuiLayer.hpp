#pragma once

#include <volk/volk.h>

#include <SDL3/SDL.h>

#include <cstdint>

namespace theia {

/// Thin Vulkan+SDL3 integration layer for Dear ImGui.
///
/// Lifecycle:
///   initialize() → once per window/device  (before first frame)
///   processEvent() → per SDL_Event in the event pump
///   beginFrame()  → once per frame, before building UI
///   render()      → records ImGui draw calls into a command buffer;
///                   the swapchain image view must be in COLOR_ATTACHMENT_OPTIMAL
///   shutdown()    → on cleanup
class ImGuiLayer {
  public:
    struct Config {
        VkInstance instance;
        VkPhysicalDevice physDevice;
        VkDevice device;
        uint32_t graphicsQueueFamily;
        VkQueue graphicsQueue;
        VkFormat swapchainColorFormat; ///< for dynamic rendering pipeline creation
        uint32_t minImageCount;
        uint32_t imageCount;
        SDL_Window* window;
    };

    /// Returns true on success.
    bool initialize(const Config& cfg);
    void shutdown();

    /// Route an SDL event to ImGui (call before your own event handling).
    void processEvent(const SDL_Event& e);

    /// Start a new ImGui frame.  Call once per frame after processEvent calls.
    void beginFrame();

    /// Record ImGui draw calls into cmd.
    /// @param swapchainView  View of the current swapchain image (COLOR_ATTACHMENT_OPTIMAL).
    /// @param extent         Swapchain extent.
    void render(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent);

    /// True after a successful initialize() and before shutdown().
    [[nodiscard]] bool isInitialized() const noexcept { return m_initialized; }

  private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    bool m_initialized = false;
};

} // namespace theia
