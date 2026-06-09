#pragma once

#include <volk/volk.h>

#include <SDL3/SDL.h>

#include <glm/glm.hpp>

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "demo/ImGuiLayer.hpp"
#include "demo/presentation/Swapchain.hpp"
#include "demo/presentation/ToneMapper.hpp"
#include "demo/vulkan_init/Context.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/core/Image.hpp"
#include "hyperion/renderer/Descriptors.hpp"
#include "hyperion/scene/IblProbe.hpp"
#include "hyperion/scene/Scene.hpp"
#include "theia/renderer/ForwardRenderer.hpp"
#include "theia/renderer/IblPrecompute.hpp"
#include "theia/renderer/LightCuller.hpp"
#include "theia/renderer/SSRPass.hpp"

namespace theia {

class Application {
  public:
    struct Config {
        std::string title = "Theia -- Real-Time Renderer";
        uint32_t width = 1024;
        uint32_t height = 768;
        bool validation = false;
        std::string initialScene; ///< optional: path or bare filename in assets/
        bool hideUi = false;      ///< start with the ImGui overlay hidden (toggle with F1)
    };

    Application() = default;
    ~Application();

    int run(const Config& config);

  private:
    /// Per-frame-in-flight resources (double-buffered, same pattern as Hyperion).
    struct FrameResources {
        VkCommandBuffer renderCmd{};  ///< scene geometry recording
        VkCommandBuffer displayCmd{}; ///< tonemap recording
        VkSemaphore imageAvailable{}; ///< binary: signalled when swapchain image acquired
        uint64_t completionValue{};   ///< timeline value signalled when this slot completes
    };

    /// First-person camera controller (WASD + right-mouse-drag + scroll).
    struct CameraController {
        float yaw = -90.0f;        ///< degrees, -90 = looking toward -Z (Cornell default)
        float pitch = 0.0f;        ///< degrees, clamped ±89
        float speed = 50.0f;       ///< units/second (Cornell box ~560 units tall)
        float sensitivity = 0.15f; ///< degrees per pixel of mouse movement
        bool captured = false;     ///< whether relative mouse mode is active
        bool wDown = false;
        bool aDown = false;
        bool sDown = false;
        bool dDown = false;
        bool qDown = false;    ///< Q = move down
        bool eDown = false;    ///< E = move up
        uint64_t lastTick = 0; ///< SDL_GetTicksNS from previous frame
    };

    bool initialize(const Config& config);
    void shutdown();
    void mainLoop();

    /// Submit scene rendering into m_hdrImage; returns timeline value signalled on completion.
    uint64_t renderFrame();
    void handleResize(uint32_t w, uint32_t h);

    /// Load (or reload) a scene file.  Waits for GPU idle, destroys existing GPU
    /// scene/IBL resources, then re-creates them from the new file.
    bool loadScene(const std::filesystem::path& sceneFile);

    /// Compute initial yaw/pitch from a camera direction vector.
    static void directionToYawPitch(const glm::vec3& dir, float& yaw, float& pitch);

    SDL_Window* m_window = nullptr;
    std::unique_ptr<Context> m_context;
    std::unique_ptr<Swapchain> m_swapchain;
    Descriptors m_descriptors;
    ToneMapper m_toneMapper;
    std::unique_ptr<ForwardRenderer> m_renderer;
    std::unique_ptr<CommandPool> m_commandPool;
    std::unique_ptr<Scene> m_scene;
    std::optional<IblProbe> m_envProbe;
    IblPrecompute m_ibl;
    LightCuller m_lightCuller;
    SSRPass m_ssrPass;

    Image m_hdrImage;

    std::array<FrameResources, 2> m_frames{};
    /// One binary semaphore per swapchain image — signalled by display submit,
    /// consumed by vkQueuePresentKHR.
    std::vector<VkSemaphore> m_renderComplete;
    VkSemaphore m_timelineSemaphore = VK_NULL_HANDLE;
    uint64_t m_nextTimelineValue = 1;
    uint32_t m_currentFrame = 0;
    uint32_t m_frameIndex = 0;
    std::vector<VkImageLayout> m_swapchainLayouts;

    /// Live camera state (updated each frame by CameraController).
    ForwardRenderer::CameraParams m_camera{};
    CameraController m_camCtrl{};
    ImGuiLayer m_imgui{};

    /// Tone mapper selected by the current scene (matches tonemap.slang switch;
    /// 0 = ACES, 1 = AgX, 2 = Reinhard, 3 = Hable). Default ACES.
    uint32_t m_tonemapper = 0;

    // Scene switching
    std::filesystem::path m_assetsDir;
    std::vector<std::string> m_sceneNames; ///< just filenames (not full paths)
    int m_selectedScene = 0;               ///< index into m_sceneNames for ImGui combo

    bool m_running = false;
    bool m_showUi = true; ///< toggled with F1; hides the ImGui overlay for clean comparisons
};

} // namespace theia
