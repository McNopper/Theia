#pragma once

#include <volk/volk.h>

#include <glm/glm.hpp>

#include <memory>

#include "harmonia/app/App.hpp"
#include "harmonia/app/IRenderer.hpp"
#include "theia/renderer/ForwardRenderer.hpp"
#include "theia/renderer/IblPrecompute.hpp"
#include "theia/renderer/LightCuller.hpp"
#include "theia/renderer/SSRPass.hpp"
#include "theia/scene/Scene.hpp"

namespace theia {

/// Theia demo: the real-time forward renderer injected into the shared
/// harmonia::App host.
///
/// The host owns window/context/swapchain/HDR image/tonemap/present; this
/// class owns only what is rasterizer specific — ForwardRenderer, Forward+
/// light culling, SSR, IBL precompute, the Scene and the interactive camera
/// controller.  Per the host contract, record() produces a linear image in
/// the scene-referred working color space and leaves it in
/// VK_IMAGE_LAYOUT_GENERAL (SSR does this when post-fx is active; record()
/// issues the transition explicitly otherwise).
class Application final : public harmonia::App, public harmonia::IRenderer {
  public:
    // harmonia::IRenderer
    void record(VkCommandBuffer cmd, const harmonia::RenderTarget& target) noexcept override;
    void onResize(VkExtent2D extent) noexcept override;
    [[nodiscard]] VkPipelineStageFlags2 outputStageMask() const noexcept override {
        // Must name the *actual* final producer of the HDR write this frame.
        // SSR/SSAO compute writes the final pixels only when post-fx runs; with
        // --no-postfx (or SSR uninitialized) the forward graphics pass is last.
        return postFxActive() ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                              : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    [[nodiscard]] VkAccessFlags2 outputAccessMask() const noexcept override {
        return postFxActive() ? VK_ACCESS_2_SHADER_WRITE_BIT : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    [[nodiscard]] VkImageView gNormalView() const noexcept override { return m_renderer ? m_renderer->gbufferView() : VK_NULL_HANDLE; }
    [[nodiscard]] VkImageView gDepthView() const noexcept override { return m_renderer ? m_renderer->depthView() : VK_NULL_HANDLE; }
    [[nodiscard]] const char* name() const noexcept override { return "Theia ForwardRenderer"; }

  protected:
    // harmonia::App hooks
    [[nodiscard]] harmonia::IRenderer& renderer() noexcept override { return *this; }
    [[nodiscard]] ISceneBuilder& sceneBuilder() noexcept override { return *m_scene; }
    [[nodiscard]] bool onInitialize() override;
    [[nodiscard]] bool onSceneLoaded(const SceneLoader::SceneConfig& sceneConfig) override;
    void onSceneUnload() override;
    bool onEvent(const SDL_Event& event) override;
    void onUpdate(float dtSeconds) override;
    // Theia is a real-time renderer (no SPP accumulation); a small fixed
    // number of warmup frames lets light culling / SSR history settle.
    [[nodiscard]] uint32_t offscreenFrameCount() const noexcept override { return 4; }

  private:
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
        bool qDown = false; ///< Q = move down
        bool eDown = false; ///< E = move up
    };

    /// Compute initial yaw/pitch from a camera direction vector.
    static void directionToYawPitch(const glm::vec3& dir, float& yaw, float& pitch);

    /// Single source of truth for "are screen-space post-effects running this
    /// frame?". Both the dispatch decision in record() and the producer stage
    /// reported via outputStageMask() must agree, or the host's pre-tonemap
    /// barrier would name the wrong source stage.
    [[nodiscard]] bool postFxActive() const noexcept { return config().postProcess && m_ssrPass.isInitialized(); }

    std::unique_ptr<ForwardRenderer> m_renderer;
    LightCuller m_lightCuller;
    SSRPass m_ssrPass;
    IblPrecompute m_ibl;
    std::unique_ptr<Scene> m_scene = std::make_unique<Scene>();

    /// Live camera state (updated each frame by CameraController).
    ForwardRenderer::CameraParams m_camera{};
    CameraController m_camCtrl{};
};

} // namespace theia
