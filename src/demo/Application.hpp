#pragma once

#include <algorithm>
#include <volk/volk.h>

#include <glm/glm.hpp>

#include <memory>

#include "harmonia/app/App.hpp"
#include "harmonia/app/IRenderer.hpp"
#include "theia/renderer/ForwardRenderer.hpp"
#include "theia/renderer/GiPass.hpp"
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
    void setCameraJitterEnabled(bool enabled) noexcept { m_cameraJitterEnabled = enabled; }
    // harmonia::IRenderer
    void record(VkCommandBuffer cmd, const harmonia::RenderTarget& target) noexcept override;
    void onResize(VkExtent2D extent) noexcept override;
    [[nodiscard]] VkPipelineStageFlags2 outputStageMask() const noexcept override {
        // Must name the *actual* final producer of the HDR write this frame.
        // SSR/SSAO compute writes the final pixels only when post-fx runs; the GI
        // compute stage writes them when active; otherwise the forward graphics pass.
        return computeWritesHdr() ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                  : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    [[nodiscard]] VkAccessFlags2 outputAccessMask() const noexcept override {
        return computeWritesHdr() ? VK_ACCESS_2_SHADER_WRITE_BIT : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
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
    // Theia is a real-time renderer: default offscreen warmup is 4 frames, but
    // allow overrides via the shared host flag `--offscreen-frames`.
    [[nodiscard]] uint32_t offscreenFrameCount() const noexcept override {
        return std::max(config().offscreenFrames, 1U);
    }

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

    /// The ray-query GI compute stage runs as the indirect-lighting provider in the
    /// non-post-fx (parity) path. GI and SSR are mutually exclusive HDR writers. GI takes
    /// precedence when enabled and initialized; otherwise SSR owns the composite.
    [[nodiscard]] bool giActive() const noexcept {
        return config().rtGi && m_giPass.isInitialized() && !postFxActive();
    }

    /// SSR only owns HDR when enabled and GI is not currently active.
    [[nodiscard]] bool postFxActive() const noexcept {
        return config().postProcess && m_ssrPass.isInitialized() && !(config().rtGi && m_giPass.isInitialized());
    }

    /// True when a compute stage (SSR or GI) is the final writer of the HDR image this
    /// frame, so the host's pre-tonemap barrier names COMPUTE rather than the raster pass.
    [[nodiscard]] bool computeWritesHdr() const noexcept { return postFxActive() || giActive(); }

    std::unique_ptr<ForwardRenderer> m_renderer;
    LightCuller m_lightCuller;
    SSRPass m_ssrPass;
    GiPass m_giPass;
    IblPrecompute m_ibl;
    std::unique_ptr<Scene> m_scene = std::make_unique<Scene>();

    /// Environment state captured at scene-load for the per-frame GI dispatch.
    VkImageView m_envView = VK_NULL_HANDLE;
    VkSampler m_envSampler = VK_NULL_HANDLE;
    VkBuffer m_envMarginalCdf = VK_NULL_HANDLE;
    VkBuffer m_envConditionalCdf = VK_NULL_HANDLE;
    uint32_t m_envCdfWidth = 0;
    uint32_t m_envCdfHeight = 0;
    bool m_hasEnv = false;
    float m_envNits = 1.0f;

    /// Live camera state (updated each frame by CameraController).
    ForwardRenderer::CameraParams m_camera{};
    CameraController m_camCtrl{};
    bool m_cameraJitterEnabled = true;

    // Previous view signature, used to reset progressive accumulation on change.
    glm::vec3 m_prevCamPos{0.0f};
    glm::vec3 m_prevCamTarget{0.0f};
    glm::vec3 m_prevCamUp{0.0f};
    float m_prevEv100 = 0.0f;
    bool m_viewSigValid = false;
};

} // namespace theia
