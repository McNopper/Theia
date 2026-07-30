#ifndef DEMO_APPLICATION_HPP
#define DEMO_APPLICATION_HPP

#include <volk/volk.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <slang-math/slang-math.hpp>

#include "harmonia/app/App.hpp"
#include "harmonia/app/IRenderer.hpp"
#include "theia/renderer/EnvImportanceResources.hpp"
#include "theia/renderer/ForwardRenderer.hpp"
#include "theia/renderer/GiPass.hpp"
#include "theia/renderer/IblPrecompute.hpp"
#include "theia/renderer/LightCuller.hpp"
#include "theia/renderer/MotionVectorPass.hpp"
#include "theia/renderer/TaaPass.hpp"
#include "theia/scene/Scene.hpp"

namespace theia {

/// Theia demo: the real-time forward renderer injected into the shared
/// harmonia::App host.
///
/// The host owns window/context/swapchain/HDR image/tonemap/present; this
/// class owns only what is rasterizer specific — ForwardRenderer, Forward+
/// light culling, IBL precompute, the Scene and the interactive camera
/// controller.  Per the host contract, record() produces a linear image in
/// the scene-referred working color space and leaves it in
/// VK_IMAGE_LAYOUT_GENERAL.
class Application final : public harmonia::App, public harmonia::IRenderer {
  public:
    ~Application() override;
    void setCameraJitterEnabled(bool enabled) noexcept { m_cameraJitterEnabled = enabled; }
    /// A4: toggle ReSTIR DI spatiotemporal reservoir resampling (default on). When off the
    /// forward pass keeps ownership of emissive direct lighting (bit-identical pre-A4 path).
    void setRestirDiEnabled(bool enabled) noexcept { m_useRestirDi = enabled; }
    /// GI2: toggle ReSTIR PT Enhanced — the unified DI+GI path reservoir (default on).
    /// Mutually exclusive with ReSTIR DI: when PT is on, DI is forced off and the path
    /// integrator owns primary emissive NEE (Hyperion bit-identical in expectation).
    void setRestirPtEnabled(bool enabled) noexcept { m_useRestirPt = enabled; }
    /// GI2 full PT: toggle the multi-bounce path reservoir for the indirect term
    /// (default on, only effective with PT). Off → per-sample multi-bounce walk.
    void setRestirPtPathEnabled(bool enabled) noexcept { m_useRestirPtPath = enabled; }
    void setTaaEnabled(bool enabled) noexcept { m_useTaa = enabled; }
    // harmonia::IRenderer
    void record(VkCommandBuffer cmd, const harmonia::RenderTarget& target) noexcept override;
    void onResize(VkExtent2D extent) noexcept override;
    [[nodiscard]] VkPipelineStageFlags2 outputStageMask() const noexcept override {
        // Must name the *actual* final producer of the HDR write this frame.
        // GI compute writes final pixels when active; otherwise the forward graphics pass.
        return giActive() ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    [[nodiscard]] VkAccessFlags2 outputAccessMask() const noexcept override {
        return giActive() ? VK_ACCESS_2_SHADER_WRITE_BIT : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    [[nodiscard]] VkImageView gNormalView() const noexcept override {
        return m_renderer ? m_renderer->gbufferView() : VK_NULL_HANDLE;
    }
    [[nodiscard]] VkImageView gDepthView() const noexcept override {
        return m_renderer ? m_renderer->depthView() : VK_NULL_HANDLE;
    }
    [[nodiscard]] VkImageView motionVectorView() const noexcept override {
        return m_motionVectorPass.isInitialized() ? m_motionVectorPass.motionVectorImageView() : VK_NULL_HANDLE;
    }
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
    [[nodiscard]] std::uint32_t offscreenFrameCount() const noexcept override {
        return std::max(config().offscreenFrames, 1U);
    }
    [[nodiscard]] std::pair<VkCommandBuffer, VkSemaphore>
    onBeforeSceneStages(VkCommandBuffer renderCmd) noexcept override;

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
    static void directionToYawPitch(const sm::float3& dir, float& yaw, float& pitch);

    struct CameraMatrices {
        sm::float4x4 proj;
        sm::float4x4 view;
        sm::float4x4 viewProj;
    };
    [[nodiscard]] CameraMatrices computeCameraMatrices(const harmonia::RenderTarget& target) const noexcept;
    void dispatchLightCull(VkCommandBuffer cmd, const sm::float4x4& proj, const sm::float4x4& view) noexcept;
    void transitionGbuffersForRecord(VkCommandBuffer cmd, VkImage hdrImage, bool giEnabled) noexcept;
    void submitAsyncCompute(VkCommandBuffer cmd,
                            const harmonia::RenderTarget& target,
                            const sm::float4x4& view,
                            const sm::float4x4& curViewProj) noexcept;
    void submitSingleQueueGI(VkCommandBuffer cmd, const sm::float4x4& view, const sm::float4x4& curViewProj) noexcept;

    /// True in offscreen capture mode (--output set). Offscreen rendering integrates many
    /// jittered/stochastic samples via progressive accumulation, which is incompatible with
    /// TAA's temporal-reprojection blend — so TAA is never initialized or run in this mode.
    [[nodiscard]] bool isOffscreenCapture() const noexcept { return !config().outputFile.empty(); }

    /// The ray-query GI compute stage runs whenever enabled and initialized.
    [[nodiscard]] bool giActive() const noexcept { return config().rtGi && m_giPass.isInitialized(); }

    std::unique_ptr<ForwardRenderer> m_renderer;
    LightCuller m_lightCuller;
    GiPass m_giPass;
    MotionVectorPass m_motionVectorPass;
    TaaPass m_taaPass;
    IblPrecompute m_ibl;
    std::unique_ptr<Scene> m_scene = std::make_unique<Scene>();

    /// Environment importance-sampling state captured at scene-load for the per-frame GI dispatch.
    EnvImportanceResources m_env;

    /// Live camera state (updated each frame by CameraController).
    ForwardRenderer::CameraParams m_camera{};
    CameraController m_camCtrl{};
    bool m_cameraJitterEnabled = true;
    /// A4: ReSTIR DI spatiotemporal reservoir resampling for emissive-triangle direct
    /// lighting (default on). When active the forward pass skips its emissive-derived
    /// rect lights (setRestirDiActive) so GiPass owns that term without double-counting.
    bool m_useRestirDi = true;
    /// GI2: ReSTIR PT Enhanced — unified DI+GI path reservoir (default on). Mutually
    /// exclusive with m_useRestirDi: when PT is active the path integrator owns primary
    /// emissive NEE and ReSTIR DI is skipped. The forward pass's rect-light skip is
    /// shared (same semantic — GiPass owns emissive direct either way).
    bool m_useRestirPt = true;
    /// GI2 full PT: multi-bounce path reservoir for the indirect term (default on).
    /// Gated by m_useRestirPt at the GiPass wiring (legacy DI mode has no path reservoir).
    bool m_useRestirPtPath = true;
    bool m_useTaa = true;
    std::uint32_t m_sceneMaxDepth = 3u;

    /// Previous frame's row-major view-projection matrix for motion vector computation.
    sm::float4x4 m_prevViewProj{1.0f};
    bool m_prevViewProjValid = false;

    /// True when the camera view changed this frame (fresh, non-accumulated sample). Gates the
    /// interactive-window TAA pass: TAA runs only during motion, never on a converged static view.
    bool m_cameraMoving = false;

    // Previous view signature, used to reset progressive accumulation on change.
    sm::float3 m_prevCamPos{0.0f, 0.0f, 0.0f};
    sm::float3 m_prevCamTarget{0.0f, 0.0f, 0.0f};
    sm::float3 m_prevCamUp{0.0f, 0.0f, 0.0f};
    float m_prevEv100 = 0.0f;
    bool m_viewSigValid = false;

    // Camera-cut detection for two-pass Hi-Z occlusion culling: on a large view change the
    // previous-frame visibility set is stale, so the Hi-Z test is disabled for that frame and
    // all remaining meshlets are drawn conservatively (prevents culling disoccluded geometry).
    sm::float3 m_hiZPrevPos{0.0f, 0.0f, 0.0f};
    sm::float3 m_hiZPrevDir{0.0f, 0.0f, -1.0f};
    bool m_hiZPrevValid = false;

    // ── Async compute resources ──────────────────────────────────────────────
    // When the device exposes a dedicated compute queue family (COMPUTE only),
    // GiPass is dispatched there while the next frame's raster work runs on
    // the graphics queue simultaneously. MotionVectorPass runs in the graphics
    // stages cmd to keep motionVectorImage on the graphics queue family.
    VkCommandPool m_asyncCmdPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, 2> m_asyncCmdBufs = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    /// Per-slot graphics-family cmd bufs for the scene stages (denoiser etc.) in async mode.
    std::array<VkCommandBuffer, 2> m_stagesCmdBufs = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkSemaphore, 2> m_gfxDoneSemaphores = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkSemaphore, 2> m_asyncSemaphores = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkFence, 2> m_asyncFences = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    bool m_asyncComputeEnabled = false;
    /// Pending motion-vector params set in record() and consumed by onBeforeSceneStages().
    MotionVectorPass::FrameParams m_pendingMvp{};
};

} // namespace theia
#endif // DEMO_APPLICATION_HPP
