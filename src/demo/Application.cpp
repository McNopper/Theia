#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "demo/Application.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "harmonia/core/Barrier.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "theia/renderer/CameraJitter.hpp"

namespace theia {

bool Application::onInitialize() {
    // ForwardRenderer draws into the host's linear HDR working-space image.
    m_renderer = std::make_unique<ForwardRenderer>();
    if (!m_renderer->initialize(deviceContext(),
                                ForwardRenderer::Config{
                                    .width = swapchain().extent().width,
                                    .height = swapchain().extent().height,
                                    .outputFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
                                    .hdrImage = hdrImage().handle(),
                                    .hdrImageView = hdrImage().view(),
                                })) {
        Logger::error("Failed to initialize ForwardRenderer");
        return false;
    }
    m_renderer->setIndirectAmbient(config().indirectAmbient);
    m_renderer->setRngDebug(config().rngDebug);
    m_renderer->setTransparentEnvLodDiagnostic(config().diagTransparentEnvLod);
    m_renderer->setCameraJitterEnabled(m_cameraJitterEnabled);
    if (!m_cameraJitterEnabled) {
        Logger::info("Camera jitter disabled (--no-camera-jitter)");
    }

    // Progressive accumulation in the interactive window: a stationary camera
    // converges the per-frame stochastic samples (camera jitter, stochastic
    // transparency/env sampling, RT-GI) into a stable image instead of showing
    // each raw sample (which otherwise reads as a shimmering / shaking preview).
    // onUpdate() calls resetAccumulation() whenever the view changes. The
    // offscreen capture path accumulates regardless of this flag.
    setInteractiveAccumulation(true);
    if (config().diagTransparentEnvLod) {
        Logger::warn("DIAGNOSTIC enabled: transparent env taps use deterministic roughness/ray-cone LOD");
    }

    // LightCuller (Forward+ tile-based light culling)
    if (!m_lightCuller.initialize(deviceContext(), swapchain().extent().width, swapchain().extent().height)) {
        Logger::warn("LightCuller failed to initialize — direct loop fallback active");
    } else {
        m_renderer->setTileBuffers(m_lightCuller.tileLightCountsBuffer(),
                                   m_lightCuller.tileLightIndicesBuffer(),
                                   m_lightCuller.tilesX(),
                                   m_lightCuller.tilesY());
    }

    // SSRPass — screen-space reflections (linear ray march + composite)
    const SSRPass::Config ssrCfg{
        .width = swapchain().extent().width,
        .height = swapchain().extent().height,
        .depthImage = m_renderer->depthImage(),
        .depthView = m_renderer->depthView(),
        .gbufferImage = m_renderer->gbufferImage(),
        .gbufferView = m_renderer->gbufferView(),
        .hdrImage = hdrImage().handle(),
        .hdrView = hdrImage().view(),
        .ssgiStrength = config().ssgiStrength,
    };
    if (!m_ssrPass.initialize(deviceContext(), ssrCfg)) {
        Logger::warn("SSRPass failed to initialize — reflections disabled");
    }

    // GiPass — ray-query global illumination (multi-bounce indirect via the shared
    // Harmonia path integrator). Runs in the non-post-fx (parity) path as the indirect
    // lighting provider, replacing the forward pass's flat-IBL approximation. Opt-in via
    // --rt-gi; off by default so non-GI scenes keep evalIBL + direct lighting unchanged.
    if (config().rtGi) {
        const GiPass::Config giCfg{
            .width = swapchain().extent().width,
            .height = swapchain().extent().height,
            .hdrImage = hdrImage().handle(),
            .hdrView = hdrImage().view(),
            .giBufferImage = m_renderer->giBufferImage(),
            .giBufferView = m_renderer->giBufferView(),
            .gbufferImage = m_renderer->gbufferImage(),
            .gbufferView = m_renderer->gbufferView(),
        };
        if (!m_giPass.initialize(deviceContext(), giCfg)) {
            Logger::warn("GiPass failed to initialize — ray-query GI disabled");
        }
        // When GI will run, the forward pass must emit direct+emission only (no IBL/ambient);
        // the GI compute stage supplies the indirect term.
        m_renderer->setGiEnabled(giActive());
    } else {
        m_renderer->setGiEnabled(false);
    }

    return true;
}

void Application::onSceneUnload() {
    m_ibl.shutdown();
    if (m_renderer) {
        m_renderer->setScene(nullptr);
    }
    m_scene = std::make_unique<Scene>();
}

bool Application::onSceneLoaded(const SceneLoader::SceneConfig& sceneConfig) {
    if (m_scene->build(deviceContext(), commandPool()) != VK_SUCCESS) {
        Logger::error("Failed to build scene TLAS");
        return false;
    }
    m_renderer->setScene(m_scene.get());

    if (sceneConfig.cameraPos && sceneConfig.cameraAt) {
        m_camera.position = *sceneConfig.cameraPos;
        m_camera.target = *sceneConfig.cameraAt;
        m_camera.up = sceneConfig.cameraUp.value_or(glm::vec3{0.0f, 1.0f, 0.0f});
        m_camera.vfovDeg = sceneConfig.cameraVfov.value_or(45.0f);
    }
    // Physical camera exposure — same pattern as Hyperion (aperture=1, ISO=100).
    {
        const float ev100 = sceneConfig.cameraEv100.value_or(7.0f);
        m_camera.physical.aperture = 1.0f;
        m_camera.physical.iso = 100.0f;
        m_camera.physical.shutterSpeedHz = std::pow(2.0f, ev100);
    }
    // Reset camera controller orientation to match the new target.
    const glm::vec3 dir = glm::normalize(m_camera.target - m_camera.position);
    directionToYawPitch(dir, m_camCtrl.yaw, m_camCtrl.pitch);

    // Scene-scale-aware near/far — shared helper from harmonia::Camera.
    const float camDist = glm::length(m_camera.target - m_camera.position);
    std::tie(m_camera.nearPlane, m_camera.farPlane) = Camera::nearFarFromDistance(camDist);
    // Camera move speed also scales with the scene.
    m_camCtrl.speed = std::max(0.1f, camDist * 0.5f);

    m_renderer->setCamera(m_camera);
    m_sceneMaxDepth = sceneConfig.maxDepth.value_or(3u);
    m_renderer->setTransparentMaxDepth(sceneConfig.maxDepth.value_or(2u));

    Logger::info("Loaded scene: {} instances, {} bytes vb, {} bytes ib",
                 m_scene->instanceCount(),
                 m_scene->vertexBuffer().size(),
                 m_scene->indexBuffer().size());

    // IBL: the host loaded the environment probe (if any); precompute the
    // renderer-specific BRDF LUT / irradiance resources from it.
    const bool hasEnv = iblProbe().has_value() && iblProbe()->isValid();
    const VkImageView envView = hasEnv ? iblProbe()->imageView() : VK_NULL_HANDLE;
    const VkSampler envSampler = hasEnv ? iblProbe()->sampler() : VK_NULL_HANDLE;
    const float envNits = sceneConfig.envUnitNits.value_or(1.0f);
    // CDF buffers for env importance sampling in the diffuse irradiance precompute.
    const VkBuffer marginalCdf = hasEnv ? iblProbe()->marginalCdfBuffer().handle() : VK_NULL_HANDLE;
    const VkBuffer conditionalCdf = hasEnv ? iblProbe()->conditionalCdfBuffer().handle() : VK_NULL_HANDLE;
    const uint32_t cdfW = hasEnv ? iblProbe()->cdfWidth() : 0u;
    const uint32_t cdfH = hasEnv ? iblProbe()->cdfHeight() : 0u;
    const uint32_t diffuseRes = std::max(1u, config().iblDiffuseResolution);
    const VkExtent2D diffuseExtent{diffuseRes, std::max(1u, diffuseRes / 2u)};

    if (!m_ibl.initialize(
            deviceContext(),
            commandPool(),
            envView,
            envSampler,
            envNits,
            diffuseExtent,
            marginalCdf,
            conditionalCdf,
            cdfW,
            cdfH)) {
        Logger::error("IBL precomputation failed");
        return false;
    }
    m_renderer->setIbl(m_ibl.resources(), envView, envNits);
    m_renderer->setEnvImportanceSampling(marginalCdf, conditionalCdf, cdfW, cdfH);
    m_renderer->setHasEnvironment(hasEnv);

    // Capture environment state for the per-frame GI dispatch. When no env map is bound,
    // fall back to the (black) IBL specular texture/sampler so the GI descriptor stays
    // valid — the shader gates all env reads on hasEnvMap.
    m_hasEnv = hasEnv;
    m_envNits = envNits;
    m_envView = hasEnv ? envView : m_ibl.resources().specularMipped.view();
    m_envSampler = hasEnv ? envSampler : m_ibl.resources().envSampler;
    m_envMarginalCdf = marginalCdf;
    m_envConditionalCdf = conditionalCdf;
    m_envCdfWidth = cdfW;
    m_envCdfHeight = cdfH;
    if (hasEnv) {
        m_renderer->setSunShadow(iblProbe()->sunDirection(), iblProbe()->sunStrength());
    } else {
        m_renderer->setSunShadow(glm::vec3(0.0f, 1.0f, 0.0f), 0.0f);
    }
    return true;
}

void Application::record(VkCommandBuffer cmd, const harmonia::RenderTarget& target) noexcept {
    m_renderer->setRngState(frameIndex(), config().rngSeed, config().deterministicReplay);

    const float aspect = static_cast<float>(target.extent.width) / static_cast<float>(target.extent.height);
    glm::mat4 proj = glm::perspective(glm::radians(m_camera.vfovDeg), aspect, m_camera.nearPlane, m_camera.farPlane);
    proj[1][1] *= -1.0f;
    if (m_cameraJitterEnabled) {
        proj = applyProjectionJitter(proj, cameraJitterNdc(frameIndex(), target.extent.width, target.extent.height));
    }
    const glm::mat4 view = glm::lookAt(m_camera.position, m_camera.target, m_camera.up);

    // Forward+ light culling compute pass (runs before geometry rendering).
    if (m_scene && m_scene->lightCount() > 0 && m_lightCuller.tilesX() > 0) {
        m_lightCuller.dispatch(cmd,
                               m_scene->lightBuffer().handle(),
                               m_scene->lightCount(),
                               proj,
                               view,
                               m_camera.nearPlane,
                               m_camera.farPlane);
    }

    m_renderer->recordFrame(cmd);

    const bool giEnabled = giActive();
    const bool postFxEnabled = postFxActive();

    if (giEnabled) {
        // Ray-query GI: walks BSDF continuation paths off the forward G-buffer and
        // additively composites indirect radiance into the HDR image.
        GiPass::FrameParams gp{};
        gp.scene = m_scene.get();
        gp.envMapView = m_envView;
        gp.envSampler = m_envSampler;
        gp.envMarginalCdf = m_envMarginalCdf;
        gp.envConditionalCdf = m_envConditionalCdf;
        gp.envImportanceWidth = m_envCdfWidth;
        gp.envImportanceHeight = m_envCdfHeight;
        gp.hasEnvMap = m_hasEnv;
        gp.envLuminanceScale = m_envNits;
        gp.viewTransposed = glm::transpose(view);
        gp.cameraPos = m_camera.position;
        gp.exposure = m_camera.physical.exposure();
        gp.frameSampleIndex = frameIndex();
        gp.rngBaseSeed = config().rngSeed;
        gp.maxDepth = m_sceneMaxDepth;
        m_giPass.record(cmd, gp);
    }

    // SSR pass: linear ray march + composite into the HDR buffer.
    // Runs after the forward pass; transitions HDR to GENERAL and leaves it there.
    // Skipped under --no-postfx (parity comparison contract: SSR/SSAO/bloom off).
    if (postFxEnabled) {
        m_ssrPass.dispatch(cmd, proj, glm::inverse(proj), giEnabled);
    } else if (!giEnabled) {
        // No compute post-effects: leave HDR in GENERAL for the host tonemap pass.
        const std::array hdrToGeneral{harmonia::imageBarrier(target.image,
                                                             VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                                             VK_IMAGE_LAYOUT_GENERAL,
                                                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                             VK_ACCESS_2_SHADER_READ_BIT)};
        harmonia::pipelineBarrier(cmd, hdrToGeneral);
    }

    // The Harmonia denoiser stage always samples the G-buffer normal and depth
    // as edge-stopping guides in VK_IMAGE_LAYOUT_GENERAL. After the renderer's
    // own passes these images sit in a path-dependent layout, so transition both
    // to GENERAL here regardless of the path (post-fx, RT-GI or forward-only).
    // The next frame re-initialises them from UNDEFINED, so leaving them in
    // GENERAL is safe.
    VkImageLayout gbufferOld = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    VkPipelineStageFlags2 gbufferSrcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkAccessFlags2 gbufferSrcAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    VkImageLayout depthOld = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    VkPipelineStageFlags2 depthSrcStage =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    VkAccessFlags2 depthSrcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (postFxEnabled) {
        // SSR sampled both guides as read-only and left them in that layout.
        gbufferOld = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        gbufferSrcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        gbufferSrcAccess = VK_ACCESS_2_SHADER_READ_BIT;
        depthOld = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthSrcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        depthSrcAccess = VK_ACCESS_2_SHADER_READ_BIT;
    } else if (giEnabled) {
        // The GI compute pass sampled the gbuffer as read-only; depth stayed a
        // write attachment (the GI pass does not read it).
        gbufferOld = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        gbufferSrcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        gbufferSrcAccess = VK_ACCESS_2_SHADER_READ_BIT;
    }

    const VkImageMemoryBarrier2 gbufferToGeneral =
        harmonia::imageBarrier(m_renderer->gbufferImage(), gbufferOld, VK_IMAGE_LAYOUT_GENERAL, gbufferSrcStage,
                               gbufferSrcAccess, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    const VkImageMemoryBarrier2 depthToGeneral{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = depthSrcStage,
        .srcAccessMask = depthSrcAccess,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = depthOld,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = m_renderer->depthImage(),
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    const std::array guideToGeneral{gbufferToGeneral, depthToGeneral};
    harmonia::pipelineBarrier(cmd, guideToGeneral);
}

void Application::onResize(VkExtent2D extent) noexcept {
    if (!m_renderer) {
        return;
    }

    // Shut down SSRPass and GiPass first — their descriptor sets reference the old
    // depth/GBuffer image views that ForwardRenderer::resize() is about to destroy.
    m_ssrPass.shutdown();
    m_giPass.shutdown();

    // Resize ForwardRenderer: recreates depth/GBuffer at new extent, updates
    // the HDR image handles (App::handleResize has already recreated the HDR image).
    if (!m_renderer->resize(extent.width, extent.height, hdrImage().handle(), hdrImage().view())) {
        Logger::error("ForwardRenderer resize failed");
        return;
    }

    // Reinitialize LightCuller for the new tile grid.
    m_lightCuller.shutdown();
    if (!m_lightCuller.initialize(deviceContext(), extent.width, extent.height)) {
        Logger::warn("LightCuller resize failed — direct loop fallback active");
    } else {
        m_renderer->setTileBuffers(m_lightCuller.tileLightCountsBuffer(),
                                   m_lightCuller.tileLightIndicesBuffer(),
                                   m_lightCuller.tilesX(),
                                   m_lightCuller.tilesY());
    }

    // Reinitialize SSRPass with new depth/GBuffer views from the resized ForwardRenderer.
    const SSRPass::Config ssrCfg{
        .width = extent.width,
        .height = extent.height,
        .depthImage = m_renderer->depthImage(),
        .depthView = m_renderer->depthView(),
        .gbufferImage = m_renderer->gbufferImage(),
        .gbufferView = m_renderer->gbufferView(),
        .hdrImage = hdrImage().handle(),
        .hdrView = hdrImage().view(),
        .ssgiStrength = config().ssgiStrength,
    };
    if (!m_ssrPass.initialize(deviceContext(), ssrCfg)) {
        Logger::warn("SSRPass resize failed — reflections disabled");
    }

    // Reinitialize GiPass with new HDR/GBuffer views from the resized ForwardRenderer.
    // Enabled by default; disable via --no-rt-gi. GI remains the non-postfx path.
    if (config().rtGi) {
        const GiPass::Config giCfg{
            .width = extent.width,
            .height = extent.height,
            .hdrImage = hdrImage().handle(),
            .hdrView = hdrImage().view(),
            .giBufferImage = m_renderer->giBufferImage(),
            .giBufferView = m_renderer->giBufferView(),
            .gbufferImage = m_renderer->gbufferImage(),
            .gbufferView = m_renderer->gbufferView(),
        };
        if (!m_giPass.initialize(deviceContext(), giCfg)) {
            Logger::warn("GiPass resize failed — ray-query GI disabled");
        }
        m_renderer->setGiEnabled(giActive());
    } else {
        m_renderer->setGiEnabled(false);
    }
}

bool Application::onEvent(const SDL_Event& event) {
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        const bool down = (event.type == SDL_EVENT_KEY_DOWN);
        switch (event.key.scancode) {
        case SDL_SCANCODE_W:
            m_camCtrl.wDown = down;
            return true;
        case SDL_SCANCODE_A:
            m_camCtrl.aDown = down;
            return true;
        case SDL_SCANCODE_S:
            m_camCtrl.sDown = down;
            return true;
        case SDL_SCANCODE_D:
            m_camCtrl.dDown = down;
            return true;
        case SDL_SCANCODE_Q:
            m_camCtrl.qDown = down;
            return true;
        case SDL_SCANCODE_E:
            m_camCtrl.eDown = down;
            return true;
        case SDL_SCANCODE_ESCAPE:
            // ESC releases mouse capture first; quitting is the host default.
            if (down && m_camCtrl.captured) {
                SDL_SetWindowRelativeMouseMode(window(), false);
                m_camCtrl.captured = false;
                return true;
            }
            return false;
        // [ / ] keys: decrease / increase EV100 by 0.5 stops
        case SDL_SCANCODE_LEFTBRACKET:
            if (down) {
                const float ev100 = m_camera.physical.ev100() - 0.5f;
                m_camera.physical.shutterSpeedHz = std::pow(2.0f, ev100);
                Logger::info("EV100 = {:.1f}", m_camera.physical.ev100());
            }
            return true;
        case SDL_SCANCODE_RIGHTBRACKET:
            if (down) {
                const float ev100 = m_camera.physical.ev100() + 0.5f;
                m_camera.physical.shutterSpeedHz = std::pow(2.0f, ev100);
                Logger::info("EV100 = {:.1f}", m_camera.physical.ev100());
            }
            return true;
        default:
            return false;
        }
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (event.button.button == SDL_BUTTON_RIGHT) {
            m_camCtrl.captured = true;
            SDL_SetWindowRelativeMouseMode(window(), true);
            return true;
        }
        return false;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (event.button.button == SDL_BUTTON_RIGHT) {
            m_camCtrl.captured = false;
            SDL_SetWindowRelativeMouseMode(window(), false);
            return true;
        }
        return false;

    case SDL_EVENT_MOUSE_MOTION:
        if (m_camCtrl.captured) {
            m_camCtrl.yaw += event.motion.xrel * m_camCtrl.sensitivity;
            m_camCtrl.pitch -= event.motion.yrel * m_camCtrl.sensitivity;
            // Clamp pitch to avoid gimbal lock at poles
            m_camCtrl.pitch = std::max(-89.0f, std::min(89.0f, m_camCtrl.pitch));
            return true;
        }
        return false;

    case SDL_EVENT_MOUSE_WHEEL:
        // Scroll up = faster, scroll down = slower
        m_camCtrl.speed *= std::pow(1.2f, event.wheel.y);
        m_camCtrl.speed = std::max(1.0f, std::min(5000.0f, m_camCtrl.speed));
        return true;

    default:
        return false;
    }
}

void Application::onUpdate(float dtSeconds) {
    // Rebuild forward direction from yaw/pitch
    const float yawRad = glm::radians(m_camCtrl.yaw);
    const float pitchRad = glm::radians(m_camCtrl.pitch);
    const glm::vec3 forward{
        std::cos(yawRad) * std::cos(pitchRad),
        std::sin(pitchRad),
        std::sin(yawRad) * std::cos(pitchRad),
    };
    const glm::vec3 worldUp{0.0f, 1.0f, 0.0f};
    const glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
    const glm::vec3 up = glm::normalize(glm::cross(right, forward));

    if (m_camCtrl.wDown)
        m_camera.position += forward * m_camCtrl.speed * dtSeconds;
    if (m_camCtrl.sDown)
        m_camera.position -= forward * m_camCtrl.speed * dtSeconds;
    if (m_camCtrl.dDown)
        m_camera.position += right * m_camCtrl.speed * dtSeconds;
    if (m_camCtrl.aDown)
        m_camera.position -= right * m_camCtrl.speed * dtSeconds;
    if (m_camCtrl.eDown)
        m_camera.position += up * m_camCtrl.speed * dtSeconds;
    if (m_camCtrl.qDown)
        m_camera.position -= up * m_camCtrl.speed * dtSeconds;

    m_camera.target = m_camera.position + forward;
    m_camera.up = worldUp;
    m_renderer->setCamera(m_camera);

    // Restart progressive accumulation whenever the view changes so a moving
    // camera doesn't smear across viewpoints and a stationary one converges.
    const float ev100 = m_camera.physical.ev100();
    const bool viewChanged = !m_viewSigValid || m_camera.position != m_prevCamPos ||
                             m_camera.target != m_prevCamTarget || m_camera.up != m_prevCamUp ||
                             ev100 != m_prevEv100;
    if (viewChanged) {
        resetAccumulation();
        m_prevCamPos = m_camera.position;
        m_prevCamTarget = m_camera.target;
        m_prevCamUp = m_camera.up;
        m_prevEv100 = ev100;
        m_viewSigValid = true;
    }

    // Update window title with FPS every second
    static uint64_t fpsLastTick = SDL_GetTicksNS();
    static uint32_t fpsFrames = 0;
    ++fpsFrames;
    const uint64_t fpsDelta = SDL_GetTicksNS() - fpsLastTick;
    if (fpsDelta >= 1'000'000'000ULL) {
        const float fps = static_cast<float>(fpsFrames) * 1e9f / static_cast<float>(fpsDelta);
        const float ms = 1000.0f / fps;
        SDL_SetWindowTitle(window(),
                           (std::string("Theia  |  ") + std::to_string(static_cast<int>(fps)) + " FPS  " +
                            std::to_string(static_cast<int>(ms * 10) / 10.0f).substr(0, 4) + " ms" + "  |  EV100 " +
                            std::to_string(static_cast<int>(m_camera.physical.ev100() * 10) / 10.0f).substr(0, 4) +
                            "  |  RMB = look  WASD = move  [/] = EV100")
                               .c_str());
        fpsFrames = 0;
        fpsLastTick = SDL_GetTicksNS();
    }
}

void Application::directionToYawPitch(const glm::vec3& dir, float& yaw, float& pitch) {
    const glm::vec3 d = glm::normalize(dir);
    pitch = glm::degrees(std::asin(d.y));
    yaw = glm::degrees(std::atan2(d.z, d.x));
}

} // namespace theia
