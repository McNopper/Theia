#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "demo/Application.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <string>

#include "harmonia/core/Logger.hpp"

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
    };
    if (!m_ssrPass.initialize(deviceContext(), ssrCfg)) {
        Logger::warn("SSRPass failed to initialize — reflections disabled");
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
        m_camera.ev100 = sceneConfig.cameraEv100.value_or(7.0f);
    }
    // Reset camera controller orientation to match the new target.
    const glm::vec3 dir = glm::normalize(m_camera.target - m_camera.position);
    directionToYawPitch(dir, m_camCtrl.yaw, m_camCtrl.pitch);

    // Scene-scale-aware near/far planes. Scenes range from sub-unit (ABeautifulGame)
    // to hundreds of units (Cornell), so a fixed near=1.0 clips small scenes entirely.
    // Derive both from the initial camera-to-target distance, which tracks scene scale.
    const float camDist = glm::length(m_camera.target - m_camera.position);
    m_camera.nearPlane = std::max(0.001f, camDist * 0.01f);
    m_camera.farPlane = std::max(10.0f, camDist * 1000.0f);
    // Camera move speed should also scale with the scene.
    m_camCtrl.speed = std::max(0.1f, camDist * 0.5f);

    m_renderer->setCamera(m_camera);

    Logger::info("Loaded scene: {} instances, {} bytes vb, {} bytes ib",
                 m_scene->instanceCount(),
                 m_scene->vertexBuffer().size(),
                 m_scene->indexBuffer().size());

    // IBL: the host loaded the environment probe (if any); precompute the
    // renderer-specific BRDF LUT / irradiance resources from it.
    const bool hasEnv = iblProbe().has_value() && iblProbe()->isValid();
    const VkImageView envView    = hasEnv ? iblProbe()->imageView() : VK_NULL_HANDLE;
    const VkSampler   envSampler = hasEnv ? iblProbe()->sampler()   : VK_NULL_HANDLE;
    const float       envNits    = sceneConfig.envUnitNits.value_or(1.0f);
    // CDF buffers for env importance sampling in the diffuse irradiance precompute.
    const VkBuffer marginalCdf    = hasEnv ? iblProbe()->marginalCdfBuffer().handle()    : VK_NULL_HANDLE;
    const VkBuffer conditionalCdf = hasEnv ? iblProbe()->conditionalCdfBuffer().handle() : VK_NULL_HANDLE;
    const uint32_t cdfW           = hasEnv ? iblProbe()->cdfWidth()  : 0u;
    const uint32_t cdfH           = hasEnv ? iblProbe()->cdfHeight() : 0u;

    if (!m_ibl.initialize(deviceContext(), commandPool(),
                          envView, envSampler, envNits,
                          marginalCdf, conditionalCdf, cdfW, cdfH)) {
        Logger::error("IBL precomputation failed");
        return false;
    }
    m_renderer->setIbl(m_ibl.resources(), envView, envNits);
    m_renderer->setHasEnvironment(hasEnv);
    if (hasEnv) {
        m_renderer->setSunShadow(iblProbe()->sunDirection(), iblProbe()->sunStrength());
    } else {
        m_renderer->setSunShadow(glm::vec3(0.0f, 1.0f, 0.0f), 0.0f);
    }
    return true;
}

void Application::record(VkCommandBuffer cmd, const harmonia::RenderTarget& target) noexcept {
    const float aspect = static_cast<float>(target.extent.width) / static_cast<float>(target.extent.height);

    // Forward+ light culling compute pass (runs before geometry rendering).
    if (m_scene && m_scene->lightCount() > 0 && m_lightCuller.tilesX() > 0) {
        glm::mat4 proj = glm::perspective(glm::radians(m_camera.vfovDeg), aspect, m_camera.nearPlane, m_camera.farPlane);
        proj[1][1] *= -1.0f;
        const glm::mat4 view = glm::lookAt(m_camera.position, m_camera.target, m_camera.up);
        m_lightCuller.dispatch(cmd,
                               m_scene->lightBuffer().handle(),
                               m_scene->lightCount(),
                               proj,
                               view,
                               m_camera.nearPlane,
                               m_camera.farPlane);
    }

    m_renderer->recordFrame(cmd);

    // SSR pass: linear ray march + composite into the HDR buffer.
    // Runs after the forward pass; transitions HDR to GENERAL and leaves it there.
    if (m_ssrPass.isInitialized()) {
        glm::mat4 proj = glm::perspective(glm::radians(m_camera.vfovDeg), aspect, m_camera.nearPlane, m_camera.farPlane);
        proj[1][1] *= -1.0f;
        m_ssrPass.dispatch(cmd, proj, glm::inverse(proj));
    }
}

void Application::onResize(VkExtent2D extent) noexcept {
    // ForwardRenderer / LightCuller / SSRPass cannot recreate their targets
    // yet — the window is created non-resizable (Config::resizable = false).
    static_cast<void>(extent);
    Logger::warn("Theia does not support resizing yet");
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
        // [ / ] keys: decrease / increase EV100
        case SDL_SCANCODE_LEFTBRACKET:
            if (down) {
                m_camera.ev100 -= 0.5f;
                Logger::info("EV100 = {:.1f}", m_camera.ev100);
            }
            return true;
        case SDL_SCANCODE_RIGHTBRACKET:
            if (down) {
                m_camera.ev100 += 0.5f;
                Logger::info("EV100 = {:.1f}", m_camera.ev100);
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
                            std::to_string(static_cast<int>(m_camera.ev100 * 10) / 10.0f).substr(0, 4) +
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
