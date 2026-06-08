#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "demo/Application.hpp"

#include <SDL3/SDL.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <imgui.h>
#include <span>
#include <string>

#include "demo/ImGuiLayer.hpp"
#include "demo/SceneLoader.hpp"
#include "hyperion/core/Logger.hpp"
#include "theia/renderer/ForwardRenderer.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace {

[[nodiscard]] VkResult createBinarySemaphore(VkDevice device, VkSemaphore& semaphore) {
    const VkSemaphoreCreateInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    return vkCreateSemaphore(device, &info, nullptr, &semaphore);
}

[[nodiscard]] VkResult createTimelineSemaphore(VkDevice device, VkSemaphore& semaphore) {
    const VkSemaphoreTypeCreateInfo typeInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    const VkSemaphoreCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &typeInfo,
    };
    return vkCreateSemaphore(device, &info, nullptr, &semaphore);
}

[[nodiscard]] VkImageMemoryBarrier2 imageBarrier(VkImage image,
                                                 VkImageLayout oldLayout,
                                                 VkImageLayout newLayout,
                                                 VkPipelineStageFlags2 srcStage,
                                                 VkAccessFlags2 srcAccess,
                                                 VkPipelineStageFlags2 dstStage,
                                                 VkAccessFlags2 dstAccess) {
    return VkImageMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
}

void pipelineBarrier(VkCommandBuffer cmd, std::span<const VkImageMemoryBarrier2> barriers) {
    const VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace

namespace theia {

Application::~Application() {
    shutdown();
}

int Application::run(const Config& config) {
    if (!initialize(config)) {
        return 1;
    }
    mainLoop();
    shutdown();
    return 0;
}

bool Application::initialize(const Config& config) {
    m_showUi = !config.hideUi;
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        Logger::error("Failed to initialize SDL");
        return false;
    }

    m_window = SDL_CreateWindow(config.title.c_str(), config.width, config.height, SDL_WINDOW_VULKAN);
    if (!m_window) {
        Logger::error("Failed to create SDL window");
        SDL_Quit();
        return false;
    }

    // Vulkan context
    auto contextResult = Context::create(Context::Config{
        .appName = "Theia",
        .appVersion = VK_MAKE_VERSION(0, 1, 0),
        .enableValidation = config.validation,
        .window = m_window,
    });
    if (!contextResult) {
        Logger::error("Failed to create Vulkan context: {}", static_cast<int>(contextResult.error()));
        return false;
    }
    m_context = std::make_unique<Context>(std::move(*contextResult));

    // Command pool
    auto cpResult = CommandPool::create(m_context->deviceContext(), m_context->deviceContext().graphicsFamily);
    if (!cpResult) {
        Logger::error("Failed to create command pool");
        return false;
    }
    m_commandPool = std::make_unique<CommandPool>(std::move(*cpResult));

    // Swapchain
    auto scResult = Swapchain::create(
        m_context->deviceContext(), m_context->surface(), VkExtent2D{config.width, config.height}, /*preferHDR=*/true);
    if (!scResult) {
        Logger::error("Failed to create swapchain: {}", static_cast<int>(scResult.error()));
        return false;
    }
    m_swapchain = std::make_unique<Swapchain>(std::move(*scResult));

    // Descriptors (shared pipeline layout — same as Hyperion, ToneMapper uses binding 1)
    auto descResult = Descriptors::create(m_context->deviceContext());
    if (!descResult) {
        Logger::error("Failed to create descriptors: {}", static_cast<int>(descResult.error()));
        return false;
    }
    m_descriptors = std::move(*descResult);

    // HDR image: STORAGE_BIT (ToneMapper), SAMPLED_BIT (SSR reads reflected colors),
    //            COLOR_ATTACHMENT_BIT (ForwardRenderer writes here).
    auto hdrResult =
        Image::create(m_context->deviceContext(),
                      m_swapchain->extent(),
                      VK_FORMAT_R32G32B32A32_SFLOAT,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT,
                      "theia.hdr");
    if (!hdrResult) {
        Logger::error("Failed to create HDR image: {}", static_cast<int>(hdrResult.error()));
        return false;
    }
    m_hdrImage = std::move(*hdrResult);

    // ToneMapper — uses Descriptors pipeline layout (same as Hyperion)
    auto tmResult = ToneMapper::create(m_context->deviceContext(),
                                       m_descriptors.pipelineLayout(),
                                       m_swapchain->format(),
                                       "spirv/tonemap_vert.spv",
                                       "spirv/tonemap.spv");
    if (!tmResult) {
        Logger::error("Failed to create ToneMapper: {}", static_cast<int>(tmResult.error()));
        return false;
    }
    m_toneMapper = std::move(*tmResult);

    // ForwardRenderer
    m_renderer = std::make_unique<ForwardRenderer>();
    if (!m_renderer->initialize(m_context->deviceContext(),
                                ForwardRenderer::Config{
                                    .width = m_swapchain->extent().width,
                                    .height = m_swapchain->extent().height,
                                    .outputFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
                                    .hdrImage = m_hdrImage.handle(),
                                    .hdrImageView = m_hdrImage.view(),
                                })) {
        Logger::error("Failed to initialize ForwardRenderer");
        return false;
    }

    // LightCuller (Forward+ tile-based light culling)
    if (!m_lightCuller.initialize(
            m_context->deviceContext(), m_swapchain->extent().width, m_swapchain->extent().height)) {
        Logger::warn("LightCuller failed to initialize — direct loop fallback active");
    } else {
        m_renderer->setTileBuffers(m_lightCuller.tileLightCountsBuffer(),
                                   m_lightCuller.tileLightIndicesBuffer(),
                                   m_lightCuller.tilesX(),
                                   m_lightCuller.tilesY());
    }

    // SSRPass — screen-space reflections (linear ray march + composite)
    {
        const SSRPass::Config ssrCfg{
            .width = m_swapchain->extent().width,
            .height = m_swapchain->extent().height,
            .depthImage = m_renderer->depthImage(),
            .depthView = m_renderer->depthView(),
            .gbufferImage = m_renderer->gbufferImage(),
            .gbufferView = m_renderer->gbufferView(),
            .hdrImage = m_hdrImage.handle(),
            .hdrView = m_hdrImage.view(),
        };
        if (!m_ssrPass.initialize(m_context->deviceContext(), ssrCfg)) {
            Logger::warn("SSRPass failed to initialize — reflections disabled");
        }
    }

    // Per-frame resources (double-buffered, same as Hyperion)
    for (auto& frame : m_frames) {
        auto rcmd = m_commandPool->allocate();
        if (!rcmd) {
            Logger::error("Failed to allocate render command buffer");
            return false;
        }
        frame.renderCmd = *rcmd;

        auto dcmd = m_commandPool->allocate();
        if (!dcmd) {
            Logger::error("Failed to allocate display command buffer");
            return false;
        }
        frame.displayCmd = *dcmd;

        if (createBinarySemaphore(m_context->deviceContext().device, frame.imageAvailable) != VK_SUCCESS) {
            Logger::error("Failed to create imageAvailable semaphore");
            return false;
        }
    }

    if (createTimelineSemaphore(m_context->deviceContext().device, m_timelineSemaphore) != VK_SUCCESS) {
        Logger::error("Failed to create timeline semaphore");
        return false;
    }

    // One renderComplete semaphore per swapchain image
    m_renderComplete.resize(m_swapchain->imageCount());
    for (auto& sem : m_renderComplete) {
        if (createBinarySemaphore(m_context->deviceContext().device, sem) != VK_SUCCESS) {
            Logger::error("Failed to create renderComplete semaphore");
            return false;
        }
    }

    m_swapchainLayouts.assign(m_swapchain->imageCount(), VK_IMAGE_LAYOUT_UNDEFINED);

    // Initialize Dear ImGui (SDL3 + Vulkan dynamic rendering)
    {
        const auto& dctx = m_context->deviceContext();
        ImGuiLayer::Config imguiCfg{
            .instance = m_context->instance(),
            .physDevice = dctx.physicalDevice,
            .device = dctx.device,
            .graphicsQueueFamily = dctx.graphicsFamily,
            .graphicsQueue = dctx.graphicsQueue,
            .swapchainColorFormat = m_swapchain->format(),
            .minImageCount = 2u,
            .imageCount = m_swapchain->imageCount(),
            .window = m_window,
        };
        if (!m_imgui.initialize(imguiCfg)) {
            Logger::warn("ImGui failed to initialize — running without UI");
        }
    }

    // Load scene
    const std::filesystem::path assetsDir = "assets";
    m_assetsDir = assetsDir;

    // Enumerate all .scene files for the scene switcher UI
    {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(assetsDir, ec)) {
            if (entry.path().extension() == ".scene") {
                m_sceneNames.push_back(entry.path().filename().string());
            }
        }
        std::sort(m_sceneNames.begin(), m_sceneNames.end());
    }

    std::string defaultScene = config.initialScene.empty()
                                   ? "cornell_classic.scene"
                                   : std::filesystem::path(config.initialScene).filename().string();
    // Tolerate a scene name supplied without the ".scene" extension (e.g. "--scene mr_sphere").
    if (!defaultScene.empty() && std::filesystem::path(defaultScene).extension() != ".scene") {
        defaultScene += ".scene";
    }
    {
        auto it = std::find(m_sceneNames.begin(), m_sceneNames.end(), defaultScene);
        m_selectedScene = it != m_sceneNames.end() ? static_cast<int>(std::distance(m_sceneNames.begin(), it)) : 0;
    }
    if (!m_sceneNames.empty()) {
        if (!loadScene(assetsDir / m_sceneNames[static_cast<size_t>(m_selectedScene)])) {
            Logger::warn("Failed to load default scene, continuing with empty scene");
        }
    }

    // Initialize controller yaw/pitch from scene camera direction.
    const glm::vec3 dir = glm::normalize(m_camera.target - m_camera.position);
    directionToYawPitch(dir, m_camCtrl.yaw, m_camCtrl.pitch);
    m_renderer->setCamera(m_camera);
    m_camCtrl.lastTick = SDL_GetTicksNS();

    m_running = true;
    return true;
}

bool Application::loadScene(const std::filesystem::path& sceneFile) {
    // Wait for GPU to finish using current scene resources before destroying them.
    if (m_context) {
        vkDeviceWaitIdle(m_context->deviceContext().device);
    }

    m_ibl.shutdown();
    m_envProbe.reset();
    m_scene.reset();
    if (m_renderer) {
        m_renderer->setScene(nullptr);
    }

    m_scene = std::make_unique<Scene>();
    SceneLoader loader;
    auto sceneConfig = loader.load(sceneFile, m_assetsDir, *m_scene, m_context->deviceContext(), *m_commandPool);
    if (!sceneConfig) {
        Logger::warn("SceneLoader: failed to load '{}'", sceneFile.string());
    } else if (m_scene->build(m_context->deviceContext(), *m_commandPool) != VK_SUCCESS) {
        Logger::warn("SceneLoader: failed to build TLAS for '{}'", sceneFile.string());
    } else {
        m_renderer->setScene(m_scene.get());

        if (sceneConfig->cameraPos && sceneConfig->cameraAt) {
            m_camera.position = *sceneConfig->cameraPos;
            m_camera.target = *sceneConfig->cameraAt;
            m_camera.up = sceneConfig->cameraUp.value_or(glm::vec3{0.0f, 1.0f, 0.0f});
            m_camera.vfovDeg = sceneConfig->cameraVfov.value_or(45.0f);
            m_camera.ev100 = sceneConfig->cameraEv100.value_or(7.0f);
        }
        // Reset camera controller orientation to match new target.
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

        Logger::info("Loaded '{}': {} instances, {} bytes vb, {} bytes ib",
                     sceneFile.filename().string(),
                     m_scene->instanceCount(),
                     m_scene->vertexBuffer().size(),
                     m_scene->indexBuffer().size());
    }

    VkImageView envView = VK_NULL_HANDLE;
    VkSampler envSampler = VK_NULL_HANDLE;
    if (sceneConfig && sceneConfig->envMapFile) {
        auto probeResult =
            IblProbe::loadFromEXR(m_context->deviceContext(), *m_commandPool, m_assetsDir / *sceneConfig->envMapFile);
        if (probeResult) {
            m_envProbe = std::move(*probeResult);
            envView = m_envProbe->imageView();
            envSampler = m_envProbe->sampler();
        } else {
            Logger::warn("Failed to load environment map '{}'", sceneConfig->envMapFile->string());
        }
    }

    if (!m_ibl.initialize(m_context->deviceContext(),
                          *m_commandPool,
                          envView,
                          envSampler,
                          sceneConfig ? sceneConfig->envUnitNits.value_or(1.0f) : 1.0f)) {
        Logger::error("IBL precomputation failed for '{}'", sceneFile.filename().string());
        return false;
    }
    m_renderer->setIbl(m_ibl.resources());
    m_renderer->setHasEnvironment(envView != VK_NULL_HANDLE);
    if (envView != VK_NULL_HANDLE && m_envProbe) {
        m_renderer->setSunShadow(m_envProbe->sunDirection(), m_envProbe->sunStrength());
    } else {
        m_renderer->setSunShadow(glm::vec3(0.0f, 1.0f, 0.0f), 0.0f);
    }
    return true;
}

void Application::shutdown() {
    if (m_context) {
        vkDeviceWaitIdle(m_context->deviceContext().device);
    }

    m_imgui.shutdown();
    m_renderer.reset();
    m_lightCuller.shutdown();
    m_ssrPass.shutdown();
    m_ibl.shutdown();
    m_envProbe.reset();

    if (m_context) {
        auto& dev = m_context->deviceContext().device;
        for (auto& frame : m_frames) {
            if (frame.imageAvailable != VK_NULL_HANDLE) {
                vkDestroySemaphore(dev, frame.imageAvailable, nullptr);
                frame.imageAvailable = VK_NULL_HANDLE;
            }
        }
        for (auto& sem : m_renderComplete) {
            if (sem != VK_NULL_HANDLE)
                vkDestroySemaphore(dev, sem, nullptr);
        }
        m_renderComplete.clear();
        if (m_timelineSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(dev, m_timelineSemaphore, nullptr);
            m_timelineSemaphore = VK_NULL_HANDLE;
        }
    }

    m_hdrImage = {};
    m_toneMapper = {};
    m_descriptors = {};
    m_commandPool.reset();
    m_swapchain.reset();
    m_context.reset();

    if (m_window) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }
    SDL_Quit();
}

void Application::directionToYawPitch(const glm::vec3& dir, float& yaw, float& pitch) {
    const glm::vec3 d = glm::normalize(dir);
    pitch = glm::degrees(std::asin(d.y));
    yaw = glm::degrees(std::atan2(d.z, d.x));
}

void Application::mainLoop() {
    SDL_Event event;
    while (m_running) {
        // ---- Input processing ----
        while (SDL_PollEvent(&event)) {
            // Route every event to ImGui first so it can capture keyboard/mouse
            m_imgui.processEvent(event);

            // Only handle camera/app input when ImGui is not consuming it
            const bool imguiWantsKbd = ImGui::GetIO().WantCaptureKeyboard;
            const bool imguiWantsMouse = ImGui::GetIO().WantCaptureMouse;

            switch (event.type) {
            case SDL_EVENT_QUIT:
                m_running = false;
                break;

            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP: {
                if (imguiWantsKbd)
                    break;
                const bool down = (event.type == SDL_EVENT_KEY_DOWN);
                switch (event.key.scancode) {
                case SDL_SCANCODE_W:
                    m_camCtrl.wDown = down;
                    break;
                case SDL_SCANCODE_A:
                    m_camCtrl.aDown = down;
                    break;
                case SDL_SCANCODE_S:
                    m_camCtrl.sDown = down;
                    break;
                case SDL_SCANCODE_D:
                    m_camCtrl.dDown = down;
                    break;
                case SDL_SCANCODE_Q:
                    m_camCtrl.qDown = down;
                    break;
                case SDL_SCANCODE_E:
                    m_camCtrl.eDown = down;
                    break;
                case SDL_SCANCODE_ESCAPE:
                    if (down) {
                        if (m_camCtrl.captured) {
                            SDL_SetWindowRelativeMouseMode(m_window, false);
                            m_camCtrl.captured = false;
                        } else {
                            m_running = false;
                        }
                    }
                    break;
                // [ / ] keys: decrease / increase EV100
                case SDL_SCANCODE_LEFTBRACKET:
                    if (down) {
                        m_camera.ev100 -= 0.5f;
                        Logger::info("EV100 = {:.1f}", m_camera.ev100);
                    }
                    break;
                case SDL_SCANCODE_RIGHTBRACKET:
                    if (down) {
                        m_camera.ev100 += 0.5f;
                        Logger::info("EV100 = {:.1f}", m_camera.ev100);
                    }
                    break;
                case SDL_SCANCODE_F1:
                    if (down) {
                        m_showUi = !m_showUi;
                    }
                    break;
                default:
                    break;
                }
                break;
            }

            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (!imguiWantsMouse && event.button.button == SDL_BUTTON_RIGHT) {
                    m_camCtrl.captured = true;
                    SDL_SetWindowRelativeMouseMode(m_window, true);
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    m_camCtrl.captured = false;
                    SDL_SetWindowRelativeMouseMode(m_window, false);
                }
                break;

            case SDL_EVENT_MOUSE_MOTION:
                if (m_camCtrl.captured) {
                    m_camCtrl.yaw += event.motion.xrel * m_camCtrl.sensitivity;
                    m_camCtrl.pitch -= event.motion.yrel * m_camCtrl.sensitivity;
                    // Clamp pitch to avoid gimbal lock at poles
                    m_camCtrl.pitch = std::max(-89.0f, std::min(89.0f, m_camCtrl.pitch));
                }
                break;

            case SDL_EVENT_MOUSE_WHEEL:
                if (!imguiWantsMouse) {
                    // Scroll up = faster, scroll down = slower
                    m_camCtrl.speed *= std::pow(1.2f, event.wheel.y);
                    m_camCtrl.speed = std::max(1.0f, std::min(5000.0f, m_camCtrl.speed));
                }
                break;

            case SDL_EVENT_WINDOW_RESIZED:
                handleResize(static_cast<uint32_t>(event.window.data1), static_cast<uint32_t>(event.window.data2));
                break;
            default:
                break;
            }
        }
        if (!m_running)
            continue;

        // ---- Camera controller update ----
        const uint64_t nowTick = SDL_GetTicksNS();
        const float dt = static_cast<float>(nowTick - m_camCtrl.lastTick) * 1e-9f;
        m_camCtrl.lastTick = nowTick;

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
            m_camera.position += forward * m_camCtrl.speed * dt;
        if (m_camCtrl.sDown)
            m_camera.position -= forward * m_camCtrl.speed * dt;
        if (m_camCtrl.dDown)
            m_camera.position += right * m_camCtrl.speed * dt;
        if (m_camCtrl.aDown)
            m_camera.position -= right * m_camCtrl.speed * dt;
        if (m_camCtrl.eDown)
            m_camera.position += up * m_camCtrl.speed * dt;
        if (m_camCtrl.qDown)
            m_camera.position -= up * m_camCtrl.speed * dt;

        m_camera.target = m_camera.position + forward;
        m_camera.up = worldUp;
        m_renderer->setCamera(m_camera);

        // ---- Dear ImGui: start frame + build settings UI ----
        m_imgui.beginFrame();
        if (m_imgui.isInitialized() && m_showUi) {
            // Performance overlay (top-left, small, transparent)
            const ImGuiWindowFlags overlayFlags =
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
            ImGui::SetNextWindowPos({10.0f, 10.0f}, ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.50f);
            if (ImGui::Begin("##perf", nullptr, overlayFlags)) {
                ImGui::Text("Theia  |  Real-Time Renderer");
                ImGui::Separator();
                ImGui::Text("%.1f FPS  (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
                if (m_scene) {
                    ImGui::Text("%u instances  |  %.1fk verts",
                                m_scene->instanceCount(),
                                static_cast<float>(m_scene->vertexBuffer().size()) / sizeof(float) / 12.0f / 1000.0f);
                }
            }
            ImGui::End();

            // Main settings window (right side)
            ImGui::SetNextWindowPos({static_cast<float>(m_swapchain->extent().width) - 310.0f, 10.0f},
                                    ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize({300.0f, 440.0f}, ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Theia Settings")) {
                // ── Rendering ─────────────────────────────────────────────
                if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::SliderFloat("EV100", &m_camera.ev100, -5.0f, 20.0f, "%.1f");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Physical camera exposure (EV100)\n"
                                          "Lower = brighter, Higher = darker\n"
                                          "Keyboard: [ / ]");
                    const float exp = 1.0f / (1.2f * std::pow(2.0f, m_camera.ev100));
                    ImGui::Text("  exposure multiplier: %.5f", exp);

                    if (m_ssrPass.isInitialized()) {
                        ImGui::Separator();
                        static float ssrStrength = 1.0f;
                        if (ImGui::SliderFloat("SSR Strength", &ssrStrength, 0.0f, 1.0f, "%.2f")) {
                            m_ssrPass.setSSRStrength(ssrStrength);
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Screen-Space Reflection blend weight.\n"
                                              "0 = IBL only, 1 = full SSR on smooth surfaces.");
                    }
                }

                // ── Camera ────────────────────────────────────────────────
                if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::SliderFloat(
                        "Move speed", &m_camCtrl.speed, 1.0f, 2000.0f, "%.0f u/s", ImGuiSliderFlags_Logarithmic);
                    ImGui::SliderFloat("Sensitivity", &m_camCtrl.sensitivity, 0.05f, 1.0f, "%.2f");
                    ImGui::SliderFloat("FOV", &m_camera.vfovDeg, 10.0f, 120.0f, "%.1f°");
                    ImGui::Separator();
                    ImGui::Text(
                        "Position: (%.1f, %.1f, %.1f)", m_camera.position.x, m_camera.position.y, m_camera.position.z);
                    ImGui::Text("Yaw: %.1f°  Pitch: %.1f°", m_camCtrl.yaw, m_camCtrl.pitch);
                }

                // ── Scene ─────────────────────────────────────────────────
                if (ImGui::CollapsingHeader("Scene")) {
                    // Scene picker dropdown
                    if (!m_sceneNames.empty()) {
                        const char* current = m_sceneNames[static_cast<size_t>(m_selectedScene)].c_str();
                        if (ImGui::BeginCombo("Scene file", current)) {
                            for (int i = 0; i < static_cast<int>(m_sceneNames.size()); ++i) {
                                const bool selected = (i == m_selectedScene);
                                if (ImGui::Selectable(m_sceneNames[static_cast<size_t>(i)].c_str(), selected)) {
                                    if (i != m_selectedScene) {
                                        m_selectedScene = i;
                                        loadScene(m_assetsDir / m_sceneNames[static_cast<size_t>(i)]);
                                    }
                                }
                                if (selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                    }
                    ImGui::Separator();
                    if (m_scene) {
                        ImGui::Text("Instances:        %u", m_scene->instanceCount());
                        ImGui::Text("Lights:           %u", m_scene->lightCount());
                        ImGui::Text("Emissive tris:    %u", m_scene->emissiveTriangleCount());
                        const uint64_t vbKB = m_scene->vertexBuffer().size() / 1024;
                        const uint64_t ibKB = m_scene->indexBuffer().size() / 1024;
                        ImGui::Text("Vertex buffer:    %llu KB", vbKB);
                        ImGui::Text("Index buffer:     %llu KB", ibKB);
                    } else {
                        ImGui::TextDisabled("No scene loaded");
                    }
                }

                // ── Controls ──────────────────────────────────────────────
                if (ImGui::CollapsingHeader("Controls")) {
                    ImGui::TextDisabled("RMB drag       — look around");
                    ImGui::TextDisabled("WASD           — move");
                    ImGui::TextDisabled("Q / E          — move down / up");
                    ImGui::TextDisabled("Scroll wheel   — adjust move speed");
                    ImGui::TextDisabled("[ / ]          — EV100 -/+0.5");
                    ImGui::TextDisabled("Escape         — release mouse / quit");
                }
            }
            ImGui::End();
        }

        const uint32_t slot = m_currentFrame;
        FrameResources& frame = m_frames[slot];

        // Wait for this frame slot to be free
        if (frame.completionValue > 0) {
            const VkSemaphoreWaitInfo waitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .semaphoreCount = 1,
                .pSemaphores = &m_timelineSemaphore,
                .pValues = &frame.completionValue,
            };
            vkWaitSemaphores(m_context->deviceContext().device, &waitInfo, UINT64_MAX);
        }

        // Submit scene render — signals timeline when HDR image is ready
        const uint64_t renderValue = renderFrame();

        // Acquire swapchain image
        uint32_t imageIndex = 0;
        VkResult result = m_swapchain->acquireNextImage(frame.imageAvailable, imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            handleResize(m_swapchain->extent().width, m_swapchain->extent().height);
            continue;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            Logger::error("Swapchain acquire failed: {}", static_cast<int>(result));
            continue;
        }

        // Record display command: barriers + tonemap + present barrier
        vkResetCommandBuffer(frame.displayCmd, 0);
        const VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        if (vkBeginCommandBuffer(frame.displayCmd, &beginInfo) != VK_SUCCESS)
            continue;

        const std::array preToneMapBarriers{
            // HDR: SSRPass left it in GENERAL; memory sync only for ToneMapper read.
            imageBarrier(m_hdrImage.handle(),
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT),
            imageBarrier(m_swapchain->image(imageIndex),
                         m_swapchainLayouts[imageIndex],
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_NONE,
                         0,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
        };
        pipelineBarrier(frame.displayCmd, preToneMapBarriers);

        m_toneMapper.record(frame.displayCmd,
                            m_hdrImage.view(),
                            m_swapchain->imageView(imageIndex),
                            m_swapchain->extent(),
                            m_swapchain->outputColorSpace());

        // Render ImGui over the tonemapped image (swapchain still in COLOR_ATTACHMENT_OPTIMAL)
        m_imgui.render(frame.displayCmd, m_swapchain->imageView(imageIndex), m_swapchain->extent());

        const std::array presentBarrier{
            imageBarrier(m_swapchain->image(imageIndex),
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_NONE,
                         0),
        };
        pipelineBarrier(frame.displayCmd, presentBarrier);

        if (vkEndCommandBuffer(frame.displayCmd) != VK_SUCCESS)
            continue;

        // Submit display: wait on renderValue + imageAvailable; signal renderComplete + next timeline
        const uint64_t displayValue = m_nextTimelineValue++;

        const std::array<VkSemaphoreSubmitInfo, 2> waitInfos{{
            {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
             .semaphore = m_timelineSemaphore,
             .value = renderValue,
             .stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT},
            {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
             .semaphore = frame.imageAvailable,
             .value = 0,
             .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
        }};
        const std::array<VkSemaphoreSubmitInfo, 2> signalInfos{{
            {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
             .semaphore = m_renderComplete[imageIndex],
             .value = 0,
             .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
            {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
             .semaphore = m_timelineSemaphore,
             .value = displayValue,
             .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
        }};
        const VkCommandBufferSubmitInfo displayCmdInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = frame.displayCmd,
        };
        const VkSubmitInfo2 displaySubmit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount = 2,
            .pWaitSemaphoreInfos = waitInfos.data(),
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &displayCmdInfo,
            .signalSemaphoreInfoCount = 2,
            .pSignalSemaphoreInfos = signalInfos.data(),
        };
        result = vkQueueSubmit2(m_context->deviceContext().graphicsQueue, 1, &displaySubmit, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) {
            Logger::error("Display submit failed: {}", static_cast<int>(result));
            continue;
        }

        frame.completionValue = displayValue;

        // Present
        result =
            m_swapchain->present(m_context->deviceContext().graphicsQueue, imageIndex, m_renderComplete[imageIndex]);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            handleResize(m_swapchain->extent().width, m_swapchain->extent().height);
        } else if (result != VK_SUCCESS) {
            Logger::error("Present failed: {}", static_cast<int>(result));
        }
        m_swapchainLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        m_currentFrame = (m_currentFrame + 1) % 2;
        ++m_frameIndex;

        // Update window title with FPS every second
        static uint64_t fpsLastTick = SDL_GetTicksNS();
        static uint32_t fpsFrames = 0;
        ++fpsFrames;
        const uint64_t fpsDelta = SDL_GetTicksNS() - fpsLastTick;
        if (fpsDelta >= 1'000'000'000ULL) {
            const float fps = static_cast<float>(fpsFrames) * 1e9f / static_cast<float>(fpsDelta);
            const float ms = 1000.0f / fps;
            SDL_SetWindowTitle(m_window,
                               (std::string("Theia  |  ") + std::to_string(static_cast<int>(fps)) + " FPS  " +
                                std::to_string(static_cast<int>(ms * 10) / 10.0f).substr(0, 4) + " ms" + "  |  EV100 " +
                                std::to_string(static_cast<int>(m_camera.ev100 * 10) / 10.0f).substr(0, 4) +
                                "  |  RMB = look  WASD = move  [/] = EV100")
                                   .c_str());
            fpsFrames = 0;
            fpsLastTick = SDL_GetTicksNS();
        }
    }
}

uint64_t Application::renderFrame() {
    const uint32_t slot = m_currentFrame;
    FrameResources& frame = m_frames[slot];

    vkResetCommandBuffer(frame.renderCmd, 0);
    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(frame.renderCmd, &beginInfo) != VK_SUCCESS) {
        return m_nextTimelineValue - 1;
    }

    // Forward+ light culling compute pass (runs before geometry rendering).
    if (m_scene && m_scene->lightCount() > 0 && m_lightCuller.tilesX() > 0) {
        const float aspect =
            static_cast<float>(m_swapchain->extent().width) / static_cast<float>(m_swapchain->extent().height);
        glm::mat4 proj =
            glm::perspective(glm::radians(m_camera.vfovDeg), aspect, m_camera.nearPlane, m_camera.farPlane);
        proj[1][1] *= -1.0f;
        const glm::mat4 view = glm::lookAt(m_camera.position, m_camera.target, m_camera.up);
        m_lightCuller.dispatch(frame.renderCmd,
                               m_scene->lightBuffer().handle(),
                               m_scene->lightCount(),
                               proj,
                               view,
                               m_camera.nearPlane,
                               m_camera.farPlane);
    }

    m_renderer->recordFrame(frame.renderCmd);

    // SSR pass: linear ray march + composite into HDR buffer.
    // Runs after forward pass; transitions HDR to GENERAL and leaves it there.
    if (m_ssrPass.isInitialized()) {
        const float aspect =
            static_cast<float>(m_swapchain->extent().width) / static_cast<float>(m_swapchain->extent().height);
        glm::mat4 proj =
            glm::perspective(glm::radians(m_camera.vfovDeg), aspect, m_camera.nearPlane, m_camera.farPlane);
        proj[1][1] *= -1.0f;
        m_ssrPass.dispatch(frame.renderCmd, proj, glm::inverse(proj));
    }

    if (vkEndCommandBuffer(frame.renderCmd) != VK_SUCCESS) {
        return m_nextTimelineValue - 1;
    }

    const uint64_t renderValue = m_nextTimelineValue++;
    const VkCommandBufferSubmitInfo cmdInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = frame.renderCmd,
    };
    const VkSemaphoreSubmitInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = m_timelineSemaphore,
        .value = renderValue,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    const VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalInfo,
    };
    vkQueueSubmit2(m_context->deviceContext().graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

    return renderValue;
}

void Application::handleResize(uint32_t w, uint32_t h) {
    vkDeviceWaitIdle(m_context->deviceContext().device);
    if (m_swapchain->recreate(VkExtent2D{w, h}) == VK_SUCCESS) {
        m_swapchainLayouts.assign(m_swapchain->imageCount(), VK_IMAGE_LAYOUT_UNDEFINED);
    }
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
