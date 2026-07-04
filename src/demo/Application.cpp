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

Application::~Application() {
    const VkDevice dev = deviceContext().device;
    if (dev == VK_NULL_HANDLE) {
        return;
    }
    // Harmonia's ~App() calls vkDeviceWaitIdle; wait here so async resources are idle.
    vkDeviceWaitIdle(dev);

    if (m_stagesCmdBufs[0] != VK_NULL_HANDLE) {
        commandPool().free(m_stagesCmdBufs[0]);
        m_stagesCmdBufs[0] = VK_NULL_HANDLE;
    }
    if (m_stagesCmdBufs[1] != VK_NULL_HANDLE) {
        commandPool().free(m_stagesCmdBufs[1]);
        m_stagesCmdBufs[1] = VK_NULL_HANDLE;
    }
    for (int i = 0; i < 2; ++i) {
        if (m_asyncFences[i] != VK_NULL_HANDLE) {
            vkDestroyFence(dev, m_asyncFences[i], nullptr);
            m_asyncFences[i] = VK_NULL_HANDLE;
        }
        if (m_asyncSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(dev, m_asyncSemaphores[i], nullptr);
            m_asyncSemaphores[i] = VK_NULL_HANDLE;
        }
        if (m_gfxDoneSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(dev, m_gfxDoneSemaphores[i], nullptr);
            m_gfxDoneSemaphores[i] = VK_NULL_HANDLE;
        }
    }
    if (m_asyncCmdPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(dev, m_asyncCmdPool, nullptr);
        m_asyncCmdPool = VK_NULL_HANDLE;
    }
}

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

    // LightCuller (Forward+ tile-based light culling)
    if (!m_lightCuller.initialize(deviceContext(), swapchain().extent().width, swapchain().extent().height)) {
        Logger::warn("LightCuller failed to initialize — direct loop fallback active");
    } else {
        m_renderer->setTileBuffers(m_lightCuller.tileLightCountsBuffer(),
                                   m_lightCuller.tileLightIndicesBuffer(),
                                   m_lightCuller.tilesX(),
                                   m_lightCuller.tilesY());
    }

    // GiPass — ray-query global illumination (multi-bounce indirect via the shared
    // Harmonia path integrator). This is the single indirect provider for the
    // unified path; opt out only for debugging via --no-rt-gi.
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

        // MotionVectorPass: compute per-pixel motion vectors after GiPass
        // (giBuffer will be in SHADER_READ_ONLY_OPTIMAL at that point).
        if (m_giPass.isInitialized()) {
            const MotionVectorPass::Config mvCfg{
                .width = swapchain().extent().width,
                .height = swapchain().extent().height,
                .giBufferImage = m_renderer->giBufferImage(),
                .giBufferView = m_renderer->giBufferView(),
            };
            if (!m_motionVectorPass.initialize(deviceContext(), mvCfg)) {
                Logger::warn("MotionVectorPass failed to initialize — static history fallback active");
            }
        }
    } else {
        m_renderer->setGiEnabled(false);
    }

    // Set up the async compute queue infrastructure when a dedicated compute family
    // is available. GiPass + MotionVectorPass will be dispatched there to overlap
    // with the next frame's raster work on the graphics queue.
    if (deviceContext().hasAsyncCompute()) {
        const VkDevice dev = deviceContext().device;
        bool ok = true;

        const VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = deviceContext().asyncComputeQueueFamily,
        };
        ok = ok && (vkCreateCommandPool(dev, &poolInfo, nullptr, &m_asyncCmdPool) == VK_SUCCESS);

        if (ok) {
            const VkCommandBufferAllocateInfo allocInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = m_asyncCmdPool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = static_cast<uint32_t>(m_asyncCmdBufs.size()),
            };
            ok = (vkAllocateCommandBuffers(dev, &allocInfo, m_asyncCmdBufs.data()) == VK_SUCCESS);
        }

        if (ok) {
            for (int i = 0; i < 2 && ok; ++i) {
                auto r = commandPool().allocate();
                if (r) {
                    m_stagesCmdBufs[i] = *r;
                } else {
                    ok = false;
                }
            }
        }

        constexpr VkSemaphoreCreateInfo semInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };
        constexpr VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        for (int i = 0; i < 2 && ok; ++i) {
            ok = ok && (vkCreateSemaphore(dev, &semInfo, nullptr, &m_gfxDoneSemaphores[i]) == VK_SUCCESS);
            ok = ok && (vkCreateSemaphore(dev, &semInfo, nullptr, &m_asyncSemaphores[i]) == VK_SUCCESS);
            ok = ok && (vkCreateFence(dev, &fenceInfo, nullptr, &m_asyncFences[i]) == VK_SUCCESS);
        }

        if (ok) {
            m_asyncComputeEnabled = true;
            Logger::info("Async compute queue available — GI + denoiser will overlap with raster");
        } else {
            Logger::warn("Async compute setup failed — falling back to single-queue path");
        }
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
    // Transposed VP for Slang mul(M, v) convention (used by motion vector shader).
    const glm::mat4 curViewProjT = glm::transpose(proj * view);

    // Camera-cut detection: disable the Hi-Z occlusion test for one frame after a large view
    // change so newly disoccluded meshlets (whose stale visibility is 0) are drawn instead of
    // wrongly culled. Uses view-direction angle + focus-relative translation (scale-independent).
    {
        const glm::vec3 curDir = glm::normalize(m_camera.target - m_camera.position);
        const float focusDist = glm::length(m_camera.target - m_camera.position);
        bool cut = !m_hiZPrevValid;
        if (m_hiZPrevValid) {
            const float posDelta = glm::length(m_camera.position - m_hiZPrevPos);
            const float cosAng = glm::clamp(glm::dot(curDir, m_hiZPrevDir), -1.0f, 1.0f);
            const float angle = std::acos(cosAng);
            cut = (angle > 0.1745f /* ~10 deg */) || (posDelta > 0.5f * std::max(focusDist, 1e-3f));
        }
        m_renderer->setHiZTestEnabled(!cut);
        m_hiZPrevPos = m_camera.position;
        m_hiZPrevDir = curDir;
        m_hiZPrevValid = true;
    }

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
    const uint32_t slot = frameIndex() % 2U;

    if (m_asyncComputeEnabled && giEnabled) {
        // ── Async path: GiPass + MotionVectorPass run on the dedicated compute queue ──
        // The graphics cmd (cmd) contains LightCuller + ForwardRenderer + release barriers.
        // onBeforeSceneStages() will end+submit cmd, then dispatch the async cmd, then open
        // m_stagesCmdBuf (with acquire barriers) for the denoiser to run into.

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
        gp.useA3Regularization = true;
        gp.useA3ChromaticImportance = true;
        // A3(b): wire the A-SVGF gradient/variance guide from the shared denoiser pass.
        gp.gradientVarianceView = denoiserGradientImageView();

        // Store motion vector params for use in onBeforeSceneStages() (runs on graphics queue).
        m_pendingMvp.curViewProj  = curViewProjT;
        m_pendingMvp.prevViewProj = m_prevViewProjValid ? m_prevViewProj : curViewProjT;
        m_pendingMvp.prevInstanceTransformBuffer = m_scene->prevInstanceTransformBuffer().handle();

        const uint32_t gfxFamily   = deviceContext().graphicsFamily;
        const uint32_t asyncFamily = deviceContext().asyncComputeQueueFamily;

        // GRAPHICS RELEASE: transfer giBuffer, gbuffer, hdr ownership to async compute.
        // The release barrier performs the layout transition (Vulkan spec §7.7.5: both release
        // and acquire must specify identical oldLayout/newLayout — the transition happens once
        // during the release operation). Acquire barriers use the same layout pair.
        const std::array<VkImageMemoryBarrier2, 3> gfxRelease{{
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_NONE, .dstAccessMask = VK_ACCESS_2_NONE,
             .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
             .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,  // matches asyncAcquire
             .srcQueueFamilyIndex = gfxFamily, .dstQueueFamilyIndex = asyncFamily,
             .image = m_renderer->giBufferImage(),
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_NONE, .dstAccessMask = VK_ACCESS_2_NONE,
             .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
             .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,  // matches asyncAcquire
             .srcQueueFamilyIndex = gfxFamily, .dstQueueFamilyIndex = asyncFamily,
             .image = m_renderer->gbufferImage(),
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_NONE, .dstAccessMask = VK_ACCESS_2_NONE,
             .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
             .newLayout = VK_IMAGE_LAYOUT_GENERAL,                   // matches asyncAcquire
             .srcQueueFamilyIndex = gfxFamily, .dstQueueFamilyIndex = asyncFamily,
             .image = target.image,
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        }};
        harmonia::pipelineBarrier(cmd, gfxRelease);

        // BUILD THE ASYNC COMMAND BUFFER.
        // The fence was signaled when the stages submit from 2 frames ago completed
        // (which waited on m_asyncSemaphores[slot]); so this wait is always instant.
        vkWaitForFences(deviceContext().device, 1, &m_asyncFences[slot], VK_TRUE, UINT64_MAX);
        vkResetFences(deviceContext().device, 1, &m_asyncFences[slot]);
        vkResetCommandBuffer(m_asyncCmdBufs[slot], 0);
        constexpr VkCommandBufferBeginInfo asyncBegin{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        vkBeginCommandBuffer(m_asyncCmdBufs[slot], &asyncBegin);
        const VkCommandBuffer asyncCmd = m_asyncCmdBufs[slot];

        // ASYNC ACQUIRE: take ownership and transition layouts so GiPass can run.
        const std::array<VkImageMemoryBarrier2, 3> asyncAcquire{{
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .srcAccessMask = VK_ACCESS_2_NONE,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
             .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .srcQueueFamilyIndex = gfxFamily, .dstQueueFamilyIndex = asyncFamily,
             .image = m_renderer->giBufferImage(),
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .srcAccessMask = VK_ACCESS_2_NONE,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
             .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .srcQueueFamilyIndex = gfxFamily, .dstQueueFamilyIndex = asyncFamily,
             .image = m_renderer->gbufferImage(),
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .srcAccessMask = VK_ACCESS_2_NONE,
             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
             .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
             .newLayout = VK_IMAGE_LAYOUT_GENERAL,
             .srcQueueFamilyIndex = gfxFamily, .dstQueueFamilyIndex = asyncFamily,
             .image = target.image,
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        }};
        harmonia::pipelineBarrier(asyncCmd, asyncAcquire);

        // GI ray-query: images are already in the correct layouts from the acquire barriers.
        m_giPass.record(asyncCmd, gp, /*skipPreBarriers=*/true);

        // MotionVectorPass runs on the graphics queue (in stagesCmdBuf) to keep
        // motionVectorImage in the graphics queue family. GiPass leaves giBuffer in
        // SHADER_READ_ONLY_OPTIMAL and gbuffer in SHADER_READ_ONLY_OPTIMAL.

        // ASYNC RELEASE: return ownership of the three shared images to the graphics queue.
        // Layout changes happen in the graphics ACQUIRE barriers.
        const std::array<VkImageMemoryBarrier2, 3> asyncRelease{{
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_NONE, .dstAccessMask = VK_ACCESS_2_NONE,
             .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .srcQueueFamilyIndex = asyncFamily, .dstQueueFamilyIndex = gfxFamily,
             .image = m_renderer->giBufferImage(),
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_NONE, .dstAccessMask = VK_ACCESS_2_NONE,
             .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .newLayout = VK_IMAGE_LAYOUT_GENERAL,                   // matches gfxAcquire
             .srcQueueFamilyIndex = asyncFamily, .dstQueueFamilyIndex = gfxFamily,
             .image = m_renderer->gbufferImage(),
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr,
             .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_NONE, .dstAccessMask = VK_ACCESS_2_NONE,
             .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
             .newLayout = VK_IMAGE_LAYOUT_GENERAL,
             .srcQueueFamilyIndex = asyncFamily, .dstQueueFamilyIndex = gfxFamily,
             .image = target.image,
             .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        }};
        harmonia::pipelineBarrier(asyncCmd, asyncRelease);
        vkEndCommandBuffer(asyncCmd);

        // Store per-frame state; onBeforeSceneStages() handles the submissions.
        m_prevViewProj      = curViewProjT;
        m_prevViewProjValid = true;
        return;
    }

    // ── Single-queue path (unchanged when async compute is unavailable or GI is off) ──
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
        // A3: firefly / chromatic-SSS noise reduction (Theia-only, flag-gated;
        // Hyperion stays the unbiased ground truth). Defaults ON.
        gp.useA3Regularization = true;      // A3(a): secondary-bounce roughness regularization
        gp.useA3ChromaticImportance = true; // A3(c): σ_t-weighted hero channel selection (unbiased)
        // A3(b): wire the A-SVGF gradient/variance guide from the shared denoiser pass.
        gp.gradientVarianceView = denoiserGradientImageView();
        m_giPass.record(cmd, gp);

        // After GiPass, the GI G-buffer is in SHADER_READ_ONLY_OPTIMAL.
        // Compute motion vectors for temporal reprojection in the denoiser.
        if (m_motionVectorPass.isInitialized()) {
            MotionVectorPass::FrameParams mvp{};
            mvp.curViewProj  = curViewProjT;
            mvp.prevViewProj = m_prevViewProjValid ? m_prevViewProj : curViewProjT;
            mvp.prevInstanceTransformBuffer = m_scene->prevInstanceTransformBuffer().handle();
            m_motionVectorPass.record(cmd, mvp);
        }
    }

    // Store current VP for use as prevViewProj next frame.
    m_prevViewProj      = curViewProjT;
    m_prevViewProjValid = true;

    // No compute pass wrote HDR this frame: leave HDR in GENERAL for host tonemap.
    if (!giEnabled) {
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
    if (giEnabled) {
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
        .pNext = nullptr,
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

    // Shut down GiPass and MotionVectorPass first — their descriptor sets reference the old
    // GBuffer image views that ForwardRenderer::resize() is about to destroy.
    m_giPass.shutdown();
    m_motionVectorPass.shutdown();

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

    // Reinitialize GiPass with new HDR/GBuffer views from the resized ForwardRenderer.
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

        // Reinitialize MotionVectorPass with the new GBuffer view and extent.
        if (m_giPass.isInitialized()) {
            const MotionVectorPass::Config mvCfg{
                .width = extent.width,
                .height = extent.height,
                .giBufferImage = m_renderer->giBufferImage(),
                .giBufferView = m_renderer->giBufferView(),
            };
            if (!m_motionVectorPass.initialize(deviceContext(), mvCfg)) {
                Logger::warn("MotionVectorPass resize failed — static history fallback active");
            }
        }
    } else {
        m_renderer->setGiEnabled(false);
    }

    // Invalidate the previous VP on resize so the first post-resize frame uses
    // curVP == prevVP (zero motion vectors), preventing a one-frame disocclusion spike.
    m_prevViewProjValid = false;
}

std::pair<VkCommandBuffer, VkSemaphore>
Application::onBeforeSceneStages(VkCommandBuffer renderCmd) noexcept {
    if (!m_asyncComputeEnabled || !giActive()) {
        return {renderCmd, VK_NULL_HANDLE};
    }

    const uint32_t slot        = frameIndex() % 2U;
    const uint32_t gfxFamily   = deviceContext().graphicsFamily;
    const uint32_t asyncFamily = deviceContext().asyncComputeQueueFamily;

    // ── Submit 1: graphics (raster + release barriers) ──
    vkEndCommandBuffer(renderCmd);
    const VkCommandBufferSubmitInfo gfxCmdInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = renderCmd,
        .deviceMask = 0,
    };
    const VkSemaphoreSubmitInfo gfxSignal{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = m_gfxDoneSemaphores[slot],
        .value = 0,
        .stageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        .deviceIndex = 0,
    };
    const VkSubmitInfo2 gfxSubmit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 0,
        .pWaitSemaphoreInfos = nullptr,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &gfxCmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &gfxSignal,
    };
    vkQueueSubmit2(deviceContext().graphicsQueue, 1, &gfxSubmit, VK_NULL_HANDLE);

    // ── Submit 2: async compute (GI + MotionVector) ──
    const VkCommandBufferSubmitInfo asyncCmdInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = m_asyncCmdBufs[slot],
        .deviceMask = 0,
    };
    const VkSemaphoreSubmitInfo asyncWaitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = m_gfxDoneSemaphores[slot],
        .value = 0,
        .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .deviceIndex = 0,
    };
    const VkSemaphoreSubmitInfo asyncSignal{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = m_asyncSemaphores[slot],
        .value = 0,
        .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .deviceIndex = 0,
    };
    const VkSubmitInfo2 asyncSubmit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &asyncWaitInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &asyncCmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &asyncSignal,
    };
    vkQueueSubmit2(deviceContext().asyncComputeQueue, 1, &asyncSubmit, m_asyncFences[slot]);

    // ── Open m_stagesCmdBufs[slot] (graphics family) for scene stages ──
    VkCommandBuffer stagesCmd = m_stagesCmdBufs[slot];
    vkResetCommandBuffer(stagesCmd, 0);
    constexpr VkCommandBufferBeginInfo stagesBegin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(stagesCmd, &stagesBegin);

    // GRAPHICS ACQUIRE: reclaim ownership of the shared images from the async queue.
    // Layout transitions happen here (in the acquire barriers), not in the release.
    const VkImageMemoryBarrier2 hdrAcquire{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask = VK_ACCESS_2_NONE,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = asyncFamily,
        .dstQueueFamilyIndex = gfxFamily,
        .image = hdrImage().handle(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    // gbuffer: was SHADER_READ_ONLY after GiPass; acquire and transition to GENERAL for denoiser.
    const VkImageMemoryBarrier2 gbufferAcquire{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask = VK_ACCESS_2_NONE,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = asyncFamily,
        .dstQueueFamilyIndex = gfxFamily,
        .image = m_renderer->gbufferImage(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    const VkImageMemoryBarrier2 giBufferAcquire{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask = VK_ACCESS_2_NONE,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = asyncFamily,
        .dstQueueFamilyIndex = gfxFamily,
        .image = m_renderer->giBufferImage(),
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    // depth stays on graphics queue — just the layout transition for the denoiser.
    const VkImageMemoryBarrier2 depthToGeneral{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = m_renderer->depthImage(),
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };

    const std::array<VkImageMemoryBarrier2, 4> gfxAcquire{hdrAcquire, gbufferAcquire, giBufferAcquire, depthToGeneral};
    harmonia::pipelineBarrier(stagesCmd, gfxAcquire);

    // MotionVectorPass runs on the graphics queue so motionVectorImage stays in the
    // graphics family. giBuffer is now SHADER_READ_ONLY_OPTIMAL on the graphics queue.
    if (m_motionVectorPass.isInitialized()) {
        m_motionVectorPass.record(stagesCmd, m_pendingMvp);
    }

    return {stagesCmd, m_asyncSemaphores[slot]};
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
