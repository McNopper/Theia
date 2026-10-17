// ApplicationCompute.cpp - the async-compute / GI orchestration of the
// Theia demo application: G-buffer transitions for recording, the async
// compute submission flow and the single-queue GI fallback. Methods of
// theia::Application split out of Application.cpp (TU split, no behavior
// change).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <slang-math/slang-math.hpp>
#include <string>
#include <tuple>

#include "demo/Application.hpp"
#include "harmonia/core/Barrier.hpp"
#include "harmonia/core/Env.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "theia/camera/CameraJitter.hpp"

namespace theia {

GiPass::FrameParams Application::buildGiFrameParams(const sm::float4x4& view,
                                                    const sm::float4x4& curViewProj) const noexcept {
    GiPass::FrameParams gp{};
    gp.scene = m_scene.get();
    m_env.fill(gp);
    gp.view = view;
    gp.cameraPos = m_camera.position;
    gp.exposure = m_camera.physical.exposure();
    gp.prevViewProj = m_prevViewProjValid ? m_prevViewProj : curViewProj;
    gp.frameSampleIndex = frameIndex();
    gp.rngBaseSeed = config().rngSeed;
    gp.maxDepth = m_sceneMaxDepth;
    gp.useA3Regularization = !m_pureEstimatorCapture;
    gp.gradientVarianceView = denoiserGradientImageView();
    gp.adaptiveMaxSamples = 4;
    gp.useRestirDi = m_useRestirDi && !m_useRestirPt;
    gp.useRestirPt = m_useRestirPt;
    gp.useRestirPtPath = m_useRestirPtPath && m_useRestirPt;
    gp.fireflyClampEnabled = !m_pureEstimatorCapture;
    gp.reconnectShiftEnabled = m_reconnectShiftEnabled;
    gp.motionVectorView = VK_NULL_HANDLE;
    return gp;
}

MotionVectorPass::FrameParams Application::buildMotionVectorParams(const sm::float4x4& curViewProj) const noexcept {
    MotionVectorPass::FrameParams mvp{};
    mvp.curViewProj = curViewProj;
    mvp.prevViewProj = m_prevViewProjValid ? m_prevViewProj : curViewProj;
    mvp.invCurViewProj = sm::inverse(curViewProj);
    mvp.prevInstanceTransformBuffer = m_scene->prevInstanceTransformBuffer().handle();
    return mvp;
}

void Application::submitAsyncCompute(VkCommandBuffer cmd,
                                     const harmonia::app::RenderTarget& target,
                                     const sm::float4x4& view,
                                     const sm::float4x4& curViewProj) noexcept {
    const std::uint32_t slot = frameIndex() % 2U;

    GiPass::FrameParams gp = buildGiFrameParams(view, curViewProj);
    m_pendingMvp = buildMotionVectorParams(curViewProj);

    const std::uint32_t gfxFamily = deviceContext().graphicsFamily;
    const std::uint32_t asyncFamily = deviceContext().asyncComputeQueueFamily;

    const std::array gfxRelease{
        harmonia::core::imageBarrier(m_renderer->giBufferImage(),
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     gfxFamily,
                                     asyncFamily),
        harmonia::core::imageBarrier(m_renderer->gbufferImage(),
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     gfxFamily,
                                     asyncFamily),
        harmonia::core::imageBarrier(target.image,
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     gfxFamily,
                                     asyncFamily),
    };
    harmonia::core::pipelineBarrier(cmd, gfxRelease);

    // MOD2: wait for the same slot's PREVIOUS async-compute completion (2 frames ago)
    // via the timeline semaphore. Signal value is monotonic — no reset needed.
    if (m_asyncSignalValue >= 2) { // the first two frames have nothing to wait for
        const std::uint64_t waitValue = m_asyncSignalValue - 2;
        const VkSemaphoreWaitInfo waitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &m_asyncTimelineSemaphore,
            .pValues = &waitValue,
        };
        if (const VkResult r = vkWaitSemaphores(deviceContext().device, &waitInfo, UINT64_MAX); r != VK_SUCCESS)
            harmonia::core::Logger::error("async compute timeline wait failed: {}", static_cast<std::int32_t>(r));
    }
    if (const VkResult r = vkResetCommandBuffer(m_asyncCmdBufs[slot], 0); r != VK_SUCCESS)
        harmonia::core::Logger::error("async compute reset command buffer failed: {}", static_cast<std::int32_t>(r));
    constexpr VkCommandBufferBeginInfo asyncBegin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    if (const VkResult r = vkBeginCommandBuffer(m_asyncCmdBufs[slot], &asyncBegin); r != VK_SUCCESS)
        harmonia::core::Logger::error("async compute begin command buffer failed: {}", static_cast<std::int32_t>(r));
    const VkCommandBuffer asyncCmd = m_asyncCmdBufs[slot];

    const std::array asyncAcquire{
        harmonia::core::imageBarrier(m_renderer->giBufferImage(),
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_READ_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     gfxFamily,
                                     asyncFamily),
        harmonia::core::imageBarrier(m_renderer->gbufferImage(),
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_READ_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     gfxFamily,
                                     asyncFamily),
        harmonia::core::imageBarrier(target.image,
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     gfxFamily,
                                     asyncFamily),
    };
    harmonia::core::pipelineBarrier(asyncCmd, asyncAcquire);

    m_giPass.record(asyncCmd, gp, /*skipPreBarriers=*/true);

    const std::array asyncRelease{
        harmonia::core::imageBarrier(m_renderer->giBufferImage(),
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_READ_BIT,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     asyncFamily,
                                     gfxFamily),
        harmonia::core::imageBarrier(m_renderer->gbufferImage(),
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_READ_BIT,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     asyncFamily,
                                     gfxFamily),
        harmonia::core::imageBarrier(target.image,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_WRITE_BIT,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     VK_ACCESS_2_NONE,
                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                     asyncFamily,
                                     gfxFamily),
    };
    harmonia::core::pipelineBarrier(asyncCmd, asyncRelease);
    if (const VkResult r = vkEndCommandBuffer(asyncCmd); r != VK_SUCCESS)
        harmonia::core::Logger::error("async compute end command buffer failed: {}", static_cast<std::int32_t>(r));

    m_prevViewProj = curViewProj;
    m_prevViewProjValid = true;
}

} // namespace theia
