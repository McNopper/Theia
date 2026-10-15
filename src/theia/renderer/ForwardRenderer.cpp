#include "theia/renderer/ForwardRenderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <harmonia/core/Barrier.hpp>
#include <harmonia/core/Env.hpp>
#include <harmonia/core/Logger.hpp>
#include <harmonia/core/ShaderModule.hpp>
#include <slang-math/slang-math.hpp>
#include <theia/camera/CameraJitter.hpp>
#include <theia/renderer/ShaderPath.hpp>
#include <theia/scene/Scene.hpp>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

namespace theia {

ForwardRenderer::~ForwardRenderer() {
    shutdown();
}

void ForwardRenderer::shutdown() {
    if (!m_ctx) {
        return;
    }

    vkDeviceWaitIdle(m_ctx->device);

    m_graphicsPipeline.reset();
    m_graphicsPipelineTransparent.reset();
    m_skyPipeline.reset();
    m_skyPipelineLayout.reset();
    m_pipelineLayout.reset();
    m_desc.meshSetLayout.reset();
    m_desc.matSetLayout.reset();
    m_desc.iblSetLayout.reset();
    m_desc.textureSetLayout.reset();
    m_desc.mesh.shutdown();
    m_desc.mat.shutdown();
    m_desc.ibl.shutdown();
    m_desc.texture.shutdown();

    m_envSamplerInfo = {};
    m_envRawInfo = {};
    m_envSampler.reset();
    // VK14: no dummy resources to destroy.
    m_envMarginalCdf = nullptr;
    m_envConditionalCdf = nullptr;
    m_envImportanceWidth = 0;
    m_envImportanceHeight = 0;

    // VK14: no tile dummies.

    if (m_gpu.dgcPreprocessBuf != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_ctx->allocator, m_gpu.dgcPreprocessBuf, m_gpu.dgcPreprocessAlloc);
        m_gpu.dgcPreprocessBuf = VK_NULL_HANDLE;
        m_gpu.dgcPreprocessAlloc = VK_NULL_HANDLE;
        m_gpu.dgcPreprocessAddr = 0;
        m_gpu.dgcPreprocessSize = 0;
    }

    m_gpu.dgcLayout.reset();

    m_desc.texturesBoundFor = nullptr;
    m_desc.texturesBoundFor = nullptr;
    m_gpu.hiZPass.shutdown();
    m_gpu.gpuCullPass.shutdown();
    m_gpu.meshletVisibility[0] = {};
    m_gpu.meshletVisibility[1] = {};
    m_gpu.visBuiltFor = nullptr;
    m_gpu.visMeshletCount = 0;
    m_gpu.visFrame = 0;
    m_depthTarget = {};
    m_gbufferTarget = {};
    m_giBufferTarget = {};
    m_ctx = nullptr;

    m_initialized = false;
}

void ForwardRenderer::setTileBuffers(const harmonia::core::Buffer& tileLightCounts,
                                     const harmonia::core::Buffer& tileLightIndices,
                                     std::uint32_t tilesX,
                                     std::uint32_t tilesY) {
    m_tileLightCountsBuf = &tileLightCounts;
    m_tileLightIndicesBuf = &tileLightIndices;
    m_tilesX = tilesX;
    m_tilesY = tilesY;
}

void ForwardRenderer::setEnvironment(VkImageView rawEnvView, float envUnitNits) {
    if (!m_ctx || !m_desc.ibl.layoutSize()) {
        return;
    }

    // Raw env panorama for the sky background / transparent env path. Fall back to the
    // 1×1 placeholder when no env is supplied, so the descriptor stays valid (sky/GI gate
    // reads on hasEnv).
    // VK14 nullDescriptor: bind VK_NULL_HANDLE when no env (shader gates on hasEnv).
    const VkImageView envView = rawEnvView;
    m_envUnitNits = envUnitNits;

    // MOD1: write IBL descriptors into the descriptor buffer.
    m_desc.ibl.writeSampler(*m_ctx, 3, m_envSampler);
    m_desc.ibl.writeSampledImage(*m_ctx, 4, envView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void ForwardRenderer::drawOpaque(VkCommandBuffer cmd,
                                 const MeshPushConstants& pcBase,
                                 std::uint32_t cullPhase,
                                 std::uint32_t hiZMipCount) {
    const std::uint32_t instanceCount = m_scene->instanceCount();
    if (instanceCount == 0 || vkCmdDrawMeshTasksEXT == nullptr) {
        return;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    // MOD1: bind descriptor buffers (replaces vkCmdBindDescriptorSets).
    {
        const std::array<harmonia::core::DescriptorBufferWriter*, 4> writers{
            &m_desc.mesh, &m_desc.mat, &m_desc.ibl, &m_desc.texture};
        harmonia::core::DescriptorBufferWriter::bindSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0,
                                                    writers);
    }
    auto pc = pcBase;
    pc.presentationParams.y = 0.0f; // opaque routing in the task shader
    pc.cullPhase = cullPhase;
    pc.hiZMipCount = hiZMipCount;
    vkCmdPushConstants(cmd,
                       m_pipelineLayout,
                       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(MeshPushConstants),
                       &pc);
    m_gpu.issueDraw(cmd, m_graphicsPipeline);
}

VkShaderModule ForwardRenderer::loadShaderModule(const char* filename) {
    auto module = harmonia::core::createShaderModule(m_ctx->device, shaderPath(filename));
    if (!module) {
        harmonia::core::Logger::error("Failed to load shader: {}", filename);
        return VK_NULL_HANDLE;
    }
    return *module;
}

void ForwardRenderer::recordFrame(VkCommandBuffer cmd) {
    if (!m_scene || m_config.hdrImage == VK_NULL_HANDLE) {
        return;
    }

    // Ensure the two-pass Hi-Z visibility buffers exist for the current scene.
    const bool visReady = m_scene != nullptr && m_gpu.ensureVisibilityBuffers(*m_ctx, *m_scene);

    // VK14: no dummy env — nullDescriptor handles the absent-env case.

    prepareAttachments(cmd);

    updateSceneDescriptors(cmd);

    // harmonia::renderer::Camera matrices + push constants (needed before rendering so the two Hi-Z passes and
    // the Hi-Z occlusion test share the same projection).
    sm::float4x4 proj = sm::perspective(sm::radians(m_camera.vfovDeg),
                                        static_cast<float>(m_config.width) / static_cast<float>(m_config.height),
                                        m_camera.nearPlane,
                                        m_camera.farPlane);
    proj[1][1] *= -1.0f;
    if (m_render.cameraJitterEnabled) {
        proj = applyProjectionJitter(proj, cameraJitterNdc(m_render.frameSampleIndex, m_config.width, m_config.height));
    }
    const sm::float4x4 view = sm::lookAt(m_camera.position, m_camera.target, m_camera.up);
    const sm::float4x4 viewProj = proj * view;
    // exposure = 1 / (1.2 * 2^EV100) — matches Hyperion's PhysicalCamera calc
    const float exposure = m_camera.physical.exposure();
    // NDC projection scales for the mesh-shader Hi-Z bounding-sphere footprint estimate.
    const float projScaleX = std::abs(proj[0][0]);
    const float projScaleY = std::abs(proj[1][1]);

    const ForwardRenderer::MeshPushConstants pcBase{
        .viewProj = viewProj,
        .view = view,
        .cameraPos = sm::float4(m_camera.position, 1.0f),
        .exposure = exposure,
        .lightCount = m_scene ? m_scene->lightCount() : 0u,
        .emissiveTriangleCount = m_scene ? m_scene->emissiveTriangleCount() : 0u,
        .tilesX = m_tilesX,
        .tilesY = m_tilesY,
        .screenWidth = m_config.width,
        .screenHeight = m_config.height,
        .transparentMaxDepth = m_render.transparentMaxDepth,
        .frameSampleIndex = m_render.frameSampleIndex,
        .rngBaseSeed = m_render.rngBaseSeed,
        .rngFlags = (m_render.deterministicReplay ? 0x1u : 0u) | (m_render.rngDebug != 0u ? 0x2u : 0u) |
                    (m_render.fireflyClampEnabled ? 0x4u : 0u),
        .cullPhase = 0u,
        .envImportanceWidth = m_envImportanceWidth,
        .envImportanceHeight = m_envImportanceHeight,
        .hiZMipCount = 0u,
        .giEnabled = m_render.giEnabled,
        // Ray-traced sun shadow: direction toward the dominant IBL light + strength.
        // shadowParams: x = ray tMin (scene-scale bias from camera near plane), y = sky ambient
        // floor, z = env_unit_nits, w = |proj[0][0]| (Hi-Z footprint x scale).
        .sunDirection = sm::float4(m_sunDir, m_hasEnv ? m_sunStrength : 0.0f),
        .shadowParams = sm::float4(std::max(m_camera.nearPlane, 1e-4f), 0.35f, m_envUnitNits, projScaleX),
        // presentationParams.w = |proj[1][1]| (Hi-Z footprint y scale).
        .presentationParams = sm::float4(m_indirectAmbientStrength, 0.0f, m_render.debugRayHitMode, projScaleY),
    };

    // Set dynamic state (shared by both passes).
    VkViewport vp{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(m_config.width),
        .height = static_cast<float>(m_config.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{{0, 0}, {m_config.width, m_config.height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    const std::uint32_t instanceCount = m_scene->instanceCount();
    const bool canDraw = (instanceCount > 0) && (vkCmdDrawMeshTasksEXT != nullptr);

    // GPU-driven frustum cull (must run outside dynamic rendering — compute cannot run
    // inside a render pass). Dispatch before the Hi-Z passes and barrier results.
    m_gpu.dispatchCull(cmd, *m_scene, instanceCount, viewProj);

    const bool twoPass = canDraw && visReady && !m_gpu.forceSinglePass;
    const std::uint32_t hiZMip =
        (twoPass && m_gpu.hiZPass.isInitialized() && m_gpu.hiZTestEnabled && !m_gpu.hiZDebugDisabled)
            ? m_gpu.hiZPass.mipLevels()
            : 0u;

    if (!twoPass) {
        recordOpaquePass(cmd, pcBase);
        return;
    }

    // ---- Two-pass Hi-Z occlusion culling ----
    // Ensure the Hi-Z image is sampleable in both passes (first-use layout transition).
    m_gpu.hiZPass.prepareForSampling(cmd);

    const VkBuffer prevVisBuf = m_gpu.meshletVisibility[m_gpu.visFrame].handle();
    const VkBuffer currVisBuf = m_gpu.meshletVisibility[m_gpu.visFrame ^ 1u].handle();

    // Reset the current-frame visibility (and previous on the first frame for a scene).
    vkCmdFillBuffer(cmd, currVisBuf, 0, VK_WHOLE_SIZE, 0u);
    if (m_gpu.visClearPrev) {
        vkCmdFillBuffer(cmd, prevVisBuf, 0, VK_WHOLE_SIZE, 0u);
        m_gpu.visClearPrev = false;
    }
    const std::array visFillBarriers{
        VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .buffer = currVisBuf,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        },
        VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .buffer = prevVisBuf,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        },
    };
    const VkDependencyInfo visFillDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = static_cast<std::uint32_t>(visFillBarriers.size()),
        .pBufferMemoryBarriers = visFillBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &visFillDep);

    // Pass 1: redraw only meshlets visible last frame → current-frame initial depth.
    beginSceneRendering(cmd, VK_ATTACHMENT_LOAD_OP_CLEAR);
    recordSky(cmd);
    drawOpaque(cmd, pcBase, /*cullPhase*/ 1u, /*hiZMipCount*/ 0u);
    vkCmdEndRendering(cmd);

    // Build the current-frame Hi-Z pyramid from pass-1 depth (skipped when the test is off).
    if (hiZMip > 0u) {
        const VkImageMemoryBarrier2 depthToRead{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_depthTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
        };
        const VkDependencyInfo depthReadDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &depthToRead,
        };
        vkCmdPipelineBarrier2(cmd, &depthReadDep);

        m_gpu.hiZPass.build(cmd, m_depthTarget.view());

        // Restore depth to a write attachment for pass 2 (contents preserved for LOAD).
        const VkImageMemoryBarrier2 depthToAttach{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_depthTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
        };
        const VkDependencyInfo depthAttachDep{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &depthToAttach,
        };
        vkCmdPipelineBarrier2(cmd, &depthAttachDep);
    }

    // Make pass-1 visibility writes available to pass-2 visibility writes (WAW safety).
    const VkBufferMemoryBarrier2 visRWBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .buffer = currVisBuf,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    const VkDependencyInfo visRWDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &visRWBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &visRWDep);

    // Pass 2: remaining meshlets, occlusion-tested against the current-frame Hi-Z, then the
    // transparent pass — composited over pass-1 targets (LOAD, no clear).
    beginSceneRendering(cmd, VK_ATTACHMENT_LOAD_OP_LOAD);
    drawOpaque(cmd, pcBase, /*cullPhase*/ 2u, hiZMip);
    recordTransparent(cmd, pcBase);
    vkCmdEndRendering(cmd);

    // Swap ping-pong: this frame's visibility becomes next frame's "previous".
    m_gpu.visFrame ^= 1u;
    m_hdrFirstUse = false;
}

void ForwardRenderer::recordSky(VkCommandBuffer cmd) {
    if (!m_hasEnv || m_skyPipeline == VK_NULL_HANDLE) {
        return;
    }
    sm::float4x4 skyProj = sm::perspective(sm::radians(m_camera.vfovDeg),
                                           static_cast<float>(m_config.width) / static_cast<float>(m_config.height),
                                           m_camera.nearPlane,
                                           m_camera.farPlane);
    skyProj[1][1] *= -1.0f;
    if (m_render.cameraJitterEnabled) {
        skyProj =
            applyProjectionJitter(skyProj, cameraJitterNdc(m_render.frameSampleIndex, m_config.width, m_config.height));
    }
    const sm::float4x4 skyView = sm::lookAt(m_camera.position, m_camera.target, m_camera.up);
    const sm::float4x4 skyViewProj = skyProj * skyView;
    const SkyPushConstants skyPc{
        .invViewProj = sm::inverse(skyViewProj),
        .cameraPos = sm::float4(m_camera.position, 1.0f),
        .exposure = m_camera.physical.exposure(),
        .hasEnv = 1u,
        .envScale = m_envUnitNits,
        ._pad1 = 0u,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyPipeline);
    m_desc.ibl.bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyPipelineLayout, 0);
    vkCmdPushConstants(
        cmd, m_skyPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(skyPc), &skyPc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void ForwardRenderer::recordTransparent(VkCommandBuffer cmd, const MeshPushConstants& pcBase) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineTransparent);
    bindMeshSets(cmd);
    auto pc = pcBase;
    pc.presentationParams.y = 1.0f;
    pc.cullPhase = 0u;
    pc.hiZMipCount = 0u;
    vkCmdPushConstants(cmd,
                       m_pipelineLayout,
                       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(MeshPushConstants),
                       &pc);
    m_gpu.issueDraw(cmd, m_graphicsPipelineTransparent);
}

void ForwardRenderer::recordOpaquePass(VkCommandBuffer cmd, const MeshPushConstants& pcBase) {
    const std::uint32_t instanceCount = m_scene->instanceCount();
    const bool canDraw = (instanceCount > 0) && (vkCmdDrawMeshTasksEXT != nullptr);
    if (instanceCount > 0 && vkCmdDrawMeshTasksEXT == nullptr) {
        harmonia::core::Logger::error("vkCmdDrawMeshTasksEXT is NULL - mesh shader extension not loaded!");
    }
    m_gpu.hiZPass.prepareForSampling(cmd);
    beginSceneRendering(cmd, VK_ATTACHMENT_LOAD_OP_CLEAR);
    recordSky(cmd);
    if (canDraw) {
        drawOpaque(cmd, pcBase, /*cullPhase*/ 0u, /*hiZMipCount*/ 0u);
        recordTransparent(cmd, pcBase);
    }
    vkCmdEndRendering(cmd);
    m_hdrFirstUse = false;
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
