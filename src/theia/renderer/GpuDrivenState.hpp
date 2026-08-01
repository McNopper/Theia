#ifndef THEIA_RENDERER_GPUDRIVENSTATE_HPP
#define THEIA_RENDERER_GPUDRIVENSTATE_HPP

#include <volk/volk.h>

#include <cstdint>
#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "theia/renderer/GpuCullPass.hpp"
#include "theia/renderer/HiZPass.hpp"
#include "theia/renderer/RendererConstants.hpp"
#include "theia/scene/Scene.hpp"

namespace theia {

/// GPU-driven rendering state extracted from ForwardRenderer (R8/CH9): the frustum-cull
/// pass (GD2/GD3/GD6), the DGC layout + preprocess buffer, the two-pass Hi-Z occlusion
/// pyramid, and the ping-pong meshlet-visibility buffers, plus the visibility/cull/draw
/// methods that operate purely on this state.
///
/// The graphics-pipeline-bound draw recording (pipeline/descriptor-set/push-constant bind)
/// and the frame-level two-pass Hi-Z orchestration stay in ForwardRenderer, which calls
/// issueDraw() once bound.
class GpuDrivenState {
  public:
    // GPU-driven frustum cull pass (GD2/GD3/GD6). Writes compactInstanceList + a single
    // indirect draw command; ForwardRenderer consumes it via vkCmdDrawMeshTasksIndirectEXT
    // (GD3) or vkCmdExecuteGeneratedCommandsEXT (GD6 DGC path).
    GpuCullPass gpuCullPass;
    harmonia::UniqueIndirectCommandsLayout dgcLayout; ///< DRAW_MESH_TASKS_EXT token, stride=12
    VkBuffer dgcPreprocessBuf = VK_NULL_HANDLE;       ///< vkCmdExecuteGeneratedCommandsEXT scratch
    VmaAllocation dgcPreprocessAlloc = VK_NULL_HANDLE;
    VkDeviceAddress dgcPreprocessAddr = 0;
    VkDeviceSize dgcPreprocessSize = 0;

    HiZPass hiZPass;                       ///< current-frame depth pyramid builder (B4)
    harmonia::Buffer meshletVisibility[2]; ///< ping-pong per-meshlet visibility (uint/meshlet)
    std::uint32_t visFrame = 0;            ///< index of PREVIOUS-frame visibility buffer
    std::uint32_t visMeshletCount = 0;     ///< meshlet count the visibility buffers were sized for
    const Scene* visBuiltFor = nullptr;    ///< scene the visibility buffers were built for
    bool hiZTestEnabled = true;            ///< set false for one frame on a camera cut
    bool visClearPrev = false;             ///< clear PREV visibility next frame (freshly (re)built)
    bool hiZDebugDisabled = false;         ///< THEIA_DISABLE_HIZ: draw all meshlets (A-B debug)
    bool forceSinglePass = false;          ///< THEIA_SINGLE_PASS: bypass two-pass Hi-Z (A-B debug)

    /// (Re)create the ping-pong meshlet-visibility buffers when the scene changes.
    [[nodiscard]] bool ensureVisibilityBuffers(const harmonia::DeviceContext& ctx, const Scene& scene);
    /// Run the GPU frustum cull + the post-cull barrier (compactInstanceList + indirectDrawBuf).
    void
    dispatchCull(VkCommandBuffer cmd, const Scene& scene, std::uint32_t instanceCount, const sm::float4x4& viewProj);
    /// Issue the mesh-task draw: DGC (GD6) when available, else indirect (GD3).
    /// The caller must have already bound the pipeline, descriptor sets, and push constants.
    void issueDraw(VkCommandBuffer cmd, VkPipeline pipeline);
};

} // namespace theia

#endif // THEIA_RENDERER_GPUDRIVENSTATE_HPP
