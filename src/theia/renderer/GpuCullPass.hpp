#pragma once

#include <volk/volk.h>

#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"

namespace theia {

/// GPU-driven per-instance frustum cull pass.
/// Runs a compute shader each frame that tests every scene instance's bounding sphere
/// against the view frustum and writes the results to GPU buffers:
///   - compactInstanceList: uint[] of visible instance indices (STORAGE)
///   - indirectDrawBuf:     single VkDrawMeshTasksIndirectCommandEXT = {visibleCount, 1, 1}
///                          (STORAGE + INDIRECT + DEVICE_ADDRESS)
///
/// Two draw paths (selected by ForwardRenderer based on DeviceContext::dgcSupported):
///
///   DGC path (preferred when VK_EXT_device_generated_commands is available):
///     - vkCmdExecuteGeneratedCommandsEXT with a single sequence whose command is
///       indirectDrawBuf = {visibleCount, 1, 1} (indirectAddress = indirectDrawAddress()).
///       One GPU-generated draw dispatches `visibleCount` task workgroups.
///
///   GD3 fallback (single indirect draw):
///     - vkCmdDrawMeshTasksIndirectEXT(cmd, indirectDrawBuffer(), 0, 1, 12)
///
/// Both paths share the same task-shader semantics: gid.x = 0..visibleCount-1 indexes
/// compactInstanceList to recover the actual scene instance index.
///
/// GPU-driven design: no CPU readback; the CPU only records the indirect dispatch, and the
/// draw command / count are GPU-resident (written by the cull compute shader).
///
/// Ref: Gribb & Hartmann — "Fast Extraction of Viewing Frustum Planes from the
///      World-View-Projection Matrix" (2001)
class GpuCullPass {
  public:
    /// Maximum number of instances supported per allocation.
    /// Sized conservatively; covers all current test scenes with room to spare.
    static constexpr uint32_t kMaxInstances = 4096;

    GpuCullPass() = default;
    ~GpuCullPass();

    GpuCullPass(const GpuCullPass&) = delete;
    GpuCullPass& operator=(const GpuCullPass&) = delete;
    GpuCullPass(GpuCullPass&&) noexcept;
    GpuCullPass& operator=(GpuCullPass&&) noexcept;

    /// Initialize the cull pass pipeline and output buffers.
    /// @param spvFilename  SPIR-V filename resolved against THEIA_SHADER_DIR.
    [[nodiscard]] bool initialize(const DeviceContext& ctx,
                                  const char* spvFilename = "forward_cull.comp.spv");

    void shutdown();

    [[nodiscard]] bool isInitialized() const noexcept { return m_ctx != nullptr; }

    /// Dispatch the frustum cull compute pass.
    /// @param instanceBuf      GpuInstance[] storage buffer (scene.instances)
    /// @param instanceBoundsBuf GpuInstanceBounds[] storage buffer (scene.instanceBounds)
    /// @param instanceCount    Number of scene instances to cull
    /// @param viewProj         Current frame's row-major view-projection matrix
    ///
    /// Before recording the draws that consume the results, the caller MUST insert
    /// a pipeline barrier on compactInstanceListBuffer() and indirectDrawBuffer()
    /// from COMPUTE_SHADER_WRITE → (TASK_SHADER_READ, INDIRECT_COMMAND_READ).
    void dispatch(VkCommandBuffer cmd,
                  VkBuffer instanceBuf,
                  VkBuffer instanceBoundsBuf,
                  uint32_t instanceCount,
                  const sm::float4x4& viewProj);

    /// GPU buffer of visible instance indices (uint[kMaxInstances]).
    /// Bind to ForwardRenderer set 0, binding 10 for the task shader to read.
    [[nodiscard]] VkBuffer compactInstanceListBuffer() const noexcept {
        return m_compactInstanceListBuf.handle();
    }

    /// Single VkDrawMeshTasksIndirectCommandEXT entry: {visibleCount, 1, 1}.
    /// GD3: pass to vkCmdDrawMeshTasksIndirectEXT (drawCount=1, stride=12).
    /// GD6 DGC: used as the single sequence's indirect command (indirectAddress).
    [[nodiscard]] VkBuffer indirectDrawBuffer() const noexcept { return m_indirectDrawBuf.handle(); }

    /// Device address of indirectDrawBuf = {visibleCount, 1, 1}.
    /// GD6 DGC path: VkGeneratedCommandsInfoEXT::indirectAddress (single-sequence command).
    [[nodiscard]] VkDeviceAddress indirectDrawAddress() const noexcept {
        return m_indirectDrawBuf.deviceAddress();
    }

  private:
    const DeviceContext* m_ctx = nullptr;

    VkPipeline            m_pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool           = VK_NULL_HANDLE;
    VkDescriptorSet       m_set            = VK_NULL_HANDLE;

    Buffer m_compactInstanceListBuf; ///< uint[kMaxInstances]  STORAGE
    Buffer m_indirectDrawBuf;        ///< single VkDrawMeshTasksIndirectCommandEXT  STORAGE | INDIRECT | DEVICE_ADDR
};

} // namespace theia
