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
///   - dgcBuf:              kMaxInstances × {1,1,1} DGC records (pre-filled, never modified)
///                          (INDIRECT + DEVICE_ADDRESS)
///
/// Two draw paths (selected by ForwardRenderer based on DeviceContext::dgcSupported):
///
///   DGC path (preferred when VK_EXT_device_generated_commands is available):
///     - vkCmdExecuteGeneratedCommandsEXT with:
///         indirectAddress     = dgcAddress()         (pre-filled DGC records)
///         sequenceCountAddress = indirectDrawAddress() (visibleCount in first 4 bytes)
///     - Task shader uses SV_DrawIndex as compact-list index (DrawIndex = sequence index)
///
///   GD3 fallback (single indirect draw):
///     - vkCmdDrawMeshTasksIndirectEXT(cmd, indirectDrawBuffer(), 0, 1, 12)
///     - Task shader uses gid.x as compact-list index
///
/// In the task shader, compactInstanceList[gid.x + drawIdx] handles both paths:
///   GD3: drawIdx = 0, gid.x = 0..N-1   |   DGC: drawIdx = 0..N-1, gid.x = 0
///
/// GPU-driven design: no CPU readback; the CPU only records the indirect dispatch.
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
    /// GD3 fallback: pass to vkCmdDrawMeshTasksIndirectEXT (drawCount=1, stride=12).
    /// DGC path: first 4 bytes (groupCountX = visibleCount) used as sequenceCountAddress.
    [[nodiscard]] VkBuffer indirectDrawBuffer() const noexcept { return m_indirectDrawBuf.handle(); }

    /// Device address of indirectDrawBuf[0..3] = visible instance count.
    /// Used as VkGeneratedCommandsInfoEXT::sequenceCountAddress in the DGC path.
    [[nodiscard]] VkDeviceAddress indirectDrawAddress() const noexcept {
        return m_indirectDrawBuf.deviceAddress();
    }

    /// DGC records buffer: kMaxInstances × {1,1,1} (VkDrawMeshTasksIndirectCommandEXT).
    /// Pre-filled at init; never modified per frame.
    /// Used as VkGeneratedCommandsInfoEXT::indirectAddress in the DGC path.
    [[nodiscard]] VkDeviceAddress dgcAddress() const noexcept { return m_dgcBuf.deviceAddress(); }

  private:
    const DeviceContext* m_ctx = nullptr;

    VkPipeline            m_pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool           = VK_NULL_HANDLE;
    VkDescriptorSet       m_set            = VK_NULL_HANDLE;

    Buffer m_compactInstanceListBuf; ///< uint[kMaxInstances]  STORAGE
    Buffer m_indirectDrawBuf;        ///< single VkDrawMeshTasksIndirectCommandEXT  STORAGE | INDIRECT | DEVICE_ADDR
    Buffer m_dgcBuf;                 ///< kMaxInstances × {1,1,1}  INDIRECT | DEVICE_ADDR  (pre-filled, read-only)
};

} // namespace theia
