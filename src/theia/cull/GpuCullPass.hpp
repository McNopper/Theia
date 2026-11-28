#ifndef THEIA_RENDERER_GPUCULLPASS_HPP
#define THEIA_RENDERER_GPUCULLPASS_HPP

#include <volk/volk.h>

#include <array>
#include <cstdint>
#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/DescriptorBufferWriter.hpp"
#include "harmonia/core/VulkanHandle.hpp"

namespace theia {

/// GPU-driven per-instance frustum cull pass.
/// Runs a compute shader each frame that tests every scene instance's bounding sphere
/// against the view frustum and writes the results to GPU buffers:
///   - compactInstanceList: uint[] of visible instance indices (STORAGE)
///   - indirectDrawBuf:     single VkDrawMeshTasksIndirectCommandEXT = {visibleCount, 1, 1}
///                          (STORAGE + INDIRECT + DEVICE_ADDRESS)
///
/// Two draw paths (selected by ForwardRenderer based on harmonia::DeviceContext::dgcSupported):
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
    /// Entries the compact instance list starts with. It holds one index per
    /// visible instance and grows on demand (see reserve).
    static constexpr std::uint32_t kInitialCapacity = 4096;

    GpuCullPass() = default;
    ~GpuCullPass();

    GpuCullPass(const GpuCullPass&) = delete;
    GpuCullPass& operator=(const GpuCullPass&) = delete;
    GpuCullPass(GpuCullPass&&) noexcept = default;
    GpuCullPass& operator=(GpuCullPass&&) noexcept = default;

    /// Initialize the cull pass pipeline and output buffers.
    /// The compact list starts with `kInitialCapacity` entries, or with the largest
    /// count an earlier reserve() asked for.
    /// @param spvFilename  SPIR-V filename resolved against THEIA_SHADER_DIR.
    [[nodiscard]] bool initialize(const harmonia::DeviceContext& ctx,
                                  const char* spvFilename = "forward_cull.comp.spv");

    void shutdown();

    [[nodiscard]] bool isInitialized() const noexcept { return m_ctx != nullptr; }

    /// Make room for @p instanceCount visible instances.
    ///
    /// The compact list holds one index per visible instance, so a scene with
    /// more instances than the current capacity must reserve before it is
    /// drawn; the capacity never shrinks. Call between frames with the GPU
    /// idle - the previous list is destroyed here, and the host already
    /// waits for the device when it loads a scene.
    ///
    /// Before initialize() the pass has no list to grow: the request is kept
    /// and initialize() sizes the list for it, so the call succeeds (the
    /// renderer sets its scene before it initializes).
    [[nodiscard]] bool reserve(std::uint32_t instanceCount);

    /// Entries the compact instance list currently holds.
    [[nodiscard]] std::uint32_t capacity() const noexcept { return m_capacity; }

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
                  const harmonia::core::Buffer& instanceBuf,
                  const harmonia::core::Buffer& instanceBoundsBuf,
                  std::uint32_t instanceCount,
                  const sm::float4x4& viewProj);

    /// GPU buffer of visible instance indices (uint[capacity()]).
    /// Bind to ForwardRenderer set 0, binding 10 for the task shader to read.
    [[nodiscard]] const harmonia::core::Buffer& compactInstanceListBuffer() const noexcept {
        return m_compactInstanceListBuf;
    }

    /// Single VkDrawMeshTasksIndirectCommandEXT entry: {visibleCount, 1, 1}.
    /// GD3: pass to vkCmdDrawMeshTasksIndirectEXT (drawCount=1, stride=12).
    /// GD6 DGC: used as the single sequence's indirect command (indirectAddress).
    [[nodiscard]] VkBuffer indirectDrawBuffer() const noexcept { return m_indirectDrawBuf.handle(); }

    /// Device address of indirectDrawBuf = {visibleCount, 1, 1}.
    /// GD6 DGC path: VkGeneratedCommandsInfoEXT::indirectAddress (single-sequence command).
    [[nodiscard]] VkDeviceAddress indirectDrawAddress() const noexcept { return m_indirectDrawBuf.deviceAddress(); }

  private:
    const harmonia::DeviceContext* m_ctx = nullptr;

    harmonia::core::UniquePipeline m_pipeline;
    harmonia::core::UniquePipelineLayout m_pipelineLayout;
    harmonia::core::UniqueDescriptorSetLayout m_setLayout;
    /// MOD1: descriptor buffer writer replaces the pool + allocated set.
    harmonia::core::DescriptorBufferWriter m_descWriter;

    harmonia::core::Buffer m_compactInstanceListBuf; ///< uint[m_capacity]  STORAGE
    harmonia::core::Buffer
        m_indirectDrawBuf;           ///< single VkDrawMeshTasksIndirectCommandEXT  STORAGE | INDIRECT | DEVICE_ADDR
    std::uint32_t m_capacity = 0;    ///< entries m_compactInstanceListBuf holds
    std::uint32_t m_reserved = 0;    ///< largest reserve() before initialize(); its initial capacity
    bool m_reportedOverflow = false; ///< the dispatch clamp is diagnosed once
};

} // namespace theia
#endif // THEIA_RENDERER_GPUCULLPASS_HPP
