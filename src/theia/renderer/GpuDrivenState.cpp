#include "theia/renderer/GpuDrivenState.hpp"

#include <volk/volk.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "harmonia/core/Logger.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

bool GpuDrivenState::ensureVisibilityBuffers(const DeviceContext& ctx, const Scene& scene) {
    if (visBuiltFor == &scene && meshletVisibility[0].handle() != VK_NULL_HANDLE) {
        return true;
    }
    const std::uint32_t meshletCount = std::max(1u, scene.meshletCount());
    const VkDeviceSize size = static_cast<VkDeviceSize>(meshletCount) * sizeof(std::uint32_t);
    for (auto& buf : meshletVisibility) {
        auto created = Buffer::create(ctx,
                                      size,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      "theia.meshletVisibility");
        if (!created) {
            Logger::error("Failed to create meshlet visibility buffer");
            return false;
        }
        buf = std::move(*created);
    }
    visMeshletCount = meshletCount;
    visBuiltFor = &scene;
    visFrame = 0;
    // Both buffers hold undefined VMA memory; the first frame clears both (see ForwardRenderer::recordFrame).
    visClearPrev = true;
    return true;
}

void GpuDrivenState::dispatchCull(VkCommandBuffer cmd,
                                  const Scene& scene,
                                  std::uint32_t instanceCount,
                                  const sm::float4x4& viewProj) {
    const bool canDraw = (instanceCount > 0) && (vkCmdDrawMeshTasksEXT != nullptr);
    if (!canDraw || !gpuCullPass.isInitialized()) {
        return;
    }
    gpuCullPass.dispatch(
        cmd, scene.instanceBuffer().handle(), scene.instanceBoundsBuffer().handle(), instanceCount, viewProj);
    const std::array cullBarriers{
        VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .buffer = gpuCullPass.compactInstanceListBuffer(),
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        },
        VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT,
            .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT,
            .buffer = gpuCullPass.indirectDrawBuffer(),
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        },
    };
    const VkDependencyInfo cullDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = static_cast<std::uint32_t>(cullBarriers.size()),
        .pBufferMemoryBarriers = cullBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &cullDep);
}

void GpuDrivenState::issueDraw(VkCommandBuffer cmd, VkPipeline pipeline, std::uint32_t instanceCount) {
    if (gpuCullPass.isInitialized() && dgcLayout != VK_NULL_HANDLE) {
        // GD6 DGC path: one GPU-generated DRAW_MESH_TASKS command via VK_EXT_device_generated_commands.
        const VkGeneratedCommandsPipelineInfoEXT dgcPipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,
            .pipeline = pipeline,
        };
        const VkGeneratedCommandsInfoEXT dgcInfo{
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,
            .pNext = &dgcPipelineInfo,
            .shaderStages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .indirectExecutionSet = VK_NULL_HANDLE,
            .indirectCommandsLayout = dgcLayout,
            .indirectAddress = gpuCullPass.indirectDrawAddress(),
            .indirectAddressSize = kMeshTaskIndirectCmdSize,
            .preprocessAddress = dgcPreprocessAddr,
            .preprocessSize = dgcPreprocessSize,
            .maxSequenceCount = 1u,
            .sequenceCountAddress = 0,
            .maxDrawCount = 1,
        };
        vkCmdExecuteGeneratedCommandsEXT(cmd, VK_FALSE, &dgcInfo);
    } else if (gpuCullPass.isInitialized() && vkCmdDrawMeshTasksIndirectEXT != nullptr) {
        // GD3 fallback: single indirect draw; task shader gid.x = visible instance position.
        vkCmdDrawMeshTasksIndirectEXT(cmd, gpuCullPass.indirectDrawBuffer(), 0, 1, kMeshTaskIndirectCmdSize);
    } else {
        vkCmdDrawMeshTasksEXT(cmd, instanceCount, 1, 1);
    }
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

} // namespace theia
