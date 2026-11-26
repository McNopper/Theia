// Module tests: the GPU-driven frustum cull pass. The compact instance list holds
// one index per visible instance; a scene with more instances than the list's
// initial capacity must be reserved before it is drawn, and a dispatch that was
// not must never write past the list.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <slang-math/slang-math.hpp>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "theia/cull/GpuCullPass.hpp"
#include "theia/scene/Scene.hpp"

namespace {

// A 68x68 grid of small boxes: 4624 instances, 528 more than the cull pass's
// initial capacity, every one inside the view below.
constexpr std::uint32_t kGrid = 68;
constexpr std::uint32_t kInstanceCount = kGrid * kGrid;

/// The scene under test: one box mesh instanced on the grid.
theia::Scene makeGridScene(const harmonia::DeviceContext& ctx, const harmonia::core::CommandPool& pool) {
    theia::Scene scene;
    const std::uint32_t material = scene.addMaterial(harmonia::scene::Material::diffuse(sm::float3(0.7F), 1.0F));
    harmonia::scene::MeshData box = harmonia::scene::ProceduralGeometry::makeBox(sm::float3(0.4F));
    const std::uint32_t mesh = scene.addMesh(ctx, pool, std::move(box), harmonia::scene::MeshOpacity{}, "test.grid");
    EXPECT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
    for (std::uint32_t i = 0; i < kInstanceCount; ++i) {
        const auto x = static_cast<float>(static_cast<int>(i % kGrid) - static_cast<int>(kGrid) / 2);
        const auto y = static_cast<float>(static_cast<int>(i / kGrid) - static_cast<int>(kGrid) / 2);
        const std::uint32_t placed =
            scene.addInstance(mesh, harmonia::scene::Transform{.translation = {2.0F * x, 2.0F * y, 0.0F}}, material);
        EXPECT_NE(placed, std::numeric_limits<std::uint32_t>::max()) << "instance " << i;
        if (placed == std::numeric_limits<std::uint32_t>::max()) {
            break;
        }
    }
    return scene;
}

/// View-projection exactly as ForwardRenderer builds it: 220 m in front of the
/// grid, looking at it, a 60 degree vertical field of view (half-extent 127 m
/// at the grid's distance - the grid spans 134 m, so every instance is inside).
sm::float4x4 gridViewProj() {
    sm::float4x4 proj = sm::perspective(sm::radians(60.0F), 16.0F / 9.0F, 0.1F, 1000.0F);
    proj[1][1] *= -1.0F;
    return proj *
           sm::lookAt(sm::float3(0.0F, 0.0F, 220.0F), sm::float3(0.0F, 0.0F, 0.0F), sm::float3(0.0F, 1.0F, 0.0F));
}

void cullScene(theia::GpuCullPass& pass, const theia::Scene& scene, const harmonia::core::CommandPool& pool) {
    auto cmd = pool.beginOneShot();
    ASSERT_TRUE(cmd.has_value());
    ASSERT_NE(*cmd, VK_NULL_HANDLE);
    pass.dispatch(*cmd, scene.instanceBuffer(), scene.instanceBoundsBuffer(), scene.instanceCount(), gridViewProj());
    ASSERT_EQ(pool.endOneShot(*cmd), VK_SUCCESS);
}

/// Copy a GPU buffer's bytes to the host (the one-shot submit waits for the GPU).
std::vector<std::uint8_t> readBack(VkBuffer source,
                                   VkDeviceSize size,
                                   const harmonia::DeviceContext& ctx,
                                   const harmonia::core::CommandPool& pool) {
    auto staging = harmonia::core::Buffer::create(
        ctx, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU, "test.gpuCull.readback");
    EXPECT_TRUE(staging.has_value());
    if (!staging) {
        return {};
    }
    auto cmd = pool.beginOneShot();
    EXPECT_TRUE(cmd.has_value());
    if (!cmd) {
        return {};
    }
    VkMemoryBarrier2 shaderToCopy{};
    shaderToCopy.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    shaderToCopy.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    shaderToCopy.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    shaderToCopy.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    shaderToCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkMemoryBarrier2 copyToHost{};
    copyToHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    copyToHost.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    copyToHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    copyToHost.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    copyToHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    const std::array<VkMemoryBarrier2, 2> barriers{shaderToCopy, copyToHost};
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
    dependency.pMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(*cmd, &dependency);
    const VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(*cmd, source, staging->handle(), 1, &region);
    EXPECT_EQ(pool.endOneShot(*cmd), VK_SUCCESS);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    EXPECT_EQ(staging->readData(bytes.data(), size), VK_SUCCESS);
    return bytes;
}

/// The indirect draw command {visibleCount, 1, 1} after a dispatch.
std::uint32_t visibleCount(const theia::GpuCullPass& pass,
                           const harmonia::DeviceContext& ctx,
                           const harmonia::core::CommandPool& pool) {
    const auto bytes = readBack(pass.indirectDrawBuffer(), 12, ctx, pool);
    EXPECT_EQ(bytes.size(), 12U);
    std::uint32_t command[3]{};
    if (bytes.size() == 12U) {
        std::memcpy(command, bytes.data(), sizeof(command));
        EXPECT_EQ(command[1], 1U);
        EXPECT_EQ(command[2], 1U);
    }
    return command[0];
}

/// Assert that the first @p count entries of the compact list hold every
/// instance index 0..count-1 exactly once.
void expectEveryInstanceOnce(const theia::GpuCullPass& pass,
                             std::uint32_t count,
                             const harmonia::DeviceContext& ctx,
                             const harmonia::core::CommandPool& pool) {
    const VkDeviceSize size = static_cast<VkDeviceSize>(count) * sizeof(std::uint32_t);
    const auto bytes = readBack(pass.compactInstanceListBuffer().handle(), size, ctx, pool);
    ASSERT_EQ(bytes.size(), static_cast<std::size_t>(size));
    std::vector<std::uint32_t> indices(count);
    std::memcpy(indices.data(), bytes.data(), static_cast<std::size_t>(size));
    std::sort(indices.begin(), indices.end());
    for (std::uint32_t i = 0; i < count; ++i) {
        ASSERT_EQ(indices[i], i) << "entry " << i;
    }
}

} // namespace

TEST_F(RtFixture, GpuCull_CompactListGrowsForAScenePastTheInitialCapacity) {
    theia::Scene scene = makeGridScene(deviceCtx(), commandPool());
    ASSERT_EQ(scene.instanceCount(), kInstanceCount);
    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);

    theia::GpuCullPass pass;
    ASSERT_TRUE(pass.initialize(deviceCtx()));
    EXPECT_EQ(pass.capacity(), theia::GpuCullPass::kInitialCapacity);

    // reserve() grows the list for the scene and never shrinks it.
    ASSERT_TRUE(pass.reserve(scene.instanceCount()));
    EXPECT_EQ(pass.capacity(), kInstanceCount);
    EXPECT_TRUE(pass.reserve(1U));
    EXPECT_EQ(pass.capacity(), kInstanceCount);

    cullScene(pass, scene, commandPool());
    EXPECT_EQ(visibleCount(pass, deviceCtx(), commandPool()), kInstanceCount);
    expectEveryInstanceOnce(pass, kInstanceCount, deviceCtx(), commandPool());
}

TEST_F(RtFixture, GpuCull_NeverWritesPastTheCompactList) {
    // A dispatch with more instances than the list holds - reserve() not
    // called - culls only what the list can hold, instead of writing past it.
    theia::Scene scene = makeGridScene(deviceCtx(), commandPool());
    ASSERT_EQ(scene.instanceCount(), kInstanceCount);
    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);

    theia::GpuCullPass pass;
    ASSERT_TRUE(pass.initialize(deviceCtx()));
    ASSERT_EQ(pass.capacity(), theia::GpuCullPass::kInitialCapacity);
    ASSERT_GT(kInstanceCount, pass.capacity());

    cullScene(pass, scene, commandPool());
    EXPECT_EQ(visibleCount(pass, deviceCtx(), commandPool()), theia::GpuCullPass::kInitialCapacity);
    expectEveryInstanceOnce(pass, theia::GpuCullPass::kInitialCapacity, deviceCtx(), commandPool());
}
