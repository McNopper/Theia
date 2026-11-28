// Module tests: the GPU-driven frustum cull pass. The compact instance list holds
// one index per visible instance; a scene with more instances than the list's
// initial capacity must be reserved before it is drawn, and a dispatch that was
// not must never write past the list.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <slang-math/slang-math.hpp>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "harmonia/scene/SceneLoader.hpp"
#include "theia/cull/GpuCullPass.hpp"
#include "theia/scene/Scene.hpp"

namespace {

/// The side of the smallest square grid that holds more instances than @p capacity.
consteval std::uint32_t gridSide(std::uint32_t capacity) {
    std::uint32_t side = 1U;
    while (side * side <= capacity) {
        ++side;
    }
    return side;
}

// A square grid of small boxes with more instances than the compact list's initial
// capacity (the grid follows the capacity, not a number of its own), every one inside
// the view below.
constexpr std::uint32_t kGrid = gridSide(theia::GpuCullPass::kInitialCapacity);
constexpr std::uint32_t kInstanceCount = kGrid * kGrid;
static_assert(kInstanceCount > theia::GpuCullPass::kInitialCapacity);

constexpr float kFieldOfViewY = 60.0F;
constexpr float kAspect = 16.0F / 9.0F;
// The grid spans two metres per instance. Seen from twice its side in front of it, it
// fills half of the vertical field of view.
const float kCameraDistance = 2.0F * float(kGrid) / std::tan(sm::radians(kFieldOfViewY) * 0.5F);

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

/// View-projection as ForwardRenderer builds it: in front of the grid at
/// kCameraDistance, looking at it (see there for the distance).
sm::float4x4 gridViewProj() {
    sm::float4x4 proj = sm::perspective(sm::radians(kFieldOfViewY), kAspect, 0.1F, 4.0F * kCameraDistance);
    proj[1][1] *= -1.0F;
    return proj * sm::lookAt(sm::float3(0.0F, 0.0F, kCameraDistance),
                             sm::float3(0.0F, 0.0F, 0.0F),
                             sm::float3(0.0F, 1.0F, 0.0F));
}

/// A cube mesh, wound outward, and an animated scene that moves it: into the view at
/// one second and back out at two. Written next to each other, like a scene of the
/// assets; the camera is the one of gridViewProj().
std::filesystem::path writeAnimatedMover() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "theia_gpu_cull_anim_test";
    fs::create_directories(dir);
    {
        std::ofstream obj(dir / "mover.obj");
        obj << "v -0.4 -0.4 -0.4\nv 0.4 -0.4 -0.4\nv 0.4 0.4 -0.4\nv -0.4 0.4 -0.4\n"
               "v -0.4 -0.4 0.4\nv 0.4 -0.4 0.4\nv 0.4 0.4 0.4\nv -0.4 0.4 0.4\n"
               "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
               "vn 0 0 -1\nvn 0 0 1\nvn 0 -1 0\nvn 0 1 0\nvn -1 0 0\nvn 1 0 0\n"
               "f 1/1/1 4/2/1 3/3/1\nf 1/1/1 3/3/1 2/4/1\n"
               "f 5/1/2 6/2/2 7/3/2\nf 5/1/2 7/3/2 8/4/2\n"
               "f 1/1/3 2/2/3 6/3/3\nf 1/1/3 6/3/3 5/4/3\n"
               "f 4/1/4 8/2/4 7/3/4\nf 4/1/4 7/3/4 3/4/4\n"
               "f 1/1/5 5/2/5 8/3/5\nf 1/1/5 8/3/5 4/4/5\n"
               "f 2/1/6 3/2/6 7/3/6\nf 2/1/6 7/3/6 6/4/6\n";
    }
    // Twice the camera's distance: far outside the view from wherever the camera is.
    const float farX = 2.0F * kCameraDistance;
    {
        std::ofstream scn(dir / "mover.scene.toml");
        scn << "[animation_clock]\nfps = 30\n\n[camera]\ntranslate = [0.0, 0.0, " << kCameraDistance
            << "]\nrotate_x = 0.0\nrotate_y = 0.0\n\n"
               "[[mesh]]\nname = \"mover\"\npath = \"mover.obj\"\n\n"
               "[[node]]\nid = \"mover\"\nmesh = \"mover\"\n\n"
               "[[animation]]\ntarget = \"nodes/mover/translation\"\ninterpolation = \"linear\"\n"
               "times = [0.0, 1.0, 2.0]\nvalues = ["
            << farX << ", 0.0, 0.0, 0.0, 0.0, 0.0, " << farX << ", 0.0, 0.0]\n";
    }
    return dir;
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

TEST_F(RtFixture, GpuCull_FollowsAnInstanceMovedByAnimation) {
    // The frustum cull tests an instance's world-space bounds. Animation moves
    // instances through the transform sync; the bounds have to move with them,
    // or the instance is culled wherever it actually is and never rasterized.
    theia::Scene scene;
    const std::uint32_t material = scene.addMaterial(harmonia::scene::Material::diffuse(sm::float3(0.7F), 1.0F));
    harmonia::scene::MeshData box = harmonia::scene::ProceduralGeometry::makeBox(sm::float3(0.4F));
    const std::uint32_t mesh =
        scene.addMesh(deviceCtx(), commandPool(), std::move(box), harmonia::scene::MeshOpacity{}, "test.mover");
    ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
    // Outside the view of gridViewProj() (its half-width there is a little more than the
    // camera's distance).
    const std::uint32_t mover = scene.addInstance(
        mesh, harmonia::scene::Transform{.translation = {2.0F * kCameraDistance, 0.0F, 0.0F}}, material);
    ASSERT_NE(mover, std::numeric_limits<std::uint32_t>::max());
    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);

    theia::GpuCullPass pass;
    ASSERT_TRUE(pass.initialize(deviceCtx()));
    cullScene(pass, scene, commandPool());
    EXPECT_EQ(visibleCount(pass, deviceCtx(), commandPool()), 0U) << "built outside the view";

    // Animation stamps the new pose on the instance - the same seam
    // SceneBase::setAnimationTime uses - and the scene syncs it.
    scene.instanceTransform(mover).translation = {0.0F, 0.0F, 0.0F};
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);

    cullScene(pass, scene, commandPool());
    EXPECT_EQ(visibleCount(pass, deviceCtx(), commandPool()), 1U) << "moved into the view";
    expectEveryInstanceOnce(pass, 1U, deviceCtx(), commandPool());
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

// ForwardRenderer::setScene reserves before the renderer is initialized. The supported
// order is a success, not an error: nothing is grown (there is no list yet), and the
// pass keeps the request for initialize(). No device needed.
TEST(TheiaGpuCullContract, ReserveBeforeInitializeSucceeds) {
    theia::GpuCullPass pass;
    ASSERT_FALSE(pass.isInitialized());
    EXPECT_TRUE(pass.reserve(kInstanceCount)) << "the reserve of setScene, before initialize()";
    EXPECT_EQ(pass.capacity(), 0U) << "there is no list to grow yet";
    EXPECT_TRUE(pass.reserve(1U));
}

TEST_F(RtFixture, GpuCull_ReserveBeforeInitializeSizesTheListOnceInitialized) {
    theia::Scene scene = makeGridScene(deviceCtx(), commandPool());
    ASSERT_EQ(scene.instanceCount(), kInstanceCount);
    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);

    theia::GpuCullPass pass;
    ASSERT_TRUE(pass.reserve(scene.instanceCount()));
    ASSERT_TRUE(pass.initialize(deviceCtx()));
    // The list the scene asked for, whether or not the caller repeats its reserve.
    EXPECT_EQ(pass.capacity(), std::max(theia::GpuCullPass::kInitialCapacity, kInstanceCount));
    EXPECT_TRUE(pass.reserve(scene.instanceCount()));
    EXPECT_EQ(pass.capacity(), std::max(theia::GpuCullPass::kInitialCapacity, kInstanceCount));

    cullScene(pass, scene, commandPool());
    EXPECT_EQ(visibleCount(pass, deviceCtx(), commandPool()), kInstanceCount);
    expectEveryInstanceOnce(pass, kInstanceCount, deviceCtx(), commandPool());
}

// The bounds of an instance follow the animation of a loaded scene: the track of the
// scene file stamps the transform (SceneBase::setAnimationTime), updateTlas syncs it
// and the bounds with it. Computed once at build, they culled the mover wherever it
// was built.
TEST_F(RtFixture, GpuCull_FollowsAnInstanceMovedByTheAnimationOfALoadedScene) {
    const std::filesystem::path dir = writeAnimatedMover();
    theia::Scene scene;
    const auto config =
        harmonia::scene::SceneLoader::load(dir / "mover.scene.toml", dir, scene, deviceCtx(), commandPool());
    ASSERT_TRUE(config.has_value());
    ASSERT_TRUE(scene.hasAnimation());
    ASSERT_EQ(scene.instanceCount(), 1U);
    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);

    // Match the application's animation-clock lifecycle before the first rendered frame.
    scene.setAnimationTime(0.0F);
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
    theia::GpuCullPass pass;
    ASSERT_TRUE(pass.initialize(deviceCtx()));
    cullScene(pass, scene, commandPool());
    EXPECT_EQ(visibleCount(pass, deviceCtx(), commandPool()), 0U) << "built outside the view";

    scene.setAnimationTime(1.0F);
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
    cullScene(pass, scene, commandPool());
    EXPECT_EQ(visibleCount(pass, deviceCtx(), commandPool()), 1U) << "the track moved it into the view";
    expectEveryInstanceOnce(pass, 1U, deviceCtx(), commandPool());

    scene.setAnimationTime(2.0F);
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
    cullScene(pass, scene, commandPool());
    EXPECT_EQ(visibleCount(pass, deviceCtx(), commandPool()), 0U) << "and out of it again";

    std::filesystem::remove_all(dir);
}
