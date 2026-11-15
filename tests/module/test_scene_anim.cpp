// ANI2 receipt (Theia half of the PERF3 follow-through): the shared
// harmonia::scene_rt TLAS-update machinery consumed 1:1 like Hyperion's.
//
// (1) Loader-driven path: a parsed [[animation]] scene attaches its source +
//     sceneInstance->descInstance mapping at load; setAnimationTime stamps the
//     resolved TRS onto the scene instances; updateTlas refits the TLAS in
//     place (handle + address stay valid - the per-frame TLAS descriptor write
//     keeps working unchanged).
// (2) Direct path: instanceTransform mutations on a static (non-update-capable)
//     TLAS take the full-rebuild branch (fresh handle + address).
// (3) The dirty flag gates GPU work: an update without changes is a no-op.
//
// theia::Scene::build() must be valid Vulkan usage even when the validation
// layer is absent (same contract as test_scene_build.cpp).

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "harmonia/scene/SceneLoader.hpp"
#include "theia/scene/Scene.hpp"

using theia::Scene;

namespace {

/// Writes a throwaway animated scene (sphere node + one linear translation
/// track, ANI1/ANI2 TOML schema) and loads it into @p scene.
[[nodiscard]] bool loadAnimatedScene(Scene& scene,
                                     const harmonia::DeviceContext& ctx,
                                     const harmonia::core::CommandPool& pool,
                                     std::filesystem::path& dirOut) {
    namespace fs = std::filesystem;
    dirOut = fs::temp_directory_path() / "theia_anim_tlas_test";
    fs::create_directories(dirOut);
    constexpr std::string_view kScene = R"toml([animation_clock]
fps = 30

[[mesh]]
name = "ball"
type = "sphere"
radius = 0.5

[[node]]
id = "n"
mesh = "ball"

[[animation]]
target = "nodes/n/translation"
interpolation = "linear"
times = [0.0, 1.0]
values = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0]
)toml";
    {
        std::ofstream scn(dirOut / "s.scene.toml");
        scn << kScene;
    }
    const auto cfg = harmonia::scene::SceneLoader::load(dirOut / "s.scene.toml", dirOut, scene, ctx, pool);
    return cfg.has_value();
}

} // namespace

// Loader-driven animation -> in-place refit (handle + address stay valid).
TEST_F(RtFixture, SceneAnim_LoadedSceneRefitsInPlace) {
    auto& ctx = deviceCtx();
    Scene scene;
    std::filesystem::path dir;
    ASSERT_TRUE(loadAnimatedScene(scene, ctx, commandPool(), dir));

    ASSERT_EQ(scene.build(ctx, commandPool()), VK_SUCCESS);
    EXPECT_TRUE(scene.hasAnimation());
    EXPECT_NEAR(scene.animationFps(), 30.0, 1e-6);
    EXPECT_EQ(scene.instanceCount(), 1U);
    EXPECT_TRUE(scene.meshletBuffer().isValid());

    const VkAccelerationStructureKHR handle0 = scene.tlas();
    const VkDeviceAddress address0 = scene.tlasAddress();
    EXPECT_NE(handle0, VK_NULL_HANDLE);
    EXPECT_NE(address0, VkDeviceAddress{0});

    // t = 0: authored placement; first update refits (counter 0, update-capable).
    scene.setAnimationTime(0.0F);
    EXPECT_TRUE(scene.instancesDirty());
    ASSERT_EQ(scene.updateTlas(ctx, commandPool()), VK_SUCCESS);
    EXPECT_FALSE(scene.instancesDirty()) << "the update must consume the dirty flag";
    EXPECT_EQ(scene.tlas(), handle0) << "refit keeps the TLAS handle";
    EXPECT_EQ(scene.tlasAddress(), address0) << "refit keeps the device address";

    // t = 1 s: the track resolves the node to x = 4 (mapping + stamping receipt).
    scene.setAnimationTime(1.0F);
    EXPECT_NEAR(scene.instances()[0].transform.translation.x, 4.0F, 1e-4);
    ASSERT_EQ(scene.updateTlas(ctx, commandPool()), VK_SUCCESS);
    EXPECT_EQ(scene.tlas(), handle0) << "refit keeps the TLAS handle";
    EXPECT_EQ(scene.tlasAddress(), address0) << "refit keeps the device address";

    // No-change update: the dirty gate makes it a no-op (still VK_SUCCESS).
    ASSERT_EQ(scene.updateTlas(ctx, commandPool()), VK_SUCCESS);
    EXPECT_EQ(scene.tlas(), handle0);

    std::filesystem::remove_all(dir);
}

// Direct mutation on a static (non-update-capable) TLAS -> rebuild branch.
TEST_F(RtFixture, SceneAnim_StaticSceneRebuilds) {
    auto& ctx = deviceCtx();
    Scene scene;
    const std::uint32_t mat = scene.addMaterial(harmonia::scene::Material::diffuse(sm::float3(0.7F), 1.0F));

    harmonia::scene::MeshData floor = harmonia::scene::ProceduralGeometry::makeBox(sm::float3(4.0F, 0.1F, 4.0F));
    const std::uint32_t floorMesh =
        scene.addMesh(ctx, commandPool(), std::move(floor), harmonia::scene::MeshOpacity{}, "test.anim.floor");
    ASSERT_NE(floorMesh, std::numeric_limits<std::uint32_t>::max());
    harmonia::scene::MeshData box = harmonia::scene::ProceduralGeometry::makeBox(sm::float3(1.0F));
    const std::uint32_t boxMesh =
        scene.addMesh(ctx, commandPool(), std::move(box), harmonia::scene::MeshOpacity{}, "test.anim.box");
    ASSERT_NE(boxMesh, std::numeric_limits<std::uint32_t>::max());
    ASSERT_NE(scene.addInstance(floorMesh, harmonia::scene::Transform{}, mat),
              std::numeric_limits<std::uint32_t>::max());
    ASSERT_NE(scene.addInstance(boxMesh, harmonia::scene::Transform{.translation = {0.0F, 1.0F, 0.0F}}, mat),
              std::numeric_limits<std::uint32_t>::max());

    ASSERT_EQ(scene.build(ctx, commandPool()), VK_SUCCESS);
    EXPECT_FALSE(scene.hasAnimation()) << "static scene: the TLAS is not built update-capable";

    const VkAccelerationStructureKHR handle0 = scene.tlas();

    // Direct transform mutation + update: the TLAS is not update-capable, so
    // the shared policy takes the full-rebuild branch (fresh AS + address).
    scene.instanceTransform(1).translation.x = 3.0F;
    EXPECT_TRUE(scene.instancesDirty());
    ASSERT_EQ(scene.updateTlas(ctx, commandPool()), VK_SUCCESS);
    EXPECT_FALSE(scene.instancesDirty());
    EXPECT_NE(scene.tlas(), handle0) << "rebuild creates a fresh acceleration structure";
    EXPECT_NE(scene.tlasAddress(), VkDeviceAddress{0});
    EXPECT_EQ(scene.instanceCount(), 2U);

    // Second update without changes: dirty-gated no-op, TLAS untouched.
    const VkAccelerationStructureKHR handle1 = scene.tlas();
    ASSERT_EQ(scene.updateTlas(ctx, commandPool()), VK_SUCCESS);
    EXPECT_EQ(scene.tlas(), handle1);
}
