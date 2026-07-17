// Module tests: Scene build path (meshlets + BLAS/TLAS + light/emissive buffers)
// under the mesh/instance model (unique meshes registered once; instances placed
// with a transform).

#include <slang-math/slang-math.hpp>

#include <gtest/gtest.h>
#include <limits>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "theia/scene/Scene.hpp"

TEST_F(RtFixture, Scene_BuildWithMeshAndSphere) {
    Scene scene;
    const uint32_t matDiffuse = scene.addMaterial(Material::diffuse(sm::float3(0.8F), 1.0F));
    const uint32_t matMetal = scene.addMaterial(Material::metal(sm::float3(0.9F, 0.3F, 0.2F), 0.15F));

    MeshData floor = ProceduralGeometry::makeBox(sm::float3(2.0F, 0.1F, 2.0F)); // object space
    const uint32_t floorMesh = scene.addMesh(deviceCtx(), commandPool(), std::move(floor), "test.floor");
    ASSERT_NE(floorMesh, std::numeric_limits<uint32_t>::max());

    const uint32_t sphereMesh = scene.addSphereMesh(deviceCtx(), commandPool(), 0.5F, "test.sphere");
    ASSERT_NE(sphereMesh, std::numeric_limits<uint32_t>::max());

    ASSERT_NE(scene.addInstance(floorMesh, Xform{}, matDiffuse), std::numeric_limits<uint32_t>::max());
    ASSERT_NE(scene.addInstance(sphereMesh, Xform{.translation = {0.0F, 0.5F, 0.0F}}, matMetal),
              std::numeric_limits<uint32_t>::max());

    const VkResult result = scene.build(deviceCtx(), commandPool());
    ASSERT_EQ(result, VK_SUCCESS) << "scene.build() failed: VkResult=" << static_cast<int>(result);

    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE);
    EXPECT_NE(scene.tlasAddress(), VkDeviceAddress{0});
    EXPECT_EQ(scene.instanceCount(), 2U);
    EXPECT_TRUE(scene.meshletBuffer().isValid());
    EXPECT_TRUE(scene.meshletVertexBuffer().isValid());
    EXPECT_TRUE(scene.meshletTriangleBuffer().isValid());
}

TEST_F(RtFixture, Scene_SynthesizesLightFromEmissiveMesh) {
    Scene scene;
    const uint32_t matEmitter = scene.addMaterial(Material::emissive(sm::float3(1.0F), 1000.0F));

    MeshData quad = ProceduralGeometry::makeBox(sm::float3(1.0F, 0.01F, 1.0F));
    const uint32_t emitterMesh = scene.addMesh(deviceCtx(), commandPool(), std::move(quad), "test.emitter");
    ASSERT_NE(emitterMesh, std::numeric_limits<uint32_t>::max());
    ASSERT_NE(scene.addInstance(emitterMesh, Xform{}, matEmitter), std::numeric_limits<uint32_t>::max());

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_GT(scene.lightCount(), 0U);
    EXPECT_GT(scene.emissiveTriangleCount(), 0U);
    EXPECT_TRUE(scene.lightBuffer().isValid());
    EXPECT_TRUE(scene.emissiveTriangleBuffer().isValid());
}

TEST_F(RtFixture, Scene_BuildEmptyFails) {
    Scene scene;
    const VkResult result = scene.build(deviceCtx(), commandPool());
    EXPECT_EQ(result, VK_ERROR_INITIALIZATION_FAILED);
}

TEST_F(RtFixture, Scene_DoesNotSynthesizeLightsForNonEmissiveGeometry) {
    Scene scene;
    const uint32_t matDiffuse = scene.addMaterial(Material::diffuse(sm::float3(0.7F), 1.0F));

    MeshData floor = ProceduralGeometry::makeBox(sm::float3(1.0F, 0.1F, 1.0F));
    const uint32_t floorMesh = scene.addMesh(deviceCtx(), commandPool(), std::move(floor), "test.floor");
    ASSERT_NE(floorMesh, std::numeric_limits<uint32_t>::max());
    ASSERT_NE(scene.addInstance(floorMesh, Xform{}, matDiffuse), std::numeric_limits<uint32_t>::max());

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_EQ(scene.lightCount(), 0U);
    EXPECT_EQ(scene.emissiveTriangleCount(), 0U);
    EXPECT_TRUE(scene.lightBuffer().isValid());
    EXPECT_TRUE(scene.emissiveTriangleBuffer().isValid());
}

// One mesh instanced N times — the core instancing case (one BLAS, N TLAS instances).
TEST_F(RtFixture, Scene_BuildWithMultipleInstancesOfOneMesh) {
    Scene scene;
    const uint32_t mat = scene.addMaterial(Material::diffuse(sm::float3(0.7F), 1.0F));

    MeshData box = ProceduralGeometry::makeBox(sm::float3(0.8F)); // one unique mesh
    const uint32_t mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(box), "test.shared");
    ASSERT_NE(mesh, std::numeric_limits<uint32_t>::max());

    for (int i = 0; i < 4; ++i) {
        const Xform xform{.translation = sm::float3(static_cast<float>(i) * 2.0F, 0.0F, 0.0F)};
        ASSERT_NE(scene.addInstance(mesh, xform, mat), std::numeric_limits<uint32_t>::max()) << "instance " << i;
    }

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE);
    EXPECT_EQ(scene.instanceCount(), 4U);
}
