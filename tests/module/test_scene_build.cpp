// Module tests: Scene build path (meshlets + BLAS/TLAS + light/emissive buffers).

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <gtest/gtest.h>
#include <limits>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "theia/scene/Scene.hpp"

TEST_F(RtFixture, Scene_BuildWithMeshAndSphere) {
    Scene scene;
    const uint32_t matDiffuse = scene.addMaterial(Material::diffuse(glm::vec3(0.8F), 1.0F));
    const uint32_t matMetal = scene.addMaterial(Material::metal(glm::vec3(0.9F, 0.3F, 0.2F), 0.15F));

    MeshData floor = ProceduralGeometry::makeBox(glm::vec3(2.0F, 0.1F, 2.0F), glm::mat4(1.0F));
    const uint32_t meshInst = scene.addMesh(deviceCtx(), commandPool(), std::move(floor), matDiffuse, "test.floor");
    ASSERT_NE(meshInst, std::numeric_limits<uint32_t>::max());

    const uint32_t sphereInst =
        scene.addSphere(deviceCtx(), commandPool(), glm::vec3(0.0F, 0.5F, 0.0F), 0.5F, matMetal);
    ASSERT_NE(sphereInst, std::numeric_limits<uint32_t>::max());

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
    const uint32_t matEmitter = scene.addMaterial(Material::emissive(glm::vec3(1.0F), 1000.0F));

    MeshData quad = ProceduralGeometry::makeBox(glm::vec3(1.0F, 0.01F, 1.0F), glm::mat4(1.0F));
    const uint32_t emitterInst = scene.addMesh(deviceCtx(), commandPool(), std::move(quad), matEmitter, "test.emitter");
    ASSERT_NE(emitterInst, std::numeric_limits<uint32_t>::max());

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_GT(scene.lightCount(), 0U);
    EXPECT_GT(scene.emissiveTriangleCount(), 0U);
    EXPECT_TRUE(scene.lightBuffer().isValid());
    EXPECT_TRUE(scene.emissiveTriangleBuffer().isValid());
}
