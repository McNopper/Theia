#pragma once

#include <volk/volk.h>

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/renderer/AccelerationStructure.hpp"
#include "harmonia/scene/Geometry.hpp"
#include "harmonia/scene/ISceneBuilder.hpp"
#include "harmonia/scene/Light.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/Texture.hpp"

class SceneBuilder;

/// Theia (rasterizer) per-instance GPU layout (std430, 32 bytes). The mesh-shader path
/// draws meshlets, so this carries meshletOffset/meshletCount. Distinct from Hyperion's
/// path-tracer layout, which carries indexOffset instead.
struct GpuInstance {
    uint32_t meshIndex     = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexOffset  = 0; ///< first vertex in global vertex buffer (absolute)
    uint32_t meshletOffset = 0; ///< first meshlet index in meshlet buffer
    uint32_t meshletCount  = 0; ///< number of meshlets for this instance
    uint32_t geometryKind  = 0; ///< 0 = triangle mesh, 1 = sphere
    float    sphereRadius  = 0.0f;
    uint32_t _pad          = 0;
};
static_assert(std::is_trivially_copyable_v<GpuInstance>);
static_assert(sizeof(GpuInstance) == 32);

/// Per-meshlet descriptor uploaded to GPU (std430, 32 bytes).
struct GpuMeshlet {
    uint32_t vertexOffset   = 0; ///< first entry in meshletVertices[]
    uint32_t triangleOffset = 0; ///< first uint32 in meshletTriangles[] (holds 4 packed uint8)
    uint32_t vertexCount    = 0; ///< number of vertices  (<= 64)
    uint32_t triangleCount  = 0; ///< number of triangles (<= 124)
    float    centerX        = 0.0f;
    float    centerY        = 0.0f;
    float    centerZ        = 0.0f;
    float    radius         = 0.0f; ///< bounding sphere radius for task-shader culling
};
static_assert(std::is_trivially_copyable_v<GpuMeshlet>);
static_assert(sizeof(GpuMeshlet) == 32);

class Scene : public ISceneBuilder {
  public:
    using Builder = SceneBuilder;

    [[nodiscard]] uint32_t addMaterial(Material mat) override;
    [[nodiscard]] uint32_t addMesh(const DeviceContext& ctx,
                                   const CommandPool& pool,
                                   MeshData data,
                                   uint32_t materialIdx,
                                   std::string_view name = "") override;
    [[nodiscard]] uint32_t addSphere(const DeviceContext& ctx,
                                     const CommandPool& pool,
                                     glm::vec3 center,
                                     float radius,
                                     uint32_t materialIdx) override;

    /// Add a light to the scene. Returns the light index.
    /// Must be called before build().
    uint32_t addLight(std::unique_ptr<Light> light);

    /// Add a texture to the scene. Returns the bindless texture index.
    /// Must be called before build() / updateSceneSet().
    [[nodiscard]] uint32_t addTexture(Texture texture) override;

    VkResult build(const DeviceContext& ctx, const CommandPool& pool);

    [[nodiscard]] VkAccelerationStructureKHR tlas() const noexcept { return m_tlas.handle(); }
    [[nodiscard]] VkDeviceAddress tlasAddress() const noexcept { return m_tlasAddress; }
    [[nodiscard]] const Buffer& instanceBuffer() const noexcept { return m_instanceBuffer; }
    [[nodiscard]] const Buffer& materialBuffer() const noexcept { return m_materialBuffer; }
    [[nodiscard]] const Buffer& vertexBuffer() const noexcept { return m_vertexBuffer; }
    [[nodiscard]] const Buffer& indexBuffer() const noexcept { return m_indexBuffer; }
    [[nodiscard]] const Buffer& meshletBuffer() const noexcept { return m_meshletBuffer; }
    [[nodiscard]] const Buffer& meshletVertexBuffer() const noexcept { return m_meshletVertexBuffer; }
    [[nodiscard]] const Buffer& meshletTriangleBuffer() const noexcept { return m_meshletTriangleBuffer; }
    [[nodiscard]] const Buffer& lightBuffer() const noexcept { return m_lightBuffer; }
    [[nodiscard]] const Buffer& emissiveTriangleBuffer() const noexcept { return m_emissiveTriangleBuffer; }
    [[nodiscard]] const std::vector<Texture>& textures() const noexcept { return m_textures; }
    [[nodiscard]] uint32_t instanceCount() const noexcept { return static_cast<uint32_t>(m_geometries.size()); }
    [[nodiscard]] uint32_t lightCount() const noexcept { return m_lightCount; }
    [[nodiscard]] uint32_t emissiveTriangleCount() const noexcept { return m_emissiveTriangleCount; }

  private:
    VkResult buildSceneBuffers(const DeviceContext& ctx, const CommandPool& pool);
    VkResult buildTlas(const DeviceContext& ctx, const CommandPool& pool);

    std::vector<Material> m_materials;
    std::vector<GpuInstance> m_instances;
    std::vector<std::unique_ptr<Geometry>> m_geometries;
    std::vector<std::unique_ptr<Light>> m_lights;
    std::vector<Texture> m_textures;
    Buffer m_instanceBuffer{};
    Buffer m_materialBuffer{};
    Buffer m_vertexBuffer{};
    Buffer m_indexBuffer{};
    Buffer m_meshletBuffer{};
    Buffer m_meshletVertexBuffer{};
    Buffer m_meshletTriangleBuffer{};
    Buffer m_lightBuffer{};
    Buffer m_emissiveTriangleBuffer{};
    uint32_t m_emissiveTriangleCount = 0;
    uint32_t m_lightCount = 0;
    AccelerationStructure m_tlas{};
    VkDeviceAddress m_tlasAddress{};
};
