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
#include "harmonia/scene/SceneBase.hpp"
#include "harmonia/scene/Texture.hpp"

class SceneBuilder;

/// Theia (rasterizer) per-instance GPU layout (std430, 36 bytes). The mesh-shader path
/// draws meshlets, so this carries meshletOffset/meshletCount. Distinct from Hyperion's
/// path-tracer layout, which carries indexOffset instead.
struct GpuInstance {
    uint32_t meshIndex     = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexOffset  = 0; ///< first vertex in global vertex buffer (absolute)
    uint32_t indexOffset   = 0; ///< first index in global index buffer (absolute triangle index stream)
    uint32_t indexCount    = 0; ///< number of indices in global index buffer for this instance
    uint32_t meshletOffset = 0; ///< first meshlet index in meshlet buffer
    uint32_t meshletCount  = 0; ///< number of meshlets for this instance
    uint32_t geometryKind  = 0; ///< 0 = triangle mesh, 1 = sphere
    float    sphereRadius  = 0.0f;
};
static_assert(std::is_trivially_copyable_v<GpuInstance>);
static_assert(sizeof(GpuInstance) == 36);

/// Per-instance bounding sphere for GPU-driven frustum culling (GpuCullPass).
/// Separate from GpuInstance so culling reads a compact, cache-friendly layout.
/// std430, 16 bytes — stored world-space; vertices are baked into world space at load time.
struct GpuInstanceBounds {
    float centerX = 0.0f; ///< world-space bounding sphere centre x
    float centerY = 0.0f; ///< world-space bounding sphere centre y
    float centerZ = 0.0f; ///< world-space bounding sphere centre z
    float radius  = 0.0f; ///< bounding sphere radius; 0 = empty / never drawn
};
static_assert(std::is_trivially_copyable_v<GpuInstanceBounds>);
static_assert(sizeof(GpuInstanceBounds) == 16);

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

class Scene : public harmonia::SceneBase {
  public:
    using Builder = SceneBuilder;

    [[nodiscard]] uint32_t addMaterial(Material&& mat) override { return harmonia::SceneBase::addMaterial(std::move(mat)); }
    [[nodiscard]] uint32_t addTexture(Texture&& texture) override { return harmonia::SceneBase::addTexture(std::move(texture)); }

    [[nodiscard]] uint32_t addMesh(const DeviceContext& ctx,
                                   const CommandPool& pool,
                                   MeshData&& data,
                                   uint32_t materialIdx,
                                   std::string_view name = "") override;
    [[nodiscard]] uint32_t addSphere(const DeviceContext& ctx,
                                     const CommandPool& pool,
                                     sm::float3 center,
                                     float radius,
                                     uint32_t materialIdx) override;

    VkResult build(const DeviceContext& ctx, const CommandPool& pool);

    [[nodiscard]] VkAccelerationStructureKHR tlas() const noexcept { return m_tlas.handle(); }
    [[nodiscard]] VkDeviceAddress tlasAddress() const noexcept { return m_tlasAddress; }
    [[nodiscard]] const Buffer& instanceBuffer() const noexcept { return m_instanceBuffer; }
    [[nodiscard]] const Buffer& instanceBoundsBuffer() const noexcept { return m_instanceBoundsBuffer; }
    [[nodiscard]] const Buffer& materialBuffer() const noexcept { return m_materialBuffer; }
    [[nodiscard]] const Buffer& vertexBuffer() const noexcept { return m_vertexBuffer; }
    [[nodiscard]] const Buffer& indexBuffer() const noexcept { return m_indexBuffer; }
    [[nodiscard]] const Buffer& meshletBuffer() const noexcept { return m_meshletBuffer; }
    [[nodiscard]] const Buffer& meshletVertexBuffer() const noexcept { return m_meshletVertexBuffer; }
    [[nodiscard]] const Buffer& meshletTriangleBuffer() const noexcept { return m_meshletTriangleBuffer; }
    [[nodiscard]] const Buffer& lightBuffer() const noexcept { return m_lightBuffer; }
    [[nodiscard]] const Buffer& emissiveTriangleBuffer() const noexcept { return m_emissiveTriangleBuffer; }
    [[nodiscard]] const Buffer& emissiveCdfBuffer() const noexcept { return m_emissiveCdfBuffer; }

    /// Per-instance world transforms (one sm::float4x4 per instance, row-major).
    /// For the current frame.  In the present static-scene model, transforms are
    /// baked into vertex positions at load time so these matrices are always identity.
    [[nodiscard]] const Buffer& instanceTransformBuffer() const noexcept { return m_instanceTransformBuffer; }
    /// Per-instance world transforms from the previous frame.  Used by the motion
    /// vector pass to correct per-object motion for dynamic objects.  Equal to
    /// instanceTransformBuffer() for static scenes (all identity).
    [[nodiscard]] const Buffer& prevInstanceTransformBuffer() const noexcept { return m_prevInstanceTransformBuffer; }

    /// Stable per-object IDs: buffer of uint32_t, one per instance,
    /// where objectId[i] == i (the instance's index in the scene array).
    /// Persistent across frames as long as the scene is not rebuilt.
    /// Suitable as a temporal reservoir key for ReSTIR DI / A-SVGF.
    [[nodiscard]] const Buffer& objectIdBuffer() const noexcept { return m_objectIdBuffer; }

    /// Light IDs are stable indices [0, lightCount()).
    /// Safe for temporal reservoir indexing — the light array does not
    /// change between frames within a loaded scene.
    [[nodiscard]] uint32_t instanceCount() const noexcept { return static_cast<uint32_t>(m_geometries.size()); }
    /// Total number of meshlets across all instances (size of the per-meshlet visibility buffer).
    [[nodiscard]] uint32_t meshletCount() const noexcept { return m_meshletCount; }
    [[nodiscard]] uint32_t lightCount() const noexcept { return m_lightCount; }
    [[nodiscard]] uint32_t emissiveTriangleCount() const noexcept { return m_emissiveTriangleCount; }

  private:
    VkResult buildSceneBuffers(const DeviceContext& ctx, const CommandPool& pool);
    VkResult buildTlas(const DeviceContext& ctx, const CommandPool& pool);

    std::vector<GpuInstance> m_instances;
    std::vector<GpuInstanceBounds> m_instanceBounds;
    Buffer m_instanceBuffer{};
    Buffer m_instanceBoundsBuffer{};
    Buffer m_materialBuffer{};
    Buffer m_vertexBuffer{};
    Buffer m_indexBuffer{};
    Buffer m_meshletBuffer{};
    Buffer m_meshletVertexBuffer{};
    Buffer m_meshletTriangleBuffer{};
    Buffer m_lightBuffer{};
    Buffer m_emissiveTriangleBuffer{};
    Buffer m_emissiveCdfBuffer{};
    Buffer m_instanceTransformBuffer{};
    Buffer m_prevInstanceTransformBuffer{};
    Buffer m_objectIdBuffer{};
    uint32_t m_emissiveTriangleCount = 0;
    uint32_t m_lightCount = 0;
    uint32_t m_meshletCount = 0; ///< total meshlets across all instances (visibility buffer size)
    AccelerationStructure m_tlas{};
    VkDeviceAddress m_tlasAddress{};
};
