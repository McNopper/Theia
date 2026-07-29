#ifndef THEIA_SCENE_SCENE_HPP
#define THEIA_SCENE_SCENE_HPP

#include <volk/volk.h>

#include <cstdint>
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

/// Theia (rasterizer) per-instance GPU layout (std430, 36 bytes). The mesh-shader path
/// draws meshlets, so this carries the mesh's meshlet range. The ranges come from the
/// referenced unique mesh (shared across instances); object→world placement is the
/// per-instance transform (instanceTransformBuffer), read in the mesh shader.
struct GpuInstance {
    uint32_t meshIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexOffset = 0;  ///< mesh's first vertex in global vertex buffer (absolute)
    uint32_t indexOffset = 0;   ///< mesh's first index in global index buffer (absolute)
    uint32_t indexCount = 0;    ///< number of indices for this mesh
    uint32_t meshletOffset = 0; ///< mesh's first meshlet index in meshlet buffer
    uint32_t meshletCount = 0;  ///< number of meshlets for this mesh
    uint32_t geometryKind = 0;  ///< 0 = triangle mesh, 1 = sphere
    float sphereRadius = 0.0f;
};
static_assert(std::is_trivially_copyable_v<GpuInstance>);
static_assert(sizeof(GpuInstance) == 36);

/// Per-instance bounding sphere for GPU-driven frustum culling (GpuCullPass).
/// Separate from GpuInstance so culling reads a compact, cache-friendly layout.
/// std430, 16 bytes — stored world-space (computed from the mesh's object-space
/// bounds transformed by the instance transform).
struct GpuInstanceBounds {
    float centerX = 0.0f; ///< world-space bounding sphere centre x
    float centerY = 0.0f; ///< world-space bounding sphere centre y
    float centerZ = 0.0f; ///< world-space bounding sphere centre z
    float radius = 0.0f;  ///< bounding sphere radius; 0 = empty / never drawn
};
static_assert(std::is_trivially_copyable_v<GpuInstanceBounds>);
static_assert(sizeof(GpuInstanceBounds) == 16);

/// Per-meshlet descriptor uploaded to GPU (std430, 32 bytes). Bounds are stored in the
/// mesh's OBJECT space; the mesh shader transforms them by the instance matrix for Hi-Z.
struct GpuMeshlet {
    uint32_t vertexOffset = 0;   ///< first entry in meshletVertices[]
    uint32_t triangleOffset = 0; ///< first uint32 in meshletTriangles[] (holds 4 packed uint8)
    uint32_t vertexCount = 0;    ///< number of vertices  (<= 64)
    uint32_t triangleCount = 0;  ///< number of triangles (<= 124)
    float centerX = 0.0f;        ///< object-space
    float centerY = 0.0f;        ///< object-space
    float centerZ = 0.0f;        ///< object-space
    float radius = 0.0f;         ///< bounding sphere radius (object-space) for task-shader culling
};
static_assert(std::is_trivially_copyable_v<GpuMeshlet>);
static_assert(sizeof(GpuMeshlet) == 32);

class Scene : public harmonia::SceneBase {
  public:
    // addMaterial / addTexture / addInstance / addMesh / build are inherited concrete from SceneBase.

    [[nodiscard]] uint32_t
    addSphereMesh(const DeviceContext& ctx, const CommandPool& pool, float radius, std::string_view name = "") override;

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

    /// Per-instance object→world transforms (one sm::float4x4 per instance, row-major).
    /// Current frame.  Static scenes upload once at build.
    [[nodiscard]] const Buffer& instanceTransformBuffer() const noexcept { return m_instanceTransformBuffer; }
    /// Per-instance transforms from the previous frame (motion-vector pass).
    [[nodiscard]] const Buffer& prevInstanceTransformBuffer() const noexcept { return m_prevInstanceTransformBuffer; }

    [[nodiscard]] const Buffer& objectIdBuffer() const noexcept { return m_objectIdBuffer; }

    [[nodiscard]] uint32_t instanceCount() const noexcept { return static_cast<uint32_t>(m_instances.size()); }
    [[nodiscard]] uint32_t meshletCount() const noexcept { return m_meshletCount; }
    [[nodiscard]] uint32_t lightCount() const noexcept { return m_lightCount; }
    [[nodiscard]] uint32_t emissiveTriangleCount() const noexcept { return m_emissiveTriangleCount; }

    /// Per-mesh GPU layout + object-space bounds, computed in buildSceneBuffers.
    struct MeshGpu {
        uint32_t vertexOffset = 0;
        uint32_t indexOffset = 0;
        uint32_t indexCount = 0;
        uint32_t meshletOffset = 0;
        uint32_t meshletCount = 0;
        uint32_t geometryKind = 0;
        float sphereRadius = 0.0f;
        float boundsCenterX = 0.0f; ///< object-space bounding sphere
        float boundsCenterY = 0.0f;
        float boundsCenterZ = 0.0f;
        float boundsRadius = 0.0f;
    };

  private:
    VkResult buildSceneBuffers(const DeviceContext& ctx, const CommandPool& pool) override;
    VkResult buildTlas(const DeviceContext& ctx, const CommandPool& pool) override;

    std::vector<MeshGpu> m_meshGpu;          ///< per-mesh ranges (parallel to m_meshes)
    std::vector<GpuInstance> m_gpuInstances; ///< per-instance GPU rows (built at build)
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
    uint32_t m_meshletCount = 0; ///< total meshlets across all meshes (visibility buffer size)
    AccelerationStructure m_tlas{};
    VkDeviceAddress m_tlasAddress{};
};
#endif // THEIA_SCENE_SCENE_HPP
