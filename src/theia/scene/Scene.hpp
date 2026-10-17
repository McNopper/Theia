#ifndef THEIA_SCENE_SCENE_HPP
#define THEIA_SCENE_SCENE_HPP

#include <volk/volk.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/rt/AccelerationStructure.hpp"
#include "harmonia/scene/Geometry.hpp"
#include "harmonia/scene/ISceneBuilder.hpp"
#include "harmonia/scene/Light.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/SceneBase.hpp"
#include "harmonia/scene/Texture.hpp"

namespace theia {

/// Theia (rasterizer) per-instance GPU layout (std430, 36 bytes). The mesh-shader path
/// draws meshlets, so this carries the mesh's meshlet range. The ranges come from the
/// referenced unique mesh (shared across instances); object→world placement is the
/// per-instance transform (instanceTransformBuffer), read in the mesh shader.
struct GpuInstance {
    std::uint32_t meshIndex = 0;
    std::uint32_t materialIndex = 0;
    std::uint32_t vertexOffset = 0;  ///< mesh's first vertex in global vertex buffer (absolute)
    std::uint32_t indexOffset = 0;   ///< mesh's first index in global index buffer (absolute)
    std::uint32_t indexCount = 0;    ///< number of indices for this mesh
    std::uint32_t meshletOffset = 0; ///< mesh's first meshlet index in meshlet buffer
    std::uint32_t meshletCount = 0;  ///< number of meshlets for this mesh
    std::uint32_t geometryKind = 0;  ///< 0 = triangle mesh, 1 = sphere
    float sphereRadius = 0.0f;
};
static_assert(std::is_trivially_copyable_v<GpuInstance>);
static_assert(sizeof(GpuInstance) == 36);

/// Per-instance world-space AABB for GPU-driven frustum culling (GpuCullPass).
/// Separate from GpuInstance so culling reads a compact, cache-friendly layout.
/// std430, 24 bytes — derived per instance from the mesh's object-space AABB
/// (`Geometry::objectAabb()`) transformed to world space (`worldAabbFromInstance`).
/// Tighter than the bounding sphere it replaces (authorable via `MeshDesc::bounds`).
struct GpuInstanceBounds {
    float minX = 0.0f; ///< world-space AABB min corner x
    float minY = 0.0f; ///< world-space AABB min corner y
    float minZ = 0.0f; ///< world-space AABB min corner z
    float maxX = 0.0f; ///< world-space AABB max corner x
    float maxY = 0.0f; ///< world-space AABB max corner y
    float maxZ = 0.0f; ///< world-space AABB max corner z
};
static_assert(std::is_trivially_copyable_v<GpuInstanceBounds>);
static_assert(sizeof(GpuInstanceBounds) == 24);

/// Per-meshlet descriptor uploaded to GPU (std430, 32 bytes). Bounds are stored in the
/// mesh's OBJECT space; the mesh shader transforms them by the instance matrix for Hi-Z.
struct GpuMeshlet {
    std::uint32_t vertexOffset = 0;   ///< first entry in meshletVertices[]
    std::uint32_t triangleOffset = 0; ///< first uint32 in meshletTriangles[] (holds 4 packed uint8)
    std::uint32_t vertexCount = 0;    ///< number of vertices  (<= 64)
    std::uint32_t triangleCount = 0;  ///< number of triangles (<= 124)
    float centerX = 0.0f;             ///< object-space
    float centerY = 0.0f;             ///< object-space
    float centerZ = 0.0f;             ///< object-space
    float radius = 0.0f;              ///< bounding sphere radius (object-space) for task-shader culling
};
static_assert(std::is_trivially_copyable_v<GpuMeshlet>);
static_assert(sizeof(GpuMeshlet) == 32);

class Scene : public harmonia::scene::SceneBase {
  public:
    // addMaterial / addTexture / addInstance / addMesh / build are inherited concrete from SceneBase.

    [[nodiscard]] std::uint32_t addSphereMesh(const harmonia::DeviceContext& ctx,
                                              const harmonia::core::CommandPool& pool,
                                              float radius,
                                              std::string_view name = "") override;

    [[nodiscard]] VkAccelerationStructureKHR tlas() const noexcept { return m_tlas.handle(); }
    [[nodiscard]] VkDeviceAddress tlasAddress() const noexcept { return m_tlasAddress; }
    [[nodiscard]] const harmonia::core::Buffer& instanceBuffer() const noexcept { return m_instanceBuffer; }
    [[nodiscard]] const harmonia::core::Buffer& instanceBoundsBuffer() const noexcept { return m_instanceBoundsBuffer; }
    [[nodiscard]] const harmonia::core::Buffer& materialBuffer() const noexcept { return m_materialBuffer; }
    [[nodiscard]] const harmonia::core::Buffer& vertexBuffer() const noexcept { return m_vertexBuffer; }
    [[nodiscard]] const harmonia::core::Buffer& indexBuffer() const noexcept { return m_indexBuffer; }
    [[nodiscard]] const harmonia::core::Buffer& meshletBuffer() const noexcept { return m_meshletBuffer; }
    [[nodiscard]] const harmonia::core::Buffer& meshletVertexBuffer() const noexcept { return m_meshletVertexBuffer; }
    [[nodiscard]] const harmonia::core::Buffer& meshletTriangleBuffer() const noexcept {
        return m_meshletTriangleBuffer;
    }
    [[nodiscard]] const harmonia::core::Buffer& lightBuffer() const noexcept { return m_lightBuffer; }
    [[nodiscard]] const harmonia::core::Buffer& emissiveTriangleBuffer() const noexcept {
        return m_emissiveTriangleBuffer;
    }
    [[nodiscard]] const harmonia::core::Buffer& emissiveCdfBuffer() const noexcept { return m_emissiveCdfBuffer; }

    /// Per-instance object→world transforms (one sm::float4x4 per instance, row-major).
    /// Current frame.  Static scenes upload once at build.
    [[nodiscard]] const harmonia::core::Buffer& instanceTransformBuffer() const noexcept {
        return m_instanceTransformBuffer;
    }
    /// Per-instance transforms from the previous frame (motion-vector pass).
    [[nodiscard]] const harmonia::core::Buffer& prevInstanceTransformBuffer() const noexcept {
        return m_prevInstanceTransformBuffer;
    }

    [[nodiscard]] const harmonia::core::Buffer& objectIdBuffer() const noexcept { return m_objectIdBuffer; }

    [[nodiscard]] std::uint32_t instanceCount() const noexcept {
        return static_cast<std::uint32_t>(m_instances.size());
    }
    [[nodiscard]] std::uint32_t meshletCount() const noexcept { return m_meshletCount; }
    [[nodiscard]] std::uint32_t lightCount() const noexcept { return m_lightCount; }
    [[nodiscard]] std::uint32_t emissiveTriangleCount() const noexcept { return m_emissiveTriangleCount; }

    /// Per-mesh GPU layout, computed in buildSceneBuffers.
    struct MeshGpu {
        std::uint32_t vertexOffset = 0;
        std::uint32_t indexOffset = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t meshletOffset = 0;
        std::uint32_t meshletCount = 0;
        std::uint32_t geometryKind = 0;
        float sphereRadius = 0.0f;
    };

  private:
    VkResult buildSceneBuffers(const harmonia::DeviceContext& ctx, const harmonia::core::CommandPool& pool) override;

    [[nodiscard]] std::unique_ptr<harmonia::scene::Geometry>
    createTriangleMesh(const harmonia::DeviceContext& ctx,
                       const harmonia::core::CommandPool& pool,
                       harmonia::scene::MeshData&& data,
                       const harmonia::scene::MeshOpacity& opacity,
                       std::string_view name) override;

    VkResult finishBuild(const harmonia::DeviceContext& ctx, const harmonia::core::CommandPool& pool) override;

    harmonia::rt::AccelerationStructure m_tlas{};
    VkDeviceAddress m_tlasAddress{};

    std::uint32_t instanceMask(std::size_t instanceIndex) const override;

    VkResult uploadBuffer(const harmonia::DeviceContext& ctx,
                          const harmonia::core::CommandPool& pool,
                          std::span<const std::byte> data,
                          VkBufferUsageFlags usage,
                          const char* name,
                          harmonia::core::Buffer& out);

    void buildMeshlets(std::vector<harmonia::GpuVertex>& vertices,
                       std::vector<std::uint32_t>& indices,
                       std::vector<GpuMeshlet>& gpuMeshlets,
                       std::vector<std::uint32_t>& meshletVertices,
                       std::vector<std::uint32_t>& meshletTriangles);
    void buildGpuInstances();
    VkResult uploadSceneBuffers(const harmonia::DeviceContext& ctx,
                                const harmonia::core::CommandPool& pool,
                                const std::vector<harmonia::GpuMaterial>& gpuMaterials,
                                const std::vector<harmonia::GpuVertex>& vertices,
                                const std::vector<std::uint32_t>& indices,
                                const std::vector<GpuMeshlet>& gpuMeshlets,
                                const std::vector<std::uint32_t>& meshletVertices,
                                const std::vector<std::uint32_t>& meshletTriangles);
    void synthesizeEmissiveLights(const std::vector<harmonia::GpuMaterial>& gpuMaterials,
                                  std::vector<harmonia::GpuLight>& gpuLights);

    std::vector<MeshGpu> m_meshGpu;          ///< per-mesh ranges (parallel to m_meshes)
    std::vector<GpuInstance> m_gpuInstances; ///< per-instance GPU rows (built at build)
    std::vector<GpuInstanceBounds> m_instanceBounds;
    harmonia::core::Buffer m_instanceBuffer{};
    harmonia::core::Buffer m_instanceBoundsBuffer{};
    harmonia::core::Buffer m_materialBuffer{};
    harmonia::core::Buffer m_vertexBuffer{};
    harmonia::core::Buffer m_indexBuffer{};
    harmonia::core::Buffer m_meshletBuffer{};
    harmonia::core::Buffer m_meshletVertexBuffer{};
    harmonia::core::Buffer m_meshletTriangleBuffer{};
    harmonia::core::Buffer m_lightBuffer{};
    harmonia::core::Buffer m_emissiveTriangleBuffer{};
    harmonia::core::Buffer m_emissiveCdfBuffer{};
    harmonia::core::Buffer m_instanceTransformBuffer{};
    harmonia::core::Buffer m_prevInstanceTransformBuffer{};
    harmonia::core::Buffer m_objectIdBuffer{};
    std::uint32_t m_emissiveTriangleCount = 0;
    std::uint32_t m_lightCount = 0;
    std::uint32_t m_meshletCount = 0; ///< total meshlets across all meshes (visibility buffer size)
};

} // namespace theia

#endif // THEIA_SCENE_SCENE_HPP
