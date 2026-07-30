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

class Scene : public harmonia::SceneBase {
  public:
    // addMaterial / addTexture / addInstance / addMesh / build are inherited concrete from SceneBase.

    [[nodiscard]] std::uint32_t
    addSphereMesh(const harmonia::DeviceContext& ctx, const harmonia::CommandPool& pool, float radius, std::string_view name = "") override;

    [[nodiscard]] VkAccelerationStructureKHR tlas() const noexcept { return m_tlas.handle(); }
    [[nodiscard]] VkDeviceAddress tlasAddress() const noexcept { return m_tlasAddress; }
    [[nodiscard]] const harmonia::Buffer& instanceBuffer() const noexcept { return m_instanceBuffer; }
    [[nodiscard]] const harmonia::Buffer& instanceBoundsBuffer() const noexcept { return m_instanceBoundsBuffer; }
    [[nodiscard]] const harmonia::Buffer& materialBuffer() const noexcept { return m_materialBuffer; }
    [[nodiscard]] const harmonia::Buffer& vertexBuffer() const noexcept { return m_vertexBuffer; }
    [[nodiscard]] const harmonia::Buffer& indexBuffer() const noexcept { return m_indexBuffer; }
    [[nodiscard]] const harmonia::Buffer& meshletBuffer() const noexcept { return m_meshletBuffer; }
    [[nodiscard]] const harmonia::Buffer& meshletVertexBuffer() const noexcept { return m_meshletVertexBuffer; }
    [[nodiscard]] const harmonia::Buffer& meshletTriangleBuffer() const noexcept { return m_meshletTriangleBuffer; }
    [[nodiscard]] const harmonia::Buffer& lightBuffer() const noexcept { return m_lightBuffer; }
    [[nodiscard]] const harmonia::Buffer& emissiveTriangleBuffer() const noexcept { return m_emissiveTriangleBuffer; }
    [[nodiscard]] const harmonia::Buffer& emissiveCdfBuffer() const noexcept { return m_emissiveCdfBuffer; }

    /// Per-instance object→world transforms (one sm::float4x4 per instance, row-major).
    /// Current frame.  Static scenes upload once at build.
    [[nodiscard]] const harmonia::Buffer& instanceTransformBuffer() const noexcept { return m_instanceTransformBuffer; }
    /// Per-instance transforms from the previous frame (motion-vector pass).
    [[nodiscard]] const harmonia::Buffer& prevInstanceTransformBuffer() const noexcept { return m_prevInstanceTransformBuffer; }

    [[nodiscard]] const harmonia::Buffer& objectIdBuffer() const noexcept { return m_objectIdBuffer; }

    [[nodiscard]] std::uint32_t instanceCount() const noexcept {
        return static_cast<std::uint32_t>(m_instances.size());
    }
    [[nodiscard]] std::uint32_t meshletCount() const noexcept { return m_meshletCount; }
    [[nodiscard]] std::uint32_t lightCount() const noexcept { return m_lightCount; }
    [[nodiscard]] std::uint32_t emissiveTriangleCount() const noexcept { return m_emissiveTriangleCount; }

    /// Per-mesh GPU layout + object-space bounds, computed in buildSceneBuffers.
    struct MeshGpu {
        std::uint32_t vertexOffset = 0;
        std::uint32_t indexOffset = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t meshletOffset = 0;
        std::uint32_t meshletCount = 0;
        std::uint32_t geometryKind = 0;
        float sphereRadius = 0.0f;
        float boundsCenterX = 0.0f; ///< object-space bounding sphere
        float boundsCenterY = 0.0f;
        float boundsCenterZ = 0.0f;
        float boundsRadius = 0.0f;
    };

  private:
    VkResult buildSceneBuffers(const harmonia::DeviceContext& ctx, const harmonia::CommandPool& pool) override;

    std::uint32_t instanceMask(std::size_t instanceIndex) const override;

    VkResult uploadBuffer(const harmonia::DeviceContext& ctx,
                          const harmonia::CommandPool& pool,
                          std::span<const std::byte> data,
                          VkBufferUsageFlags usage,
                          const char* name,
                          harmonia::Buffer& out);

    void buildMeshlets(std::vector<harmonia::GpuVertex>& vertices,
                       std::vector<std::uint32_t>& indices,
                       std::vector<GpuMeshlet>& gpuMeshlets,
                       std::vector<std::uint32_t>& meshletVertices,
                       std::vector<std::uint32_t>& meshletTriangles);
    void buildGpuInstances();
    VkResult uploadSceneBuffers(const harmonia::DeviceContext& ctx,
                                const harmonia::CommandPool& pool,
                                const std::vector<harmonia::GpuMaterial>& gpuMaterials,
                                const std::vector<harmonia::GpuVertex>& vertices,
                                const std::vector<std::uint32_t>& indices,
                                const std::vector<GpuMeshlet>& gpuMeshlets,
                                const std::vector<std::uint32_t>& meshletVertices,
                                const std::vector<std::uint32_t>& meshletTriangles);
    void synthesizeEmissiveLights(const std::vector<harmonia::GpuMaterial>& gpuMaterials, std::vector<harmonia::GpuLight>& gpuLights);

    std::vector<MeshGpu> m_meshGpu;          ///< per-mesh ranges (parallel to m_meshes)
    std::vector<GpuInstance> m_gpuInstances; ///< per-instance GPU rows (built at build)
    std::vector<GpuInstanceBounds> m_instanceBounds;
    harmonia::Buffer m_instanceBuffer{};
    harmonia::Buffer m_instanceBoundsBuffer{};
    harmonia::Buffer m_materialBuffer{};
    harmonia::Buffer m_vertexBuffer{};
    harmonia::Buffer m_indexBuffer{};
    harmonia::Buffer m_meshletBuffer{};
    harmonia::Buffer m_meshletVertexBuffer{};
    harmonia::Buffer m_meshletTriangleBuffer{};
    harmonia::Buffer m_lightBuffer{};
    harmonia::Buffer m_emissiveTriangleBuffer{};
    harmonia::Buffer m_emissiveCdfBuffer{};
    harmonia::Buffer m_instanceTransformBuffer{};
    harmonia::Buffer m_prevInstanceTransformBuffer{};
    harmonia::Buffer m_objectIdBuffer{};
    std::uint32_t m_emissiveTriangleCount = 0;
    std::uint32_t m_lightCount = 0;
    std::uint32_t m_meshletCount = 0; ///< total meshlets across all meshes (visibility buffer size)
};
#endif // THEIA_SCENE_SCENE_HPP
