#pragma once

#include <volk/volk.h>

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "hyperion/GpuTypes.hpp"
#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/renderer/AccelerationStructure.hpp"
#include "hyperion/scene/Geometry.hpp"
#include "hyperion/scene/Light.hpp"
#include "hyperion/scene/Material.hpp"
#include "hyperion/scene/Texture.hpp"

class SceneBuilder;

class Scene {
  public:
    using Builder = SceneBuilder;

    [[nodiscard]] uint32_t addMaterial(Material mat);
    [[nodiscard]] uint32_t addMesh(const DeviceContext& ctx,
                                   const CommandPool& pool,
                                   MeshData data,
                                   uint32_t materialIdx,
                                   std::string_view name = "");
    [[nodiscard]] uint32_t
    addSphere(const DeviceContext& ctx, const CommandPool& pool, glm::vec3 center, float radius, uint32_t materialIdx);

    /// Add a light to the scene. Returns the light index.
    /// Must be called before build().
    uint32_t addLight(std::unique_ptr<Light> light);

    /// Add a texture to the scene. Returns the bindless texture index.
    /// Must be called before build() / updateSceneSet().
    [[nodiscard]] uint32_t addTexture(Texture texture);

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
