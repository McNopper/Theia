#pragma once

#include <volk/volk.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <expected>
#include <memory>
#include <string>
#include <string_view>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/renderer/AccelerationStructure.hpp"
#include "hyperion/scene/Mesh.hpp"

struct Xform {
    glm::vec3 translation = {0.0f, 0.0f, 0.0f};
    glm::quat rotation = glm::identity<glm::quat>();
    glm::vec3 scale = {1.0f, 1.0f, 1.0f};

    [[nodiscard]] glm::mat4 matrix() const noexcept;
    [[nodiscard]] glm::mat4 inverseMatrix() const noexcept;
    [[nodiscard]] VkTransformMatrixKHR toVkTransform() const noexcept;
};

class Geometry {
  public:
    Xform xform;
    uint32_t materialIndex = 0;

    virtual ~Geometry() = default;

    virtual VkResult buildBlas(const DeviceContext& ctx, const CommandPool& pool) = 0;
    [[nodiscard]] virtual VkAccelerationStructureInstanceKHR makeInstance(uint32_t instanceIndex) const noexcept = 0;

    [[nodiscard]] VkAccelerationStructureKHR blas() const noexcept { return m_blas; }

  protected:
    VkAccelerationStructureKHR m_blas = VK_NULL_HANDLE;
};

class TriangleMesh final : public Geometry {
  public:
    [[nodiscard]] static std::expected<std::unique_ptr<TriangleMesh>, VkResult> create(const DeviceContext& ctx,
                                                                                       const CommandPool& pool,
                                                                                       MeshData data,
                                                                                       uint32_t materialIndex,
                                                                                       std::string_view debugName = "");

    VkResult buildBlas(const DeviceContext& ctx, const CommandPool& pool) override;
    [[nodiscard]] VkAccelerationStructureInstanceKHR makeInstance(uint32_t instanceIndex) const noexcept override;

    [[nodiscard]] const Buffer& vertexBuffer() const noexcept;
    [[nodiscard]] const Buffer& indexBuffer() const noexcept;
    [[nodiscard]] uint32_t vertexCount() const noexcept;
    [[nodiscard]] uint32_t indexCount() const noexcept;
    [[nodiscard]] const MeshData& data() const noexcept;

  private:
    MeshData m_data{};
    Mesh m_mesh{};
    AccelerationStructure m_accelerationStructure{};
    std::string m_debugName{};
};

class Sphere final : public Geometry {
  public:
    [[nodiscard]] static std::expected<std::unique_ptr<Sphere>, VkResult> create(const DeviceContext& ctx,
                                                                                 const CommandPool& pool,
                                                                                 glm::vec3 center,
                                                                                 float radius,
                                                                                 uint32_t materialIndex,
                                                                                 std::string_view debugName = "");

    VkResult buildBlas(const DeviceContext& ctx, const CommandPool& pool) override;
    [[nodiscard]] VkAccelerationStructureInstanceKHR makeInstance(uint32_t instanceIndex) const noexcept override;

    [[nodiscard]] glm::vec3 center() const noexcept;
    [[nodiscard]] float radius() const noexcept;

  private:
    glm::vec3 m_center{};
    float m_radius = 0.0f;
    Buffer m_aabbBuffer{};
    AccelerationStructure m_accelerationStructure{};
    std::string m_debugName{};
};
