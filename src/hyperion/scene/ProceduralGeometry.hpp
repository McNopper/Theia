#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include "hyperion/scene/Mesh.hpp"

namespace ProceduralGeometry {
[[nodiscard]] MeshData makeBox(glm::vec3 halfExtent, glm::mat4 transform = glm::mat4(1.0f));

/// Generate a UV sphere mesh centred at @p center with the given radius.
/// @p rings: latitude bands (default 32), @p slices: longitude segments (default 64).
/// Normals, tangents and UVs are generated analytically — no flat shading artefacts.
[[nodiscard]] MeshData makeSphere(glm::vec3 center, float radius, uint32_t rings = 32, uint32_t slices = 64);

struct SphereAabb {
    glm::vec3 min{};
    glm::vec3 max{};
};

[[nodiscard]] SphereAabb makeSphereAabb(glm::vec3 center, float radius) noexcept;
} // namespace ProceduralGeometry
