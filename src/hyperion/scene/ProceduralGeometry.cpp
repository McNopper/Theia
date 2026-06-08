#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "hyperion/scene/ProceduralGeometry.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cstdint>

namespace ProceduralGeometry {
MeshData makeBox(glm::vec3 halfExtent, glm::mat4 transform) {
    struct Face {
        glm::vec3 normal;
        std::array<glm::vec3, 4> positions;
    };

    const glm::vec3 hx = {halfExtent.x, 0.0f, 0.0f};
    const glm::vec3 hy = {0.0f, halfExtent.y, 0.0f};
    const glm::vec3 hz = {0.0f, 0.0f, halfExtent.z};

    const std::array<Face, 6> faces{{
        {{1.0f, 0.0f, 0.0f}, {hx - hy - hz, hx + hy - hz, hx + hy + hz, hx - hy + hz}},
        {{-1.0f, 0.0f, 0.0f}, {-hx - hy + hz, -hx + hy + hz, -hx + hy - hz, -hx - hy - hz}},
        {{0.0f, 1.0f, 0.0f}, {-hx + hy - hz, -hx + hy + hz, hx + hy + hz, hx + hy - hz}},
        {{0.0f, -1.0f, 0.0f}, {-hx - hy + hz, -hx - hy - hz, hx - hy - hz, hx - hy + hz}},
        {{0.0f, 0.0f, 1.0f}, {-hx - hy + hz, hx - hy + hz, hx + hy + hz, -hx + hy + hz}},
        {{0.0f, 0.0f, -1.0f}, {hx - hy - hz, -hx - hy - hz, -hx + hy - hz, hx + hy - hz}},
    }};

    const glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(transform));
    const std::array<glm::vec2, 4> uvs{{
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
    }};

    MeshData mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    for (const Face& face : faces) {
        const uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
        const glm::vec3 transformedNormal = glm::normalize(normalMatrix * face.normal);

        for (size_t i = 0; i < face.positions.size(); ++i) {
            const glm::vec4 transformedPosition = transform * glm::vec4(face.positions[i], 1.0f);
            mesh.vertices.push_back(GpuVertex{
                .position = glm::vec3(transformedPosition),
                .tangentX = 0.0f,
                .normal = transformedNormal,
                .tangentY = 0.0f,
                .uv = uvs[i],
                .tangentZ = 0.0f,
                .bitangentSign = 1.0f,
            });
        }

        mesh.indices.insert(mesh.indices.end(),
                            {
                                baseVertex + 0U,
                                baseVertex + 1U,
                                baseVertex + 2U,
                                baseVertex + 0U,
                                baseVertex + 2U,
                                baseVertex + 3U,
                            });
    }

    return mesh;
}

MeshData makeSphere(glm::vec3 center, float radius, uint32_t rings, uint32_t slices) {
    MeshData mesh;
    mesh.vertices.reserve((rings + 1) * (slices + 1));
    mesh.indices.reserve(rings * slices * 6);

    const float pi = glm::pi<float>();

    for (uint32_t r = 0; r <= rings; ++r) {
        const float v = static_cast<float>(r) / static_cast<float>(rings);
        const float theta = v * pi; // 0 = north pole, PI = south pole
        const float sinT = std::sin(theta);
        const float cosT = std::cos(theta);

        for (uint32_t s = 0; s <= slices; ++s) {
            const float u = static_cast<float>(s) / static_cast<float>(slices);
            const float phi = u * 2.0f * pi;
            const float sinP = std::sin(phi);
            const float cosP = std::cos(phi);

            // Y-up normal: theta=0 → (0,1,0), theta=PI → (0,-1,0)
            const glm::vec3 normal{sinT * cosP, cosT, sinT * sinP};
            // Tangent = dN/dphi normalised (east direction along longitude)
            glm::vec3 tangent{-sinP, 0.0f, cosP};
            if (sinT < 1e-4f)
                tangent = {1.0f, 0.0f, 0.0f}; // pole fallback

            mesh.vertices.push_back(GpuVertex{
                .position = center + radius * normal,
                .tangentX = tangent.x,
                .normal = normal,
                .tangentY = tangent.y,
                .uv = {u, v},
                .tangentZ = tangent.z,
                .bitangentSign = 1.0f,
            });
        }
    }

    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t s = 0; s < slices; ++s) {
            const uint32_t a = r * (slices + 1) + s;
            const uint32_t b = a + 1;
            const uint32_t c = (r + 1) * (slices + 1) + s;
            const uint32_t d = c + 1;
            mesh.indices.insert(mesh.indices.end(), {a, b, c, b, d, c});
        }
    }

    return mesh;
}

SphereAabb makeSphereAabb(glm::vec3 center, float radius) noexcept {
    const glm::vec3 extent(std::max(radius, 0.0f));
    return SphereAabb{
        .min = center - extent,
        .max = center + extent,
    };
}
} // namespace ProceduralGeometry
