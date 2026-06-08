#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <algorithm>

#include "hyperion/GpuTypes.hpp"

class Material {
  public:
    [[nodiscard]] static Material diffuse(glm::vec3 color, float roughness = 1.0f, float specIor = 1.5f) {
        Material material;
        material.m_gpu.baseColorWeight = glm::vec4(glm::max(color, glm::vec3(0.0f)), 1.0f);
        material.m_gpu.baseMetalnessDiffRough = glm::vec4(0.0f, std::clamp(roughness, 0.0f, 1.0f), 0.0f, 0.0f);
        material.m_gpu.specularColorWeight = glm::vec4(1.0f);
        material.m_gpu.specularRoughAnisoIor =
            glm::vec4(std::clamp(roughness, 0.0f, 1.0f), 0.0f, std::max(specIor, 1.0f), 0.0f);
        material.m_gpu.transmissionColorWeight = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
        material.m_gpu.subsurfaceRadiusScale = glm::vec4(1.0f);
        material.m_gpu.textureIndices = glm::uvec4(kNoTexture);
        material.m_gpu.textureIndices2 = glm::uvec4(kNoTexture);
        material.m_gpu.coatColorWeight = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
        material.m_gpu.coatRoughAnisoIorDark = glm::vec4(0.0f, 0.0f, 1.6f, 0.0f);
        material.m_gpu.fuzzColorWeight = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
        material.m_gpu.fuzzRoughPad = glm::vec4(0.5f, 0.0f, 0.0f, 0.0f);
        material.m_gpu.opacityFlagsPad = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
        return material;
    }

    [[nodiscard]] static Material metal(glm::vec3 color, float roughness = 0.0f) {
        Material material = diffuse(color, roughness, 1.5f);
        material.m_gpu.baseMetalnessDiffRough.x = 1.0f;
        material.m_gpu.baseMetalnessDiffRough.y = 0.0f;
        material.m_gpu.specularColorWeight = glm::vec4(glm::max(color, glm::vec3(0.0f)), 1.0f);
        material.m_gpu.specularRoughAnisoIor.x = std::clamp(roughness, 0.0f, 1.0f);
        return material;
    }

    [[nodiscard]] static Material mirror(glm::vec3 color = {0.99f, 0.99f, 0.99f}) {
        Material material = metal(color, 0.0f);
        material.m_gpu.specularRoughAnisoIor.x = 0.0f;
        material.m_gpu.opacityFlagsPad = glm::vec4(1.0f, 3.0f, 0.0f, 0.0f); // flags=3 → mirror (delta reflect)
        return material;
    }

    [[nodiscard]] static Material glass(float ior = 1.52f, float roughness = 0.0f) {
        Material material = diffuse(glm::vec3(1.0f), roughness, ior);
        material.m_gpu.baseColorWeight = glm::vec4(1.0f);
        material.m_gpu.baseMetalnessDiffRough = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        material.m_gpu.specularColorWeight = glm::vec4(1.0f);
        material.m_gpu.specularRoughAnisoIor =
            glm::vec4(std::clamp(roughness, 0.0f, 1.0f), 0.0f, std::max(ior, 1.0f), 0.0f);
        material.m_gpu.transmissionColorWeight = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        material.m_gpu.transmissionParams = glm::vec4(0.0f, std::clamp(roughness, 0.0f, 1.0f), 0.0f, 0.0f);
        material.m_gpu.opacityFlagsPad = glm::vec4(1.0f, 2.0f, 0.0f, 0.0f); // opacity=1, flags=2 → glass
        return material;
    }

    [[nodiscard]] static Material emissive(glm::vec3 color, float luminanceNits) {
        Material material = diffuse(color, 1.0f, 1.5f);
        material.m_gpu.emissionColorLum = glm::vec4(glm::max(color, glm::vec3(0.0f)), std::max(luminanceNits, 0.0f));
        return material;
    }

    /// Construct a Material wrapping an already-populated GpuMaterial.
    [[nodiscard]] static Material fromGpu(GpuMaterial g) {
        Material m;
        m.m_gpu = g;
        return m;
    }

    Material& baseWeight(float w) {
        m_gpu.baseColorWeight.w = std::clamp(w, 0.0f, 1.0f);
        return *this;
    }

    Material& coat(float w, glm::vec3 color = {1.0f, 1.0f, 1.0f}, float ior = 1.6f, float rough = 0.0f) {
        m_gpu.coatColorWeight = glm::vec4(glm::max(color, glm::vec3(0.0f)), std::clamp(w, 0.0f, 1.0f));
        m_gpu.coatRoughAnisoIorDark = glm::vec4(std::clamp(rough, 0.0f, 1.0f), 0.0f, std::max(ior, 1.0f), 0.0f);
        return *this;
    }

    Material& fuzz(float w, glm::vec3 color = {1.0f, 1.0f, 1.0f}, float rough = 0.5f) {
        m_gpu.fuzzColorWeight = glm::vec4(glm::max(color, glm::vec3(0.0f)), std::clamp(w, 0.0f, 1.0f));
        m_gpu.fuzzRoughPad = glm::vec4(std::clamp(rough, 0.0f, 1.0f), 0.0f, 0.0f, 0.0f);
        return *this;
    }

    Material& subsurface(float w, glm::vec3 color, glm::vec3 radius, float scale = 1.0f) {
        m_gpu.subsurfaceColorWeight = glm::vec4(glm::max(color, glm::vec3(0.0f)), std::clamp(w, 0.0f, 1.0f));
        m_gpu.subsurfaceRadiusScale = glm::vec4(glm::max(radius, glm::vec3(0.0f)), std::max(scale, 0.0f));
        return *this;
    }

    Material& opacity(float o) {
        m_gpu.opacityFlagsPad.x = std::clamp(o, 0.0f, 1.0f);
        // flags (y) intentionally NOT touched — caller must set it explicitly
        return *this;
    }

    /// Set a bindless texture index for a given map slot.
    /// Slots 0-3 → textureIndices [base_color, normal, ORM, emission];
    /// slots 4-6 → textureIndices2 [coat_normal, tangent, coat_tangent].
    void setTextureIndex(uint32_t slot, uint32_t idx) noexcept {
        switch (slot) {
        case 0:
            m_gpu.textureIndices.x = idx;
            break;
        case 1:
            m_gpu.textureIndices.y = idx;
            break;
        case 2:
            m_gpu.textureIndices.z = idx;
            break;
        case 3:
            m_gpu.textureIndices.w = idx;
            break;
        case 4:
            m_gpu.textureIndices2.x = idx;
            break;
        case 5:
            m_gpu.textureIndices2.y = idx;
            break;
        case 6:
            m_gpu.textureIndices2.z = idx;
            break;
        default:
            break;
        }
    }

    [[nodiscard]] const GpuMaterial& gpu() const noexcept { return m_gpu; }

  private:
    GpuMaterial m_gpu{};
};
