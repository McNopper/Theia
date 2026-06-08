#include "hyperion/scene/Light.hpp"

#include <bit>
#include <cmath>
#include <numbers>

namespace {

/// Luminous efficacy of radiation (lm/W).
inline constexpr float kLumEfficacy = 683.0f;

/// Pack a LightType enum value into the float slot of GpuLight.
/// (Matches the asuint() read in the shader.)
inline float packType(LightType t) noexcept {
    const uint32_t u = static_cast<uint32_t>(t);
    return std::bit_cast<float>(u);
}

/// Convert luminous flux [lm] to luminous intensity [cd] for an isotropic source.
inline float lumenToCandela(float lm) noexcept {
    return lm / (4.0f * std::numbers::pi_v<float>);
}

} // namespace

// ---------------------------------------------------------------------------
// RectLight
// ---------------------------------------------------------------------------
GpuLight RectLight::toGpu() const noexcept {
    // Exitant radiance L [W/sr/m²] = luminance [cd/m²] / 683
    GpuLight g{};
    g.position = position;
    g.type = packType(LightType::Rect);
    g.direction = glm::normalize(direction);
    g.range = 0.0f;
    g.color = color;
    g.intensity = luminance / kLumEfficacy;
    g.halfWidth = halfWidth;
    g.halfHeight = halfHeight;
    g.cosInner = 0.0f;
    g.cosOuter = 0.0f;
    return g;
}

// ---------------------------------------------------------------------------
// PointLight
// ---------------------------------------------------------------------------
GpuLight PointLight::toGpu() const noexcept {
    // Radiant intensity I [W/sr] = luminous intensity [cd] / 683
    float cd = intensity;
    if (unit == IntensityUnit::Lumen) {
        cd = lumenToCandela(intensity);
    }

    GpuLight g{};
    g.position = position;
    g.type = packType(LightType::Point);
    g.direction = glm::vec3(0.0f, -1.0f, 0.0f); // unused for point
    g.range = range;
    g.color = color;
    g.intensity = cd / kLumEfficacy;
    g.halfWidth = 0.0f;
    g.halfHeight = 0.0f;
    g.cosInner = 0.0f;
    g.cosOuter = 0.0f;
    return g;
}

// ---------------------------------------------------------------------------
// SpotLight
// ---------------------------------------------------------------------------
GpuLight SpotLight::toGpu() const noexcept {
    float cd = intensity;
    if (unit == IntensityUnit::Lumen) {
        cd = lumenToCandela(intensity);
    }

    GpuLight g{};
    g.position = position;
    g.type = packType(LightType::Spot);
    g.direction = glm::normalize(direction);
    g.range = range;
    g.color = color;
    g.intensity = cd / kLumEfficacy;
    g.halfWidth = 0.0f;
    g.halfHeight = 0.0f;
    g.cosInner = std::cos(glm::radians(innerAngleDeg));
    g.cosOuter = std::cos(glm::radians(outerAngleDeg));
    return g;
}

// ---------------------------------------------------------------------------
// DirectionalLight
// ---------------------------------------------------------------------------
GpuLight DirectionalLight::toGpu() const noexcept {
    // Store irradiance E [W/m²] = illuminance [lux] / 683.
    // In the shader: contribution = color * intensity * BSDF * visibility
    // (the cosine term is already in the BSDF evaluation).
    GpuLight g{};
    g.position = glm::vec3(0.0f); // unused
    g.type = packType(LightType::Directional);
    g.direction = glm::normalize(direction);
    g.range = 0.0f;
    g.color = color;
    g.intensity = illuminance / kLumEfficacy;
    g.halfWidth = 0.0f;
    g.halfHeight = 0.0f;
    g.cosInner = 0.0f;
    g.cosOuter = 0.0f;
    return g;
}

// ---------------------------------------------------------------------------
// SkyLight
// ---------------------------------------------------------------------------
GpuLight SkyLight::toGpu() const noexcept {
    // Sky radiance scale [W/sr/m²] = luminance [cd/m²] / 683.
    GpuLight g{};
    g.position = glm::vec3(0.0f);
    g.type = packType(LightType::Sky);
    g.direction = glm::vec3(0.0f, 1.0f, 0.0f);
    g.range = 0.0f;
    g.color = color;
    g.intensity = luminance / kLumEfficacy;
    g.halfWidth = 0.0f;
    g.halfHeight = 0.0f;
    g.cosInner = 0.0f;
    g.cosOuter = 0.0f;
    return g;
}
