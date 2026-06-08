#pragma once

#include <bit>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include "hyperion/GpuTypes.hpp"

/// Photometric unit for light intensity input.
enum class IntensityUnit : uint32_t {
    Nit = 0,     ///< cd/m²  — luminance       (RectLight, SkyLight)
    Candela = 1, ///< cd     — radiant intensity (PointLight, SpotLight)
    Lumen = 2,   ///< lm     — luminous flux     (PointLight, SpotLight isotropic)
    Lux = 3,     ///< lx     — illuminance       (DirectionalLight)
};

/// Abstract base for all scene lights.
///
/// Colors are in linear Rec.2020.
/// Subclasses store intensity in the photometric unit natural to their type and
/// convert to radiometric W via toGpu() (÷ 683 lm/W).
class Light {
  public:
    virtual ~Light() = default;

    virtual LightType lightType() const noexcept = 0;
    virtual GpuLight toGpu() const noexcept = 0;

    std::string name;
    glm::vec3 color{1.0f}; ///< linear Rec.2020 chromaticity, default white
};

/// Rectangular area light (two-sided emitter).
/// Intensity unit: cd/m² (nits / luminance).
class RectLight final : public Light {
  public:
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f}; ///< emission normal (downward default)
    float halfWidth{0.5f};                  ///< half-extent in local X
    float halfHeight{0.5f};                 ///< half-extent in local Y
    float luminance{100.0f};                ///< cd/m²

    LightType lightType() const noexcept override { return LightType::Rect; }
    GpuLight toGpu() const noexcept override;
};

/// Omnidirectional point light.
/// Intensity unit: cd (default) or lm (isotropic).
class PointLight final : public Light {
  public:
    glm::vec3 position{0.0f};
    float intensity{1000.0f}; ///< in `unit`
    IntensityUnit unit{IntensityUnit::Candela};
    float range{0.0f}; ///< 0 = physically correct falloff

    LightType lightType() const noexcept override { return LightType::Point; }
    GpuLight toGpu() const noexcept override;
};

/// Spot light (cone with inner/outer penumbra).
/// Intensity unit: cd (default) or lm.
class SpotLight final : public Light {
  public:
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    float intensity{1000.0f}; ///< in `unit`
    IntensityUnit unit{IntensityUnit::Candela};
    float innerAngleDeg{20.0f};
    float outerAngleDeg{30.0f};
    float range{0.0f}; ///< 0 = no range cutoff

    LightType lightType() const noexcept override { return LightType::Spot; }
    GpuLight toGpu() const noexcept override;
};

/// Infinitely distant directional light (sun/moon).
/// Intensity unit: lux (illuminance).
class DirectionalLight final : public Light {
  public:
    glm::vec3 direction{0.0f, -1.0f, 0.0f}; ///< direction light travels (toward scene)
    float illuminance{100'000.0f};          ///< lux; clear-sky sun ≈ 100 000 lx

    LightType lightType() const noexcept override { return LightType::Directional; }
    GpuLight toGpu() const noexcept override;
};

/// Image-based sky / ambient light.
/// Intensity unit: cd/m² (average sky luminance).
/// The actual IBL texture is managed separately; this carries the tint and scale.
class SkyLight final : public Light {
  public:
    float luminance{1000.0f}; ///< cd/m² — scales env map radiance

    LightType lightType() const noexcept override { return LightType::Sky; }
    GpuLight toGpu() const noexcept override;
};
