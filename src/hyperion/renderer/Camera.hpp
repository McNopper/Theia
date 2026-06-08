#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include "hyperion/GpuTypes.hpp"

class Camera {
  public:
    /// Physical camera properties (Unity HDRP / UE5 convention).
    /// EV100 is derived: EV100 = log2(N² × shutterSpeedHz × 100 / iso)
    struct PhysicalCamera {
        float focalLengthMm = 50.0f;          ///< focal length in mm
        float aperture = 8.0f;                ///< f-stop (N)
        float shutterSpeedHz = 100.0f;        ///< 1/shutterSpeedHz seconds (e.g. 100 → 1/100 s)
        float iso = 100.0f;                   ///< sensor sensitivity
        glm::vec2 sensorSizeMm{36.0f, 24.0f}; ///< full-frame 35 mm default

        /// EV100 = log2(N² × t_inv × 100 / ISO), where t_inv = shutterSpeedHz
        [[nodiscard]] float ev100() const noexcept;

        /// Exposure multiplier: 1 / (1.2 × 2^EV100)
        [[nodiscard]] float exposure() const noexcept;
    };

    struct Params {
        glm::vec3 position{278.0f, 273.0f, -800.0f};
        glm::vec3 target{278.0f, 273.0f, 0.0f};
        glm::vec3 up{0.0f, 1.0f, 0.0f};
        float vfovDeg{39.1f};
        float aspectRatio{16.0f / 9.0f};
        float nearPlane{0.1f};
        float farPlane{10000.0f};
        float lensRadius{0.0f};
        float focusDist{1.0f};
        PhysicalCamera physical{};
    };

    explicit Camera() noexcept;
    explicit Camera(Params params) noexcept;

    void setAspect(float aspect) noexcept;
    void setPosition(glm::vec3 pos) noexcept;
    void setTarget(glm::vec3 target) noexcept;
    void setPhysicalCamera(PhysicalCamera physical) noexcept;

    [[nodiscard]] const PhysicalCamera& physicalCamera() const noexcept { return m_params.physical; }
    [[nodiscard]] CameraData getCameraData(uint32_t frameIndex, uint32_t maxDepth) const noexcept;

  private:
    [[nodiscard]] glm::mat4 viewMatrix() const noexcept;
    [[nodiscard]] glm::mat4 projectionMatrix() const noexcept;

    Params m_params;
};
