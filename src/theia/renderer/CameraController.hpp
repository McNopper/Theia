#ifndef THEIA_RENDERER_CAMERACONTROLLER_HPP
#define THEIA_RENDERER_CAMERACONTROLLER_HPP

#include <algorithm>
#include <cmath>

#include <slang-math/slang-math.hpp>

namespace theia {

/// First-person camera controller: the interactive look/move state (WASD +
/// right-mouse-drag look + scroll-speed) and the pure camera math derived from it.
///
/// SDL event dispatch (relative-mouse mode, EV100 keys) and the camera parameters
/// (position/target/up/physical, projection) stay in Application; this class owns the
/// controller state plus the yaw/pitch -> direction, motion, and movement-integration
/// math. Extracted from the Theia Application God Object (R8/CH9).
class CameraController {
  public:
    float yaw = -90.0f;        ///< degrees, -90 = looking toward -Z (Cornell default)
    float pitch = 0.0f;        ///< degrees, clamped +/-89
    float speed = 50.0f;       ///< units/second
    float sensitivity = 0.15f; ///< degrees per pixel of mouse movement
    bool captured = false;     ///< whether relative mouse mode is active
    bool wDown = false;
    bool aDown = false;
    bool sDown = false;
    bool dDown = false;
    bool qDown = false; ///< Q = move down
    bool eDown = false; ///< E = move up

    /// Forward direction rebuilt from yaw/pitch.
    [[nodiscard]] sm::float3 forward() const noexcept {
        const float yawRad = sm::radians(yaw);
        const float pitchRad = sm::radians(pitch);
        return sm::float3{std::cos(yawRad) * std::cos(pitchRad), std::sin(pitchRad),
                          std::sin(yawRad) * std::cos(pitchRad)};
    }

    /// Apply relative mouse motion to yaw/pitch (clamped to avoid gimbal lock at the poles).
    void handleMotion(float dxRel, float dyRel) noexcept {
        yaw += dxRel * sensitivity;
        pitch -= dyRel * sensitivity;
        pitch = std::max(-89.0f, std::min(89.0f, pitch));
    }

    /// Scale movement speed by pow(1.2, steps) (mouse wheel), clamped to [1, 5000].
    void adjustSpeed(float steps) noexcept {
        speed *= std::pow(1.2f, steps);
        speed = std::max(1.0f, std::min(5000.0f, speed));
    }

    /// Integrate WASD/Q/E movement over dt, returning the updated position.
    [[nodiscard]] sm::float3 integrate(const sm::float3& pos, float dt) const noexcept {
        const sm::float3 fwd = forward();
        const sm::float3 worldUp{0.0f, 1.0f, 0.0f};
        const sm::float3 right = sm::normalize(sm::cross(fwd, worldUp));
        const sm::float3 up = sm::normalize(sm::cross(right, fwd));
        sm::float3 p = pos;
        if (wDown) p += fwd * speed * dt;
        if (sDown) p -= fwd * speed * dt;
        if (dDown) p += right * speed * dt;
        if (aDown) p -= right * speed * dt;
        if (eDown) p += up * speed * dt;
        if (qDown) p -= up * speed * dt;
        return p;
    }

    /// Compute initial yaw/pitch from a camera direction vector.
    static void directionToYawPitch(const sm::float3& dir, float& outYaw, float& outPitch) noexcept {
        const sm::float3 d = sm::normalize(dir);
        outPitch = sm::degrees(std::asin(d.y));
        outYaw = sm::degrees(std::atan2(d.z, d.x));
    }
};

} // namespace theia

#endif // THEIA_RENDERER_CAMERACONTROLLER_HPP
