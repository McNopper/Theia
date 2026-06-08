#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <limits>
#include <numbers>

namespace Math {
inline constexpr float kPi = std::numbers::pi_v<float>;
inline constexpr float k2Pi = 2.0F * kPi;
inline constexpr float kInvPi = 1.0F / kPi;
inline constexpr float kInv2Pi = 1.0F / k2Pi;

[[nodiscard]] inline glm::mat4 makeRotationY(float radians) noexcept {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return glm::mat4(glm::vec4(c, 0.0F, -s, 0.0F),
                     glm::vec4(0.0F, 1.0F, 0.0F, 0.0F),
                     glm::vec4(s, 0.0F, c, 0.0F),
                     glm::vec4(0.0F, 0.0F, 0.0F, 1.0F));
}

[[nodiscard]] inline glm::vec3 safeDivide(glm::vec3 a, float b) noexcept {
    constexpr float kEpsilon = 1.0e-8F;
    if (std::abs(b) <= kEpsilon) {
        return glm::vec3(0.0F);
    }
    return a / b;
}

[[nodiscard]] inline float luminance(glm::vec3 c) noexcept {
    return (0.2627F * c.r) + (0.6780F * c.g) + (0.0593F * c.b);
}

[[nodiscard]] inline glm::vec3 srgbToLinear(glm::vec3 c) noexcept {
    const glm::vec3 clamped = glm::max(c, glm::vec3(0.0F));
    return glm::vec3(std::pow(clamped.x, 2.2F), std::pow(clamped.y, 2.2F), std::pow(clamped.z, 2.2F));
}

[[nodiscard]] inline bool isNanOrInf(glm::vec3 v) noexcept {
    return std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z) || std::isinf(v.x) || std::isinf(v.y) ||
           std::isinf(v.z);
}
} // namespace Math
