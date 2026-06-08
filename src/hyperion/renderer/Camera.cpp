#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "hyperion/renderer/Camera.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

float Camera::PhysicalCamera::ev100() const noexcept {
    // EV100 = log2(N² × t_inv × 100 / ISO)
    return std::log2f((aperture * aperture) * shutterSpeedHz * 100.0f / iso);
}

float Camera::PhysicalCamera::exposure() const noexcept {
    // Standard photographic exposure: 1 / (1.2 × 2^EV100)
    return 1.0f / (1.2f * std::exp2f(ev100()));
}

Camera::Camera() noexcept : Camera(Params{}) {}

Camera::Camera(Params params) noexcept : m_params(params) {
    if (m_params.focusDist <= 0.0f) {
        m_params.focusDist = glm::length(m_params.target - m_params.position);
    }
}

void Camera::setAspect(float aspect) noexcept {
    m_params.aspectRatio = std::max(aspect, 0.001f);
}

void Camera::setPosition(glm::vec3 pos) noexcept {
    m_params.position = pos;
    if (m_params.focusDist <= 0.0f) {
        m_params.focusDist = glm::length(m_params.target - m_params.position);
    }
}

void Camera::setTarget(glm::vec3 target) noexcept {
    m_params.target = target;
    if (m_params.focusDist <= 0.0f) {
        m_params.focusDist = glm::length(m_params.target - m_params.position);
    }
}

void Camera::setPhysicalCamera(PhysicalCamera physical) noexcept {
    m_params.physical = physical;
}

CameraData Camera::getCameraData(uint32_t frameIndex, uint32_t maxDepth) const noexcept {
    const glm::mat4 view = viewMatrix();
    const glm::mat4 proj = projectionMatrix();

    return CameraData{
        .invView = glm::inverse(view),
        .invProj = glm::inverse(proj),
        .position = glm::vec4(m_params.position, 1.0f),
        .lensRadius = std::max(m_params.lensRadius, 0.0f),
        .focusDistance =
            m_params.focusDist > 0.0f ? m_params.focusDist : glm::length(m_params.target - m_params.position),
        .frameIndex = frameIndex,
        .maxDepth = maxDepth,
        .exposure = m_params.physical.exposure(),
        ._padCam = {},
    };
}

glm::mat4 Camera::viewMatrix() const noexcept {
    return glm::lookAtRH(m_params.position, m_params.target, m_params.up);
}

glm::mat4 Camera::projectionMatrix() const noexcept {
    return glm::perspectiveRH_ZO(glm::radians(m_params.vfovDeg),
                                 std::max(m_params.aspectRatio, 0.001f),
                                 std::max(m_params.nearPlane, 0.001f),
                                 std::max(m_params.farPlane, m_params.nearPlane + 0.001f));
}
