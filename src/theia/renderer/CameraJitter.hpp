#pragma once

#include <glm/glm.hpp>

#include <cstdint>

namespace theia {

/// Radical inverse in the given base for low-discrepancy sequences.
inline float radicalInverse(uint32_t index, uint32_t base) noexcept {
    const float invBase = 1.0f / static_cast<float>(base);
    float reversed = 0.0f;
    float place = invBase;
    while (index > 0U) {
        const uint32_t digit = index % base;
        reversed += static_cast<float>(digit) * place;
        index /= base;
        place *= invBase;
    }
    return reversed;
}

/// Deterministic Halton(2,3) sample in [0,1)^2 driven by sample index.
inline glm::vec2 halton23(uint32_t sampleIndex) noexcept {
    const uint32_t seqIndex = sampleIndex + 1U; // avoid (0,0)
    return glm::vec2{
        radicalInverse(seqIndex, 2U),
        radicalInverse(seqIndex, 3U),
    };
}

/// Sub-pixel jitter in pixel footprint units ([-0.5, 0.5) per axis).
inline glm::vec2 cameraJitterPixels(uint32_t sampleIndex) noexcept {
    return halton23(sampleIndex) - glm::vec2(0.5f);
}

/// Convert pixel-footprint jitter to NDC offsets.
inline glm::vec2 cameraJitterNdc(uint32_t sampleIndex, uint32_t width, uint32_t height) noexcept {
    if (width == 0U || height == 0U) {
        return glm::vec2(0.0f);
    }
    const glm::vec2 px = cameraJitterPixels(sampleIndex);
    return glm::vec2{
        (2.0f * px.x) / static_cast<float>(width),
        (2.0f * px.y) / static_cast<float>(height),
    };
}

/// Apply NDC jitter as a projection-center offset.
inline glm::mat4 applyProjectionJitter(glm::mat4 projection, const glm::vec2& jitterNdc) noexcept {
    projection[2][0] += jitterNdc.x;
    projection[2][1] += jitterNdc.y;
    return projection;
}

} // namespace theia
