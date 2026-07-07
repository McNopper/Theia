#pragma once

#include <slang-math/slang-math.hpp>

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
inline sm::float2 halton23(uint32_t sampleIndex) noexcept {
    const uint32_t seqIndex = sampleIndex + 1U; // avoid (0,0)
    return sm::float2{
        radicalInverse(seqIndex, 2U),
        radicalInverse(seqIndex, 3U),
    };
}

/// Sub-pixel jitter in pixel footprint units ([-0.5, 0.5) per axis).
inline sm::float2 cameraJitterPixels(uint32_t sampleIndex) noexcept {
    return halton23(sampleIndex) - sm::float2{0.5f, 0.5f};
}

/// Convert pixel-footprint jitter to NDC offsets.
inline sm::float2 cameraJitterNdc(uint32_t sampleIndex, uint32_t width, uint32_t height) noexcept {
    if (width == 0U || height == 0U) {
        return sm::float2{0.0f, 0.0f};
    }
    const sm::float2 px = cameraJitterPixels(sampleIndex);
    return sm::float2{
        (2.0f * px.x) / static_cast<float>(width),
        (2.0f * px.y) / static_cast<float>(height),
    };
}

/// Apply NDC jitter as a projection-center offset.
inline sm::float4x4 applyProjectionJitter(sm::float4x4 projection, const sm::float2& jitterNdc) noexcept {
    projection[0][2] += jitterNdc.x;
    projection[1][2] += jitterNdc.y;
    return projection;
}

} // namespace theia
