#pragma once

#include <glm/glm.hpp>

#include <cstdint>

struct Pcg32 {
    uint64_t state = 0U;
    uint64_t inc = 0U;

    explicit Pcg32(uint64_t seed, uint64_t seq = 1U) noexcept : state(0U), inc((seq << 1U) | 1U) {
        static_cast<void>(next());
        state += seed;
        static_cast<void>(next());
    }

    [[nodiscard]] uint32_t next() noexcept {
        const uint64_t oldState = state;
        state = (oldState * 6364136223846793005ULL) + inc;

        const uint32_t xorshifted = static_cast<uint32_t>(((oldState >> 18U) ^ oldState) >> 27U);
        const uint32_t rot = static_cast<uint32_t>(oldState >> 59U);
        return (xorshifted >> rot) | (xorshifted << ((32U - rot) & 31U));
    }

    [[nodiscard]] float nextFloat() noexcept {
        constexpr float kScale = 1.0F / 4294967296.0F;
        return static_cast<float>(next()) * kScale;
    }

    [[nodiscard]] glm::vec2 nextVec2() noexcept { return glm::vec2(nextFloat(), nextFloat()); }
};

[[nodiscard]] inline float halton(uint32_t index, uint32_t base) noexcept {
    if (base < 2U) {
        return 0.0F;
    }

    float result = 0.0F;
    float fraction = 1.0F / static_cast<float>(base);
    uint32_t current = index;

    while (current > 0U) {
        result += fraction * static_cast<float>(current % base);
        current /= base;
        fraction /= static_cast<float>(base);
    }

    return result;
}

[[nodiscard]] inline glm::vec2 halton2D(uint32_t index) noexcept {
    return glm::vec2(halton(index, 2U), halton(index, 3U));
}
