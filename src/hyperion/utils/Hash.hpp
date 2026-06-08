#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

#include "hyperion/GpuTypes.hpp"

namespace detail {
[[nodiscard]] inline uint64_t wyhashFoldedMultiply(uint64_t a, uint64_t b) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    uint64_t high = 0U;
    const uint64_t low = _umul128(a, b, &high);
    return low ^ high;
#elif defined(__SIZEOF_INT128__)
    const auto wide = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(wide) ^ static_cast<uint64_t>(wide >> 64U);
#else
    const uint64_t aLow = static_cast<uint32_t>(a);
    const uint64_t aHigh = a >> 32U;
    const uint64_t bLow = static_cast<uint32_t>(b);
    const uint64_t bHigh = b >> 32U;

    const uint64_t lowLow = aLow * bLow;
    const uint64_t lowHigh = aLow * bHigh;
    const uint64_t highLow = aHigh * bLow;
    const uint64_t highHigh = aHigh * bHigh;

    const uint64_t carry = ((lowLow >> 32U) + static_cast<uint32_t>(lowHigh) + static_cast<uint32_t>(highLow)) >> 32U;
    const uint64_t high = highHigh + (lowHigh >> 32U) + (highLow >> 32U) + carry;
    const uint64_t low = lowLow + (lowHigh << 32U) + (highLow << 32U);
    return low ^ high;
#endif
}

[[nodiscard]] inline uint64_t load64(const unsigned char* data) noexcept {
    uint64_t value = 0U;
    std::memcpy(&value, data, sizeof(value));
    return value;
}
} // namespace detail

struct VertexHash {
    [[nodiscard]] size_t operator()(const GpuVertex& v) const noexcept {
        constexpr uint64_t kSecret0 = 0xA0761D6478BD642FULL;
        constexpr uint64_t kSecret1 = 0xE7037ED1A0B428DBULL;
        constexpr uint64_t kSecret2 = 0x8EBC6AF09C88C6E3ULL;
        constexpr uint64_t kSecret3 = 0x589965CC75374CC3ULL;

        const auto* bytes = reinterpret_cast<const unsigned char*>(&v);
        uint64_t seed = kSecret0 ^ sizeof(GpuVertex);

        for (size_t offset = 0; offset < sizeof(GpuVertex); offset += sizeof(uint64_t)) {
            const uint64_t chunk = detail::load64(bytes + offset);
            seed = detail::wyhashFoldedMultiply(chunk ^ kSecret1, seed ^ kSecret2) ^ kSecret3;
        }

        return static_cast<size_t>(detail::wyhashFoldedMultiply(seed ^ kSecret0, sizeof(GpuVertex) ^ kSecret1));
    }
};

struct VertexEqual {
    [[nodiscard]] bool operator()(const GpuVertex& a, const GpuVertex& b) const noexcept {
        return std::memcmp(&a, &b, sizeof(GpuVertex)) == 0;
    }
};
