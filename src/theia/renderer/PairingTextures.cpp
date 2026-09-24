#include "theia/renderer/PairingTextures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>

namespace theia::pairing {
namespace {

/// Build one pairing texture as a fixed-point-free involution over S*S texels
/// whose pair displacements follow an isotropic Gaussian (kSigma per axis).
/// Greedy construction in random texel order: pair each still-free texel with a
/// free target drawn around it (wrapping, for tiling); redraw on collision, then
/// spiral outward as a total fallback. The result is symmetric by construction
/// (partner[partner[p]] == p), which is what makes neighbour selection reciprocal.
[[nodiscard]] std::vector<std::int32_t> buildInvolution(std::uint32_t s, std::mt19937_64& rng) {
    const auto wrap = [s](std::int32_t v) {
        return ((v % static_cast<std::int32_t>(s)) + static_cast<std::int32_t>(s)) % static_cast<std::int32_t>(s);
    };
    const std::int32_t n = static_cast<std::int32_t>(s * s);
    std::vector<std::int32_t> partner(n, -1);
    std::vector<std::int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::normal_distribution<float> gauss(0.0F, kSigma);

    for (std::int32_t p : order) {
        if (partner[p] >= 0) {
            continue;
        }
        const std::int32_t px = p % static_cast<std::int32_t>(s);
        const std::int32_t py = p / static_cast<std::int32_t>(s);
        std::int32_t q = -1;
        for (int attempt = 0; attempt < 32 && q < 0; ++attempt) {
            const std::int32_t qx = wrap(px + static_cast<std::int32_t>(std::lround(gauss(rng))));
            const std::int32_t qy = wrap(py + static_cast<std::int32_t>(std::lround(gauss(rng))));
            const std::int32_t cand = qy * static_cast<std::int32_t>(s) + qx;
            if (cand != p && partner[cand] < 0) {
                q = cand;
            }
        }
        if (q < 0) {
            // Spiral outward for the nearest free texel — keeps the involution total
            // (reachable whenever a free partner exists; n is even by construction).
            const std::int32_t si = static_cast<std::int32_t>(s);
            for (std::int32_t r = 1; r < si && q < 0; ++r) {
                for (std::int32_t dy = -r; dy <= r && q < 0; ++dy) {
                    for (std::int32_t dx = -r; dx <= r; ++dx) {
                        if (std::max(std::abs(dx), std::abs(dy)) != r) {
                            continue;
                        }
                        const std::int32_t cand = wrap(py + dy) * si + wrap(px + dx);
                        if (cand != p && partner[cand] < 0) {
                            q = cand;
                            break;
                        }
                    }
                }
            }
        }
        if (q < 0) {
            continue; // unreachable with even n; leave p unpaired rather than corrupt
        }
        partner[p] = q;
        partner[q] = p;
    }
    return partner;
}

} // namespace

std::vector<std::uint32_t> buildPackedOffsets(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint32_t> packed;
    packed.reserve(kTotalTexels);
    for (std::uint32_t t = 0; t < kTextureCount; ++t) {
        const std::int32_t s = static_cast<std::int32_t>(kTextureSizes[t]);
        const std::vector<std::int32_t> partner = buildInvolution(kTextureSizes[t], rng);
        for (std::int32_t p = 0; p < s * s; ++p) {
            const std::int32_t q = partner[p];
            const std::int32_t qx = q % s;
            const std::int32_t qy = q / s;
            // Offset to the partner, wrapped into the tileable range [-s/2, s/2).
            auto wrapDelta = [s](std::int32_t d) {
                d = ((d % s) + s + s / 2) % s - s / 2;
                return d;
            };
            const std::int32_t dx = wrapDelta(qx - (p % s));
            const std::int32_t dy = wrapDelta(qy - (p / s));
            const auto ux = static_cast<std::uint32_t>(static_cast<std::uint16_t>(static_cast<std::int16_t>(dx)));
            const auto uy = static_cast<std::uint32_t>(static_cast<std::uint16_t>(static_cast<std::int16_t>(dy)));
            packed.push_back(ux | (uy << 16U));
        }
    }
    return packed;
}

} // namespace theia::pairing
