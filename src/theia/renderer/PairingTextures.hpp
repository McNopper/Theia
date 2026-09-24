#ifndef THEIA_RENDERER_PAIRINGTEXTURES_HPP
#define THEIA_RENDERER_PAIRINGTEXTURES_HPP

#include <cstdint>
#include <vector>

namespace theia {

// GI-ENH (ReSTIR PT Enhanced §3, Lin/Kettunen/Wyman — I3D 2026): self-inverse
// pairing textures for Gaussian paired-neighbor selection in the path reservoir's
// spatial reuse. Three tiling offset maps (254/230/210 texels; deliberately
// different sizes so their periods do not beat against each other within the
// screen) whose pair displacements follow an isotropic Gaussian with per-axis
// sigma 16 px (matched to the uniform-disk radius 30 px of the previous random
// draws: sigma = sqrt(8/(9*pi)) * R). Texel (x, y) holds the offset to its unique
// partner, so selecting neighbour A of B implies selecting B of A — the pairing
// property the paper's shift-sharing upgrade needs (a plain quality upgrade for
// this repo's one-replay-per-neighbour scheme).
//
// The GPU applies a per-frame flip/mirror/transpose/offset to the static maps
// (paper §3.2) so the tiling never bakes a fixed spatial pattern into the
// accumulated image. Offsets pack one texel per uint32 (x = low int16,
// y = high int16, two's complement).

namespace pairing {

inline constexpr std::uint32_t kTextureCount = 3;
inline constexpr std::uint32_t kTextureSizes[kTextureCount] = {254U, 230U, 210U};
inline constexpr float kSigma = 16.0F;            ///< per-axis pair-displacement std (px)
inline constexpr std::uint32_t kPackedBits = 19U; ///< per-frame perm: 3 transform + 8 + 8 offset

/// Total texels across the concatenated maps (the shader derives the segment
/// bases from kTextureSizes).
inline constexpr std::uint32_t kTotalTexels = 254U * 254U + 230U * 230U + 210U * 210U;

/// Build the concatenated packed offset maps. Deterministic for a given seed, so
/// runs stay reproducible. The result is one uint32 per texel, concatenated in
/// texture order 0..kTextureCount-1.
[[nodiscard]] std::vector<std::uint32_t> buildPackedOffsets(std::uint64_t seed);

} // namespace pairing
} // namespace theia

#endif // THEIA_RENDERER_PAIRINGTEXTURES_HPP
