#pragma once

// CPU-side tone mapping functions mirroring shaders/tonemap.slang.
// Used for unit testing and offline processing; keeps shader and CPU logic in sync.
//
// All functions operate in linear Rec.2020.  Apply color-space conversions
// (rec2020ToRec709, etc.) and OETFs separately via ColorSpace.hpp.
//
// References:
//   ACES colour matrices — AMPAS ACES specification S-2014-003
//     https://docs.acescentral.com/specifications/acescg/
//   ACES RRT+ODT analytic fit — Stephen Hill, SIGGRAPH 2016
//     "HDR Theory and Practice" course notes
//   Hable/Uncharted-2 filmic — John Hable, GDC 2010
//   Reinhard et al. — SIGGRAPH 2002, "Photographic Tone Reproduction for Digital Images"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

#include "hyperion/utils/ColorSpace.hpp"

namespace ToneMapping {

// ── ACES matrices ─────────────────────────────────────────────────────────────
// Column-major GLM form, matching ColorSpace.cpp kRec2020_to_AP1 / kAP1_to_Rec2020.
// Composed output matrices derived from those values (see tonemap.slang for row-major form).

// clang-format off
/// Rec.2020 → ACEScg AP1
inline constexpr glm::mat3 kRec2020ToAP1 = glm::mat3(
    glm::vec3(0.6131324f,  0.0701243f,  0.0205076f),  // col 0 (AP1 from R2020.R)
    glm::vec3(0.3395255f,  0.9163394f,  0.1096098f),  // col 1 (AP1 from R2020.G)
    glm::vec3(0.0474491f,  0.0135363f,  0.8699926f)   // col 2 (AP1 from R2020.B)
);

/// ACEScg AP1 → Rec.709 D65  (composed: AP1 → Rec.2020 → Rec.709)
inline constexpr glm::mat3 kAP1ToRec709 = glm::mat3(
    glm::vec3( 2.9090f, -0.3595f, -0.0447f),  // col 0
    glm::vec3(-1.6933f,  1.3712f, -0.2477f),  // col 1
    glm::vec3(-0.2159f, -0.0114f,  1.2925f)   // col 2
);

/// ACEScg AP1 → Display P3 D65  (composed: AP1 → Rec.2020 → P3)
inline constexpr glm::mat3 kAP1ToP3 = glm::mat3(
    glm::vec3( 2.1173f, -0.2071f, -0.0219f),  // col 0
    glm::vec3(-1.0180f,  1.2150f, -0.1540f),  // col 1
    glm::vec3(-0.0996f, -0.0077f,  1.1758f)   // col 2
);
// clang-format on

// ── ACES RRT+ODT analytic fit (Stephen Hill, SIGGRAPH 2016) ─────────────────

/// ACES RRT+ODT rational-polynomial approximation.
/// Input/output: ACEScg AP1 linear.
[[nodiscard]] inline glm::vec3 acesRrtOdtFit(glm::vec3 v) noexcept {
    const glm::vec3 a = v * (v + glm::vec3(0.0245786f)) - glm::vec3(0.000090537f);
    const glm::vec3 b = v * (0.983729f * v + glm::vec3(0.4329510f)) + glm::vec3(0.238081f);
    return a / b;
}

/// Full ACES tone mapper: Rec.2020 linear → tone-mapped in target primaries.
/// outputMat must be kAP1ToRec709 (SDR / scRGB) or kAP1ToP3 (Display P3).
[[nodiscard]] inline glm::vec3 acesFitted(glm::vec3 c2020, const glm::mat3& outputMat) noexcept {
    glm::vec3 ap1 = kRec2020ToAP1 * c2020;
    ap1 = glm::max(ap1, glm::vec3(0.f));
    ap1 = acesRrtOdtFit(ap1);
    return glm::max(outputMat * ap1, glm::vec3(0.f));
}

/// ACES for SDR (Rec.709 output).
[[nodiscard]] inline glm::vec3 acesFittedSDR(glm::vec3 c2020) noexcept {
    return acesFitted(c2020, kAP1ToRec709);
}

/// ACES for Display P3 output.
[[nodiscard]] inline glm::vec3 acesFittedP3(glm::vec3 c2020) noexcept {
    return acesFitted(c2020, kAP1ToP3);
}

// ── Hable / Uncharted-2 (John Hable, GDC 2010) ───────────────────────────────

[[nodiscard]] inline glm::vec3 hablePartial(glm::vec3 x) noexcept {
    constexpr float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
    return ((x * (A * x + glm::vec3(C * B)) + glm::vec3(D * E)) / (x * (A * x + glm::vec3(B)) + glm::vec3(D * F))) -
           glm::vec3(E / F);
}

/// Hable/Uncharted-2 filmic tone mapper.
/// Output is in Rec.2020; apply rec2020ToRec709 + sRGB OETF for SDR display.
[[nodiscard]] inline glm::vec3 hableFilmic(glm::vec3 x) noexcept {
    const glm::vec3 white = hablePartial(glm::vec3(11.2f));
    return hablePartial(2.0f * x) / white;
}

// ── Reinhard (Reinhard et al., SIGGRAPH 2002) ─────────────────────────────────

/// Luminance-preserving Reinhard tone mapper.  Tone-maps luminance, preserves chromaticity.
[[nodiscard]] inline glm::vec3 reinhardLuminance(glm::vec3 x) noexcept {
    const float lum = ColorSpace::luminance(x);
    if (lum <= 0.f)
        return glm::vec3(0.f);
    const float lumTm = lum / (1.f + lum);
    return x * (lumTm / lum);
}

} // namespace ToneMapping
