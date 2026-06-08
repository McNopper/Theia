#pragma once

// Hyperion ColorSpace utility — Rec.2020 is the single working color space.
// All internal calculations operate in linear Rec.2020.
// Conversions to/from other spaces are provided only for I/O boundaries.
//
// Reference primaries (ITU-R BT.2020):
//   R: (0.708, 0.292)  G: (0.170, 0.797)  B: (0.131, 0.046)  D65 white
//
// Luminance coefficients (BT.2020):  Kr=0.2627  Kg=0.6780  Kb=0.0593

#include <glm/glm.hpp>

namespace ColorSpace {

// ── Luminance (Rec.2020 relative, scene-linear) ─────────────────────────────
// Returns relative luminance Y in Rec.2020; input must be linear Rec.2020.
[[nodiscard]] inline float luminance(glm::vec3 linearRec2020) noexcept {
    return 0.2627f * linearRec2020.r + 0.6780f * linearRec2020.g + 0.0593f * linearRec2020.b;
}

// ── Rec.2020 ↔ Rec.709 (linear) ─────────────────────────────────────────────
// Use for display output only — never convert mid-pipeline.
[[nodiscard]] glm::vec3 rec2020ToRec709(glm::vec3 c) noexcept;
[[nodiscard]] glm::vec3 rec709ToRec2020(glm::vec3 c) noexcept;

// ── Rec.2020 ↔ CIE XYZ D65 ──────────────────────────────────────────────────
[[nodiscard]] glm::vec3 rec2020ToXyz(glm::vec3 c) noexcept;
[[nodiscard]] glm::vec3 xyzToRec2020(glm::vec3 xyz) noexcept;

// ── Rec.2020 ↔ ACES AP1 (ACEScg) ────────────────────────────────────────────
[[nodiscard]] glm::vec3 rec2020ToAcesCg(glm::vec3 c) noexcept;
[[nodiscard]] glm::vec3 acesCgToRec2020(glm::vec3 c) noexcept;

// ── sRGB / Rec.709 OETF & EOTF ──────────────────────────────────────────────
// Used at SDR display output: linearize asset textures on upload,
// or encode to sRGB at display boundary.
// Input/output: linear Rec.709 (NOT Rec.2020 — convert primaries first).
[[nodiscard]] glm::vec3 linearRec709ToSrgb(glm::vec3 linear) noexcept;
[[nodiscard]] glm::vec3 srgbToLinearRec709(glm::vec3 srgb) noexcept;

// Convenience: sRGB asset texture → linear Rec.2020 (two-step, used on upload)
[[nodiscard]] inline glm::vec3 srgbAssetToRec2020(glm::vec3 srgb) noexcept {
    return rec709ToRec2020(srgbToLinearRec709(srgb));
}

// ── Peak luminance constants ─────────────────────────────────────────────────
inline constexpr float kPeakLuminanceHDR10Nits = 10000.0f; // ST.2084 reference
inline constexpr float kPeakLuminanceSDRNits = 100.0f;     // sRGB / Rec.709

// ── PQ (ST.2084) OETF & EOTF ────────────────────────────────────────────────
// HDR10 display output. Input to pqOetf: absolute luminance in nits / 10000.
// i.e. normalise: Yn = clamp(nits / 10000, 0, 1) before calling.
[[nodiscard]] float pqOetf(float Yn) noexcept;
[[nodiscard]] glm::vec3 pqOetf(glm::vec3 Yn) noexcept;
[[nodiscard]] float pqEotf(float E) noexcept; // inverse — display→scene
[[nodiscard]] glm::vec3 pqEotf(glm::vec3 E) noexcept;

/// Convenience wrapper: absolute nits → PQ code value.
/// Equivalent to pqOetf(nits / kPeakLuminanceHDR10Nits).
[[nodiscard]] inline float pqOetfFromNits(float nits) noexcept {
    return pqOetf(nits / kPeakLuminanceHDR10Nits);
}
[[nodiscard]] inline glm::vec3 pqOetfFromNits(glm::vec3 nits) noexcept {
    return pqOetf(nits / kPeakLuminanceHDR10Nits);
}

// ── HLG (BT.2100) OETF ───────────────────────────────────────────────────────
// HLG display output. Input: scene-linear, 1.0 = HLG reference white.
// Constants per ITU-R BT.2100 Table 5: a = 0.17883277, b = 1-4a, c = 0.5-a*ln(4a).
[[nodiscard]] float hlgOetf(float E) noexcept;
[[nodiscard]] glm::vec3 hlgOetf(glm::vec3 E) noexcept;

// ── Exposure ─────────────────────────────────────────────────────────────────
// EV100 → linear exposure multiplier (UE5 / photographic convention).
// Apply to linear Rec.2020 scene radiance before tone mapping.
[[nodiscard]] inline float ev100ToExposure(float ev100) noexcept {
    // exposure = 1 / (1.2 * 2^EV100)
    return 1.0f / (1.2f * std::exp2(ev100));
}

} // namespace ColorSpace
