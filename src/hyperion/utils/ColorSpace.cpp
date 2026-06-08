#include "hyperion/utils/ColorSpace.hpp"

#include <algorithm>
#include <cmath>

namespace ColorSpace {

// ── Rec.2020 ↔ Rec.709 ───────────────────────────────────────────────────────
// Matrices from ITU-R BT.2087 / IEC 61966-2-1
// clang-format off
static constexpr glm::mat3 kRec2020_to_Rec709 = glm::mat3(
    // column-major (GLM)    row 0               row 1               row 2
    glm::vec3( 1.6604910f, -0.1245505f, -0.0181508f),  // col 0 (from R)
    glm::vec3(-0.5876411f,  1.1328999f, -0.1005789f),  // col 1 (from G)
    glm::vec3(-0.0728499f, -0.0083494f,  1.1187297f)   // col 2 (from B)
);

static constexpr glm::mat3 kRec709_to_Rec2020 = glm::mat3(
    glm::vec3(0.6274039f,  0.0690973f,  0.0163914f),
    glm::vec3(0.3292830f,  0.9195404f,  0.0880133f),
    glm::vec3(0.0433131f,  0.0113623f,  0.8955953f)
);

// ── Rec.2020 ↔ CIE XYZ D65 ───────────────────────────────────────────────────
// From ITU-R BT.2020 (2015) Table 4
static constexpr glm::mat3 kRec2020_to_XYZ = glm::mat3(
    glm::vec3(0.6370f, 0.2627f, 0.0000f),
    glm::vec3(0.1446f, 0.6780f, 0.0281f),
    glm::vec3(0.1689f, 0.0593f, 1.0610f)
);

static constexpr glm::mat3 kXYZ_to_Rec2020 = glm::mat3(
    glm::vec3( 1.7166512f, -0.6666844f,  0.0176399f),
    glm::vec3(-0.3556708f,  1.6164812f, -0.0427706f),
    glm::vec3(-0.2533663f,  0.0157685f,  0.9421031f)
);

// ── Rec.2020 ↔ ACES AP1 (ACEScg) ─────────────────────────────────────────────
// Via XYZ: Rec.2020 → XYZ → AP1
static constexpr glm::mat3 kRec2020_to_AP1 = glm::mat3(
    glm::vec3( 0.6131324f,  0.0701243f,  0.0205076f),
    glm::vec3( 0.3395255f,  0.9163394f,  0.1096098f),
    glm::vec3( 0.0474491f,  0.0135363f,  0.8699926f)
);

static constexpr glm::mat3 kAP1_to_Rec2020 = glm::mat3(
    glm::vec3( 1.7048586f, -0.1300066f, -0.0240033f),
    glm::vec3(-0.6217882f,  1.1407579f, -0.1289716f),
    glm::vec3(-0.0832704f, -0.0107513f,  1.1529749f)
);
// clang-format on

glm::vec3 rec2020ToRec709(glm::vec3 c) noexcept {
    return kRec2020_to_Rec709 * c;
}
glm::vec3 rec709ToRec2020(glm::vec3 c) noexcept {
    return kRec709_to_Rec2020 * c;
}
glm::vec3 rec2020ToXyz(glm::vec3 c) noexcept {
    return kRec2020_to_XYZ * c;
}
glm::vec3 xyzToRec2020(glm::vec3 xyz) noexcept {
    return kXYZ_to_Rec2020 * xyz;
}
glm::vec3 rec2020ToAcesCg(glm::vec3 c) noexcept {
    return kRec2020_to_AP1 * c;
}
glm::vec3 acesCgToRec2020(glm::vec3 c) noexcept {
    return kAP1_to_Rec2020 * c;
}

// ── sRGB / Rec.709 OETF & EOTF ───────────────────────────────────────────────
namespace {
float linearToSrgbChannel(float v) noexcept {
    v = std::max(v, 0.f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
}
float srgbToLinearChannel(float v) noexcept {
    return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}
} // namespace

glm::vec3 linearRec709ToSrgb(glm::vec3 c) noexcept {
    return {linearToSrgbChannel(c.r), linearToSrgbChannel(c.g), linearToSrgbChannel(c.b)};
}
glm::vec3 srgbToLinearRec709(glm::vec3 c) noexcept {
    return {srgbToLinearChannel(c.r), srgbToLinearChannel(c.g), srgbToLinearChannel(c.b)};
}

// ── PQ (ST.2084) OETF & EOTF ─────────────────────────────────────────────────
// Constants per SMPTE ST.2084-2014
static constexpr float kPQ_m1 = 0.1593017578125f;
static constexpr float kPQ_m2 = 78.84375f;
static constexpr float kPQ_c1 = 0.8359375f;
static constexpr float kPQ_c2 = 18.8515625f;
static constexpr float kPQ_c3 = 18.6875f;

float pqOetf(float Yn) noexcept {
    Yn = std::clamp(Yn, 0.f, 1.f);
    const float Ym1 = std::pow(Yn, kPQ_m1);
    return std::pow((kPQ_c1 + kPQ_c2 * Ym1) / (1.f + kPQ_c3 * Ym1), kPQ_m2);
}
glm::vec3 pqOetf(glm::vec3 Yn) noexcept {
    return {pqOetf(Yn.r), pqOetf(Yn.g), pqOetf(Yn.b)};
}

float pqEotf(float E) noexcept {
    E = std::clamp(E, 0.f, 1.f);
    const float Em2 = std::pow(E, 1.f / kPQ_m2);
    return std::pow(std::max(Em2 - kPQ_c1, 0.f) / (kPQ_c2 - kPQ_c3 * Em2), 1.f / kPQ_m1);
}
glm::vec3 pqEotf(glm::vec3 E) noexcept {
    return {pqEotf(E.r), pqEotf(E.g), pqEotf(E.b)};
}

// ── HLG (BT.2100) OETF ───────────────────────────────────────────────────────
// Reference: ITU-R BT.2100-2 (2018), Table 5.
static constexpr float kHLG_a = 0.17883277f;
static constexpr float kHLG_b = 0.28466892f; // 1 - 4·a
static constexpr float kHLG_c = 0.55991073f; // 0.5 - a·ln(4·a)

float hlgOetf(float E) noexcept {
    E = std::max(E, 0.f);
    return (E <= 1.f / 12.f) ? std::sqrt(3.f * E) : (kHLG_a * std::log(12.f * E - kHLG_b) + kHLG_c);
}
glm::vec3 hlgOetf(glm::vec3 E) noexcept {
    return {hlgOetf(E.r), hlgOetf(E.g), hlgOetf(E.b)};
}

} // namespace ColorSpace
