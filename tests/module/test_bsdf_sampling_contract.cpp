#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

#include "harmonia/utils/Rng.hpp"

namespace {

float fresnelDielectric(float cosThetaI, float eta) {
    const float c = std::clamp(std::abs(cosThetaI), 0.0F, 1.0F);
    const float g2 = eta * eta - 1.0F + c * c;
    if (g2 <= 0.0F) {
        return 1.0F;
    }
    const float g = std::sqrt(g2);
    const float gp = g + c;
    const float gm = g - c;
    if (gp <= 0.0F || gm <= 0.0F) {
        return 1.0F;
    }
    const float a = gm / gp;
    const float bNum = c * gp - 1.0F;
    const float bDen = c * gm + 1.0F;
    if (bDen == 0.0F) {
        return 1.0F;
    }
    const float b = bNum / bDen;
    return std::clamp(0.5F * a * a * (1.0F + b * b), 0.0F, 1.0F);
}

} // namespace

TEST(TheiaBsdfSamplingContract, FresnelBranchProbabilityTracksExpectation) {
    constexpr float cosTheta = 0.5F;
    constexpr float eta = 1.5F;
    const float expectedF = fresnelDielectric(cosTheta, eta);

    std::uint32_t state = harmonia::Rng::composeSeed({23U, 11U}, 7U, 0U, 2026U);
    constexpr std::size_t kSamples = 100000;
    std::size_t reflectCount = 0;
    for (std::size_t i = 0; i < kSamples; ++i) {
        if (harmonia::Rng::nextFloat(state) < expectedF) {
            ++reflectCount;
        }
    }

    const float observed = static_cast<float>(reflectCount) / static_cast<float>(kSamples);
    EXPECT_NEAR(observed, expectedF, 0.01F);
}

TEST(TheiaBsdfSamplingContract, DeterministicReplayKeepsFresnelBranchSequence) {
    constexpr float cosTheta = 0.35F;
    constexpr float eta = 1.45F;
    const float expectedF = fresnelDielectric(cosTheta, eta);

    std::uint32_t s0 = harmonia::Rng::composeSeed({9U, 4U}, 2U, 3U, 77U);
    std::uint32_t s1 = harmonia::Rng::composeSeed({9U, 4U}, 2U, 3U, 77U);

    for (std::size_t i = 0; i < 256; ++i) {
        const bool b0 = harmonia::Rng::nextFloat(s0) < expectedF;
        const bool b1 = harmonia::Rng::nextFloat(s1) < expectedF;
        EXPECT_EQ(b0, b1);
    }
}

TEST(TheiaBsdfSamplingContract, FresnelStaysFiniteAcrossAnglesAndIor) {
    for (float eta : {1.01F, 1.1F, 1.5F, 1.8F, 2.4F}) {
        for (float c : {0.0F, 0.05F, 0.25F, 0.5F, 0.75F, 0.95F, 1.0F}) {
            const float f = fresnelDielectric(c, eta);
            EXPECT_TRUE(std::isfinite(f));
            EXPECT_GE(f, 0.0F);
            EXPECT_LE(f, 1.0F);
        }
    }
}
