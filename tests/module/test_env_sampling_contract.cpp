#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <slang-math/slang-math.hpp>
#include <vector>

#include "harmonia/utils/Rng.hpp"

namespace {

constexpr float kPi = 3.14159265359F;
constexpr float kInvPi = 0.31830988618F;

uint32_t sampleCdf1D(const std::vector<float>& cdf, uint32_t base, uint32_t n, float r) {
    uint32_t lo = 0U;
    uint32_t hi = n - 1U;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1U;
        if (cdf[base + mid + 1U] <= r) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}

sm::float3 sampleEnvImportanceDir(const std::vector<float>& marginalCdf,
                                  const std::vector<float>& conditionalCdf,
                                  uint32_t W,
                                  uint32_t H,
                                  sm::float2 xi,
                                  float& pdfOmega) {
    const uint32_t row = sampleCdf1D(marginalCdf, 0U, H, xi.x);
    const uint32_t col = sampleCdf1D(conditionalCdf, row * (W + 1U), W, xi.y);

    const float rowLo = marginalCdf[row];
    const float rowHi = marginalCdf[row + 1U];
    const float tRow = (rowHi > rowLo) ? std::clamp((xi.x - rowLo) / (rowHi - rowLo), 0.0F, 1.0F) : 0.5F;

    const uint32_t cBase = row * (W + 1U);
    const float colLo = conditionalCdf[cBase + col];
    const float colHi = conditionalCdf[cBase + col + 1U];
    const float tCol = (colHi > colLo) ? std::clamp((xi.y - colLo) / (colHi - colLo), 0.0F, 1.0F) : 0.5F;

    const float uFrac = (static_cast<float>(col) + tCol) / static_cast<float>(W);
    const float vFrac = (static_cast<float>(row) + tRow) / static_cast<float>(H);
    const float pRow = rowHi - rowLo;
    const float pCol = colHi - colLo;
    const float theta = kPi * vFrac;
    const float sinTheta = std::max(std::sin(theta), 1.0e-5F);
    pdfOmega = (pRow * pCol * static_cast<float>(W) * static_cast<float>(H)) / (2.0F * kPi * kPi * sinTheta);

    const float phi = 2.0F * kPi * (uFrac - 0.5F);
    sm::float3 dir;
    dir.y = std::cos(theta);
    dir.x = sinTheta * std::cos(phi);
    dir.z = sinTheta * std::sin(phi);
    return sm::normalize(dir);
}

float evalEnvImportancePdfDir(const std::vector<float>& marginalCdf,
                              const std::vector<float>& conditionalCdf,
                              uint32_t W,
                              uint32_t H,
                              sm::float3 dir) {
    const sm::float3 d = sm::normalize(dir);
    const float u = std::atan2(d.z, d.x) * (0.5F * kInvPi) + 0.5F;
    const float v = std::acos(std::clamp(d.y, -1.0F, 1.0F)) * kInvPi;

    const uint32_t col = std::min(static_cast<uint32_t>(u * static_cast<float>(W)), W - 1U);
    const uint32_t row = std::min(static_cast<uint32_t>(v * static_cast<float>(H)), H - 1U);

    const float pRow = marginalCdf[row + 1U] - marginalCdf[row];
    const float pCol = conditionalCdf[row * (W + 1U) + col + 1U] - conditionalCdf[row * (W + 1U) + col];
    const float theta = kPi * v; // exact theta from direction, not bin centre
    const float sinTheta = std::max(std::sin(theta), 1.0e-5F);
    return (pRow * pCol * static_cast<float>(W) * static_cast<float>(H)) / (2.0F * kPi * kPi * sinTheta);
}

float balanceHeuristic(float pA, float pB) {
    const float a = std::isfinite(pA) ? std::max(pA, 0.0F) : 0.0F;
    const float b = std::isfinite(pB) ? std::max(pB, 0.0F) : 0.0F;
    const float d = a + b;
    return d > 0.0F ? a / d : 0.0F;
}

} // namespace

TEST(TheiaEnvSamplingContract, DeterministicReplayProducesStableSequence) {
    constexpr uint32_t W = 4U;
    constexpr uint32_t H = 2U;
    const std::vector<float> marginal{0.0F, 0.5F, 1.0F};
    const std::vector<float> conditional{
        0.0F,
        0.25F,
        0.5F,
        0.75F,
        1.0F,
        0.0F,
        0.25F,
        0.5F,
        0.75F,
        1.0F,
    };

    uint32_t s0 = Rng::composeSeed({19U, 7U}, 3U, 2U, 123U);
    uint32_t s1 = Rng::composeSeed({19U, 7U}, 3U, 2U, 123U);

    for (int i = 0; i < 32; ++i) {
        float p0 = 0.0F;
        float p1 = 0.0F;
        const sm::float3 d0 = sampleEnvImportanceDir(marginal, conditional, W, H, Rng::nextFloat2(s0), p0);
        const sm::float3 d1 = sampleEnvImportanceDir(marginal, conditional, W, H, Rng::nextFloat2(s1), p1);
        EXPECT_NEAR(d0.x, d1.x, 1.0e-6F);
        EXPECT_NEAR(d0.y, d1.y, 1.0e-6F);
        EXPECT_NEAR(d0.z, d1.z, 1.0e-6F);
        EXPECT_NEAR(p0, p1, 1.0e-6F);
        EXPECT_TRUE(std::isfinite(p0));
        EXPECT_GT(p0, 0.0F);
    }
}

TEST(TheiaEnvSamplingContract, PdfEvaluationMatchesSampledDirections) {
    constexpr uint32_t W = 8U;
    constexpr uint32_t H = 4U;
    const std::vector<float> marginal{0.0F, 0.1F, 0.35F, 0.7F, 1.0F};
    std::vector<float> conditional;
    conditional.reserve(H * (W + 1U));
    for (uint32_t row = 0U; row < H; ++row) {
        conditional.push_back(0.0F);
        for (uint32_t col = 0U; col < W; ++col) {
            conditional.push_back(static_cast<float>(col + 1U) / static_cast<float>(W));
        }
    }

    uint32_t state = Rng::composeSeed({3U, 5U}, 2U, 1U, 77U);
    for (int i = 0; i < 64; ++i) {
        float sampledPdf = 0.0F;
        const sm::float3 dir = sampleEnvImportanceDir(marginal, conditional, W, H, Rng::nextFloat2(state), sampledPdf);
        const float evalPdf = evalEnvImportancePdfDir(marginal, conditional, W, H, dir);
        EXPECT_TRUE(std::isfinite(sampledPdf));
        EXPECT_TRUE(std::isfinite(evalPdf));
        EXPECT_GT(sampledPdf, 0.0F);
        EXPECT_GT(evalPdf, 0.0F);
        const float ratio = evalPdf / sampledPdf;
        EXPECT_GT(ratio, 0.1F);
        EXPECT_LT(ratio, 10.0F);
    }
}

TEST(TheiaEnvSamplingContract, BalanceHeuristicIsBoundedAndSymmetric) {
    const float a = balanceHeuristic(0.8F, 0.2F);
    const float b = balanceHeuristic(0.2F, 0.8F);
    EXPECT_GE(a, 0.0F);
    EXPECT_LE(a, 1.0F);
    EXPECT_GE(b, 0.0F);
    EXPECT_LE(b, 1.0F);
    EXPECT_NEAR(a + b, 1.0F, 1.0e-6F);
    EXPECT_FLOAT_EQ(balanceHeuristic(0.0F, 0.0F), 0.0F);
}
