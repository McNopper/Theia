#include <gtest/gtest.h>

#include <glm/glm.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

#include "harmonia/utils/Rng.hpp"

namespace {

struct MediumStack {
    std::array<float, 8> eta{};
    uint32_t depth = 0U;

    [[nodiscard]] float top() const { return depth > 0U ? eta[depth - 1U] : 1.0F; }
    [[nodiscard]] float belowTop() const { return depth > 1U ? eta[depth - 2U] : 1.0F; }
};

struct InterfaceStep {
    float etaI = 1.0F;
    float etaT = 1.0F;
};

InterfaceStep resolveInterface(MediumStack& stack, bool frontFace, bool thinWalled, float materialEta) {
    InterfaceStep out;
    out.etaI = stack.top();
    out.etaT = out.etaI;
    if (!thinWalled) {
        if (frontFace) {
            out.etaT = materialEta;
        } else {
            out.etaT = stack.belowTop();
        }
    }
    return out;
}

void commitTransmission(MediumStack& stack, bool frontFace, bool thinWalled, float materialEta) {
    if (thinWalled) {
        return;
    }
    if (frontFace) {
        if (stack.depth < stack.eta.size()) {
            stack.eta[stack.depth++] = materialEta;
        }
        return;
    }
    if (stack.depth > 0U) {
        --stack.depth;
    }
}

bool isTir(float etaI, float etaT, float cosThetaI) {
    const float ratio = etaI / std::max(etaT, 1.0e-6F);
    const float sin2ThetaT = ratio * ratio * (1.0F - cosThetaI * cosThetaI);
    return sin2ThetaT > 1.0F;
}

struct MultiBounceTrace {
    glm::vec3 throughput{1.0F};
    std::string branchBits;
};

MultiBounceTrace simulateTrace(uint32_t frameSampleIndex) {
    constexpr glm::vec3 kTransTint{0.94F, 0.90F, 0.86F};
    constexpr float kContinueWeight = 0.92F;
    constexpr float kFr = 0.08F;

    MultiBounceTrace out;
    for (uint32_t bounce = 0U; bounce < 8U; ++bounce) {
        uint32_t rngState = Rng::composeSeed({21U, 13U}, frameSampleIndex, bounce * 8U + 5U, 2026U);
        const bool reflect = Rng::nextFloat(rngState) < kFr;
        out.branchBits.push_back(reflect ? 'R' : 'T');

        if (reflect) {
            out.throughput *= kContinueWeight;
        } else {
            out.throughput *= kContinueWeight * kTransTint;
        }
    }
    return out;
}

} // namespace

TEST(TheiaTransparencyPathContract, StackedGlassUsesRelativeIorAcrossInterfaces) {
    MediumStack stack;

    const InterfaceStep enterOuter = resolveInterface(stack, true, false, 1.5F);
    EXPECT_NEAR(enterOuter.etaI, 1.0F, 1.0e-6F);
    EXPECT_NEAR(enterOuter.etaT, 1.5F, 1.0e-6F);
    commitTransmission(stack, true, false, 1.5F);
    EXPECT_EQ(stack.depth, 1U);

    const InterfaceStep enterInner = resolveInterface(stack, true, false, 1.3F);
    EXPECT_NEAR(enterInner.etaI, 1.5F, 1.0e-6F);
    EXPECT_NEAR(enterInner.etaT, 1.3F, 1.0e-6F);
    commitTransmission(stack, true, false, 1.3F);
    EXPECT_EQ(stack.depth, 2U);

    const InterfaceStep exitInner = resolveInterface(stack, false, false, 1.3F);
    EXPECT_NEAR(exitInner.etaI, 1.3F, 1.0e-6F);
    EXPECT_NEAR(exitInner.etaT, 1.5F, 1.0e-6F);
    commitTransmission(stack, false, false, 1.3F);
    EXPECT_EQ(stack.depth, 1U);

    const InterfaceStep exitOuter = resolveInterface(stack, false, false, 1.5F);
    EXPECT_NEAR(exitOuter.etaI, 1.5F, 1.0e-6F);
    EXPECT_NEAR(exitOuter.etaT, 1.0F, 1.0e-6F);
    commitTransmission(stack, false, false, 1.5F);
    EXPECT_EQ(stack.depth, 0U);
}

TEST(TheiaTransparencyPathContract, TirDetectionIsConsistent) {
    EXPECT_TRUE(isTir(1.5F, 1.0F, 0.5F));
    EXPECT_FALSE(isTir(1.0F, 1.5F, 0.5F));
    EXPECT_FALSE(isTir(1.5F, 1.0F, 0.95F));
}

TEST(TheiaTransparencyPathContract, DeterministicReplayKeepsMultiBounceSequenceStable) {
    const MultiBounceTrace a = simulateTrace(11U);
    const MultiBounceTrace b = simulateTrace(11U);
    const MultiBounceTrace c = simulateTrace(12U);

    EXPECT_EQ(a.branchBits, b.branchBits);
    EXPECT_NEAR(a.throughput.x, b.throughput.x, 1.0e-7F);
    EXPECT_NEAR(a.throughput.y, b.throughput.y, 1.0e-7F);
    EXPECT_NEAR(a.throughput.z, b.throughput.z, 1.0e-7F);

    EXPECT_NE(a.branchBits, c.branchBits);
}
