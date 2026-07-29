#include <cstdint>
#include <gtest/gtest.h>
#include <slang-math/slang-math.hpp>

#include "theia/renderer/CameraJitter.hpp"

TEST(TheiaCameraJitterContract, Halton23SequenceIsDeterministicBySampleIndex) {
    const sm::float2 s0 = theia::cameraJitterPixels(0U);
    const sm::float2 s1 = theia::cameraJitterPixels(1U);
    const sm::float2 s2 = theia::cameraJitterPixels(2U);

    EXPECT_NEAR(s0.x, 0.0f, 1.0e-6f);
    EXPECT_NEAR(s0.y, -1.0f / 6.0f, 1.0e-6f);
    EXPECT_NEAR(s1.x, -0.25f, 1.0e-6f);
    EXPECT_NEAR(s1.y, 1.0f / 6.0f, 1.0e-6f);
    EXPECT_NEAR(s2.x, 0.25f, 1.0e-6f);
    EXPECT_NEAR(s2.y, -7.0f / 18.0f, 1.0e-6f);

    EXPECT_EQ(theia::cameraJitterPixels(17U), theia::cameraJitterPixels(17U));
}

TEST(TheiaCameraJitterContract, JitterStaysWithinSubPixelFootprint) {
    for (std::uint32_t i = 0U; i < 1024U; ++i) {
        const sm::float2 jitter = theia::cameraJitterPixels(i);
        EXPECT_GE(jitter.x, -0.5f);
        EXPECT_LT(jitter.x, 0.5f);
        EXPECT_GE(jitter.y, -0.5f);
        EXPECT_LT(jitter.y, 0.5f);
    }
}

TEST(TheiaCameraJitterContract, ProjectionJitterOnlyShiftsProjectionCenter) {
    sm::float4x4 proj = sm::perspective(sm::radians(45.0f), 320.0f / 240.0f, 0.1f, 1000.0f);
    proj[1][1] *= -1.0f;

    const sm::float2 jitterNdc = theia::cameraJitterNdc(7U, 320U, 240U);
    const sm::float4x4 jittered = theia::applyProjectionJitter(proj, jitterNdc);

    // In row-major slang-math, m[row][col]. applyProjectionJitter modifies [0][2] and [1][2].
    for (std::int32_t row = 0; row < 4; ++row) {
        for (std::int32_t col = 0; col < 4; ++col) {
            if (col == 2 && (row == 0 || row == 1)) {
                continue;
            }
            EXPECT_NEAR(jittered[row][col], proj[row][col], 1.0e-6f);
        }
    }
    EXPECT_NEAR(jittered[0][2], proj[0][2] + jitterNdc.x, 1.0e-6f);
    EXPECT_NEAR(jittered[1][2], proj[1][2] + jitterNdc.y, 1.0e-6f);

    const sm::float4x4 unchanged = theia::applyProjectionJitter(proj, sm::float2(0.0f));
    for (std::int32_t row = 0; row < 4; ++row) {
        for (std::int32_t col = 0; col < 4; ++col) {
            EXPECT_NEAR(unchanged[row][col], proj[row][col], 1.0e-6f);
        }
    }
}
