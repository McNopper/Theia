#include <gtest/gtest.h>

#include <glm/gtc/matrix_transform.hpp>

#include "theia/renderer/CameraJitter.hpp"

TEST(TheiaCameraJitterContract, Halton23SequenceIsDeterministicBySampleIndex) {
    const glm::vec2 s0 = theia::cameraJitterPixels(0U);
    const glm::vec2 s1 = theia::cameraJitterPixels(1U);
    const glm::vec2 s2 = theia::cameraJitterPixels(2U);

    EXPECT_NEAR(s0.x, 0.0f, 1.0e-6f);
    EXPECT_NEAR(s0.y, -1.0f / 6.0f, 1.0e-6f);
    EXPECT_NEAR(s1.x, -0.25f, 1.0e-6f);
    EXPECT_NEAR(s1.y, 1.0f / 6.0f, 1.0e-6f);
    EXPECT_NEAR(s2.x, 0.25f, 1.0e-6f);
    EXPECT_NEAR(s2.y, -7.0f / 18.0f, 1.0e-6f);

    EXPECT_EQ(theia::cameraJitterPixels(17U), theia::cameraJitterPixels(17U));
}

TEST(TheiaCameraJitterContract, JitterStaysWithinSubPixelFootprint) {
    for (uint32_t i = 0U; i < 1024U; ++i) {
        const glm::vec2 jitter = theia::cameraJitterPixels(i);
        EXPECT_GE(jitter.x, -0.5f);
        EXPECT_LT(jitter.x, 0.5f);
        EXPECT_GE(jitter.y, -0.5f);
        EXPECT_LT(jitter.y, 0.5f);
    }
}

TEST(TheiaCameraJitterContract, ProjectionJitterOnlyShiftsProjectionCenter) {
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), 320.0f / 240.0f, 0.1f, 1000.0f);
    proj[1][1] *= -1.0f;

    const glm::vec2 jitterNdc = theia::cameraJitterNdc(7U, 320U, 240U);
    const glm::mat4 jittered = theia::applyProjectionJitter(proj, jitterNdc);

    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            if (c == 2 && (r == 0 || r == 1)) {
                continue;
            }
            EXPECT_NEAR(jittered[c][r], proj[c][r], 1.0e-6f);
        }
    }
    EXPECT_NEAR(jittered[2][0], proj[2][0] + jitterNdc.x, 1.0e-6f);
    EXPECT_NEAR(jittered[2][1], proj[2][1] + jitterNdc.y, 1.0e-6f);

    const glm::mat4 unchanged = theia::applyProjectionJitter(proj, glm::vec2(0.0f));
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(unchanged[c][r], proj[c][r], 1.0e-6f);
        }
    }
}
