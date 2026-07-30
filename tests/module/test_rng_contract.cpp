#include <cstdint>
#include <gtest/gtest.h>

#include "harmonia/utils/Rng.hpp"

TEST(TheiaRngContract, PixelAndFrameProduceDifferentSamples) {
    std::uint32_t a = harmonia::Rng::composeSeed({10U, 20U}, 0U, 0U, 1337U);
    std::uint32_t b = harmonia::Rng::composeSeed({11U, 20U}, 0U, 0U, 1337U);
    std::uint32_t c = harmonia::Rng::composeSeed({10U, 20U}, 1U, 0U, 1337U);
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
}

TEST(TheiaRngContract, BounceAxisIsDecorrelated) {
    const std::uint32_t b0 = harmonia::Rng::composeSeed({3U, 4U}, 8U, 0U, 42U);
    const std::uint32_t b1 = harmonia::Rng::composeSeed({3U, 4U}, 8U, 1U, 42U);
    EXPECT_NE(b0, b1);
}

TEST(TheiaRngContract, DeterministicReplayReproducesSequence) {
    std::uint32_t s0 = harmonia::Rng::composeSeed({7U, 5U}, 2U, 0U, 777U);
    std::uint32_t s1 = harmonia::Rng::composeSeed({7U, 5U}, 2U, 0U, 777U);

    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(harmonia::Rng::nextFloat(s0), harmonia::Rng::nextFloat(s1));
    }
}
