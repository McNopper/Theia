#include <cstdint>
#include <gtest/gtest.h>

#include "harmonia/utils/Rng.hpp"

TEST(TheiaRngContract, PixelAndFrameProduceDifferentSamples) {
    std::uint32_t a = Rng::composeSeed({10U, 20U}, 0U, 0U, 1337U);
    std::uint32_t b = Rng::composeSeed({11U, 20U}, 0U, 0U, 1337U);
    std::uint32_t c = Rng::composeSeed({10U, 20U}, 1U, 0U, 1337U);
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
}

TEST(TheiaRngContract, BounceAxisIsDecorrelated) {
    const std::uint32_t b0 = Rng::composeSeed({3U, 4U}, 8U, 0U, 42U);
    const std::uint32_t b1 = Rng::composeSeed({3U, 4U}, 8U, 1U, 42U);
    EXPECT_NE(b0, b1);
}

TEST(TheiaRngContract, DeterministicReplayReproducesSequence) {
    std::uint32_t s0 = Rng::composeSeed({7U, 5U}, 2U, 0U, 777U);
    std::uint32_t s1 = Rng::composeSeed({7U, 5U}, 2U, 0U, 777U);

    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(Rng::nextFloat(s0), Rng::nextFloat(s1));
    }
}
