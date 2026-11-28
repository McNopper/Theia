// CPU-contract tests for the shared CLI flag resolution Theia's main() relies on.
// I6: --frames-per-flip is parsed by the SHARED parser (harmonia::app::CliParser,
// inside Theia's applyCommonArg delegation loop in src/demo/main.cpp) into
// harmonia::app::App::Config. These tests pin that consumer contract - the flag
// resolves, defaults to the classic cadence (N = 1), clamps like its headless
// analogue --offscreen-frames, and stays independent of it. Pure config
// resolution - no Vulkan device, no g_vulkanTestCtx.
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "harmonia/app/CliParser.hpp"

namespace {

/// Parse @p args exactly the way src/demo/main.cpp does: one applyCommonArg call
/// per token, unknown tokens are a hard error (the B5 rule).
void parseLikeMain(harmonia::app::AppConfig& config, const std::vector<const char*>& args) {
    std::vector<char*> argv{};
    argv.reserve(args.size());
    for (const char* arg : args) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) — argv is writable by C linkage
        argv.push_back(const_cast<char*>(arg));
    }
    const int argc = static_cast<int>(argv.size());
    for (int i = 0; i < argc; ++i) {
        ASSERT_TRUE(harmonia::app::CliParser::applyCommonArg(config, i, argc, argv.data()))
            << "applyCommonArg rejected '" << argv[static_cast<std::size_t>(i)] << "'";
    }
}

} // namespace

TEST(TheiaCliContract, FramesPerFlipDefaultsToClassicCadence) {
    const harmonia::app::AppConfig config{};
    EXPECT_EQ(config.framesPerFlip, 1U);
}

TEST(TheiaCliContract, FramesPerFlipResolvesIntoConfig) {
    harmonia::app::AppConfig config{};
    parseLikeMain(config, {"--frames-per-flip", "4"});
    EXPECT_EQ(config.framesPerFlip, 4U);
}

TEST(TheiaCliContract, FramesPerFlipClampsToOneLikeOffscreenFrames) {
    harmonia::app::AppConfig config{};
    parseLikeMain(config, {"--frames-per-flip", "0"});
    EXPECT_EQ(config.framesPerFlip, 1U);
}

TEST(TheiaCliContract, FramesPerFlipInvalidValueKeepsDefaultAndConsumesArg) {
    harmonia::app::AppConfig config{};
    // The parser reports the bad value but still consumes the flag token —
    // main()'s loop must not see it again as an unknown argument.
    parseLikeMain(config, {"--frames-per-flip", "not-a-number"});
    EXPECT_EQ(config.framesPerFlip, 1U);
}

TEST(TheiaCliContract, FramesPerFlipIsIndependentOfOffscreenFrames) {
    harmonia::app::AppConfig config{};
    parseLikeMain(config, {"--offscreen-frames", "2", "--frames-per-flip", "4"});
    EXPECT_EQ(config.offscreenFrames, 2U);
    EXPECT_EQ(config.framesPerFlip, 4U);
}

// Lens glare is a stage of the shared host: Theia's main() does not parse it, so both
// flags reach the AppConfig through applyCommonArg, like every shared flag. A value the
// parser refuses keeps the default and is consumed: the token after it is parsed.

TEST(TheiaCliContract, GlareDefaultsToNoneAtReferenceWhite) {
    const harmonia::app::AppConfig config{};
    EXPECT_FLOAT_EQ(config.glare, 0.0F);
    EXPECT_FLOAT_EQ(config.glareThreshold, 1.0F);
}

TEST(TheiaCliContract, BothGlareFlagsResolveIntoConfig) {
    harmonia::app::AppConfig config{};
    parseLikeMain(config, {"--glare", "0.2", "--glare-threshold", "6"});
    EXPECT_FLOAT_EQ(config.glare, 0.2F);
    EXPECT_FLOAT_EQ(config.glareThreshold, 6.0F);
}

TEST(TheiaCliContract, GlareFlagsAreIndependentOfEachOtherAndOfTheirOrder) {
    harmonia::app::AppConfig thresholdOnly{};
    parseLikeMain(thresholdOnly, {"--glare-threshold", "6"});
    EXPECT_FLOAT_EQ(thresholdOnly.glare, 0.0F);
    EXPECT_FLOAT_EQ(thresholdOnly.glareThreshold, 6.0F);

    harmonia::app::AppConfig glareOnly{};
    parseLikeMain(glareOnly, {"--glare", "0.35"});
    EXPECT_FLOAT_EQ(glareOnly.glare, 0.35F);
    EXPECT_FLOAT_EQ(glareOnly.glareThreshold, 1.0F);

    harmonia::app::AppConfig thresholdFirst{};
    parseLikeMain(thresholdFirst, {"--glare-threshold", "6", "--glare", "0.2"});
    EXPECT_FLOAT_EQ(thresholdFirst.glare, 0.2F);
    EXPECT_FLOAT_EQ(thresholdFirst.glareThreshold, 6.0F);
}

TEST(TheiaCliContract, GlareAcceptsZeroWhichIsNone) {
    harmonia::app::AppConfig config{};
    parseLikeMain(config, {"--glare", "0", "--glare-threshold", "2.5"});
    EXPECT_FLOAT_EQ(config.glare, 0.0F);
    EXPECT_FLOAT_EQ(config.glareThreshold, 2.5F) << "a zero glare is a value, not a rejected one";
}

TEST(TheiaCliContract, RefusedGlareKeepsNoneAndConsumesItsValue) {
    // Below zero, not a number, a number with a tail, and the two non-finite ones.
    for (const char* value : {"-0.5", "abc", "0.2x", "nan", "inf"}) {
        SCOPED_TRACE(value);
        harmonia::app::AppConfig config{};
        parseLikeMain(config, {"--glare", value, "--offscreen-frames", "2"});
        EXPECT_FLOAT_EQ(config.glare, 0.0F);
        EXPECT_EQ(config.offscreenFrames, 2U) << "the refused value is consumed, the flag after it still parses";
    }
}

TEST(TheiaCliContract, RefusedGlareThresholdKeepsReferenceWhiteAndConsumesItsValue) {
    // Zero and below (the threshold must be above 0), not a number, and the non-finite ones.
    for (const char* value : {"0", "-1", "abc", "1.5x", "nan", "inf"}) {
        SCOPED_TRACE(value);
        harmonia::app::AppConfig config{};
        parseLikeMain(config, {"--glare-threshold", value, "--offscreen-frames", "2"});
        EXPECT_FLOAT_EQ(config.glareThreshold, 1.0F);
        EXPECT_EQ(config.offscreenFrames, 2U) << "the refused value is consumed, the flag after it still parses";
    }
}

TEST(TheiaCliContract, GlareFlagWithoutAValueKeepsTheDefaults) {
    // The last token of the command line: the parser reports the missing value and
    // leaves the configuration alone.
    harmonia::app::AppConfig glare{};
    parseLikeMain(glare, {"--glare"});
    EXPECT_FLOAT_EQ(glare.glare, 0.0F);

    harmonia::app::AppConfig threshold{};
    parseLikeMain(threshold, {"--glare-threshold"});
    EXPECT_FLOAT_EQ(threshold.glareThreshold, 1.0F);
}
