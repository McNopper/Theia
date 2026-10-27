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
