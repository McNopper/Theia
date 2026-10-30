// CPU-contract tests for the PERF5 10a per-pass timing toggle (PassTimer).
// THEIA_PASS_TIMING is read once at Application init (passTimingEnabled, a pure
// function in theia/renderer/PassTimer.hpp). These tests pin that contract:
// exactly "1" enables, everything else (unset, "0", junk) stays off — the same
// strict-toggle family as THEIA_FORCE_GD3. Pure string resolution — no Vulkan
// device, no GPU work.
#include <gtest/gtest.h>
#include <optional>
#include <string>

#include "theia/renderer/PassTimer.hpp"

namespace {

constexpr bool parse(const char* raw) {
    return theia::passTimingEnabled(raw == nullptr ? std::nullopt : std::optional<std::string>(raw));
}

} // namespace

TEST(TheiaPassTimerContract, UnsetToggleDisablesTiming) {
    EXPECT_FALSE(parse(nullptr));
    EXPECT_FALSE(theia::passTimingEnabled(std::nullopt));
}

TEST(TheiaPassTimerContract, ExactlyOneEnablesTiming) {
    EXPECT_TRUE(parse("1"));
}

TEST(TheiaPassTimerContract, ZeroAndJunkKeepTimingOff) {
    EXPECT_FALSE(parse("0"));
    EXPECT_FALSE(parse(""));
    EXPECT_FALSE(parse("true"));
    EXPECT_FALSE(parse("on"));
    EXPECT_FALSE(parse("2"));
    EXPECT_FALSE(parse(" 1")); // no implicit trim
    EXPECT_FALSE(parse("1 ")); // no implicit trim
}
