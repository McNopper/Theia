// CPU-contract tests for the PERF5 10a per-pass timing toggle (PassTimer).
// THEIA_PASS_TIMING is read once at Application init (passTimingEnabled, a pure
// function in theia/renderer/PassTimer.hpp). These tests pin that contract:
// exactly "1" enables, everything else (unset, "0", junk) stays off — the same
// strict-toggle family as THEIA_FORCE_GD3. Pure string resolution — no Vulkan
// device, no GPU work.
#include <atomic>
#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <string>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Env.hpp"
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

namespace {
struct TimingToggle {
    std::optional<std::string> previous = harmonia::core::getEnvString("THEIA_PASS_TIMING");
    static int set(const char* value) {
#ifdef _WIN32
        return _putenv_s("THEIA_PASS_TIMING", value == nullptr ? "" : value);
#else
        return value == nullptr ? unsetenv("THEIA_PASS_TIMING") : setenv("THEIA_PASS_TIMING", value, 1);
#endif
    }
    int result = set("1");
    ~TimingToggle() { static_cast<void>(set(previous ? previous->c_str() : nullptr)); }
};
struct TimerValidationProbe {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    std::atomic_uint messages{0};
    static VKAPI_ATTR VkBool32 VKAPI_CALL callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                                   VkDebugUtilsMessageTypeFlagsEXT,
                                                   const VkDebugUtilsMessengerCallbackDataEXT*,
                                                   void* user) {
        ++static_cast<TimerValidationProbe*>(user)->messages;
        return VK_FALSE;
    }
    ~TimerValidationProbe() {
        if (messenger != VK_NULL_HANDLE)
            vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
    }
};
// Test-process-only fault injection at the Vulkan seam. This never changes the
// renderer or its shaders; the identical validation oracle must reject no reset.
struct QueryResetControl {
    PFN_vkCmdResetQueryPool original = vkCmdResetQueryPool;
    static VKAPI_ATTR void VKAPI_CALL ignore(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}
    QueryResetControl() {
        if (harmonia::core::getEnvString("THEIA_TIMER_SKIP_RESET_CONTROL") == std::optional<std::string>{"1"})
            vkCmdResetQueryPool = ignore;
    }
    ~QueryResetControl() { vkCmdResetQueryPool = original; }
};
} // namespace

TEST_F(VulkanFixture, PassTimerResetsFirstUseReusedSlotsAndPreviouslySkippedPairs) {
    const QueryResetControl resetControl;
    TimingToggle toggle;
    ASSERT_EQ(toggle.result, 0);
    TimerValidationProbe validation;
    validation.instance = g_vulkanTestCtx->context->instance();
    ASSERT_NE(vkCreateDebugUtilsMessengerEXT, nullptr);
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = TimerValidationProbe::callback;
    info.pUserData = &validation;
    ASSERT_EQ(vkCreateDebugUtilsMessengerEXT(validation.instance, &info, nullptr, &validation.messenger), VK_SUCCESS);
    theia::PassTimer timer;
    timer.initialize(deviceCtx());
    ASSERT_TRUE(timer.enabled()) << "Actual timestamp execution is required by this test";
    for (std::uint32_t frame = 0; frame < 6; ++frame) {
        timer.beginFrame(frame % 2);
        const auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        timer.giBegin(*cmd);
        timer.giEnd(*cmd);
        if (frame != 2) {
            timer.motionBegin(*cmd);
            timer.motionEnd(*cmd);
        }
        if (frame >= 2) {
            timer.taaBegin(*cmd);
            timer.taaEnd(*cmd);
        }
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
    }
    timer.beginFrame(0);
    timer.beginFrame(1);
    timer.shutdown();
    EXPECT_EQ(validation.messages.load(), 0U)
        << "First/reused/skipped query pairs must be reset without hostQueryReset";
}
