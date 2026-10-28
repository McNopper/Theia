// CPU-contract tests for the move-resets-accumulation seam behind
// Application::onUpdate() (src/demo/ViewAccumulation.hpp).
//
// The contract: a camera view change (position / target / up / EV100), or the
// very first frame where no signature exists yet, restarts progressive
// accumulation -- Application calls App::resetAccumulation(), which bumps the
// m_accumViewEpoch token; ViewAccumulationGate::epoch mirrors exactly that
// token. The decision consumes NO presentation-cadence input: --frames-per-flip
// gates only the PRESENT half (Harmonia's AppPresent), so it cannot suppress
// the reset -- rendering and this decision run on every iteration.
// Pure CPU logic -- no Vulkan device, no g_vulkanTestCtx.
#include <cstdint>
#include <gtest/gtest.h>
#include <slang-math/slang-math.hpp>

#include "demo/ViewAccumulation.hpp"

namespace {

const sm::float3 kPos{0.0f, 0.0f, 5.0f};
const sm::float3 kTarget{0.0f, 0.0f, 0.0f};
const sm::float3 kUp{0.0f, 1.0f, 0.0f};
constexpr float kEv100 = 0.0f;

} // namespace

// First frame: no signature stored yet, so the view always counts as changed
// and the accumulation resets once (a fresh, non-accumulated sample).
TEST(TheiaViewAccumulationContract, FirstFrameHasNoSignatureAndAlwaysResets) {
    theia::ViewAccumulationGate gate;
    EXPECT_EQ(gate.epoch, 0U);
    EXPECT_TRUE(gate.viewChanged(kPos, kTarget, kUp, kEv100));
    EXPECT_TRUE(gate.update(kPos, kTarget, kUp, kEv100));
    EXPECT_EQ(gate.epoch, 1U);
}

// Identical view state -> no change and no spurious reset: a stationary camera
// converges (exactly one reset, the first frame).
TEST(TheiaViewAccumulationContract, IdenticalViewDoesNotReset) {
    theia::ViewAccumulationGate gate;
    ASSERT_TRUE(gate.update(kPos, kTarget, kUp, kEv100));
    EXPECT_FALSE(gate.viewChanged(kPos, kTarget, kUp, kEv100));
    EXPECT_FALSE(gate.update(kPos, kTarget, kUp, kEv100));
    EXPECT_EQ(gate.epoch, 1U);
}

// Each signature dimension separately: any single one changing is a new
// viewpoint and must reset.
TEST(TheiaViewAccumulationContract, CameraPositionChangeResets) {
    theia::ViewAccumulationGate gate;
    ASSERT_TRUE(gate.update(kPos, kTarget, kUp, kEv100));
    const sm::float3 moved{1.0f, 0.0f, 5.0f};
    EXPECT_TRUE(gate.viewChanged(moved, kTarget, kUp, kEv100));
    EXPECT_TRUE(gate.update(moved, kTarget, kUp, kEv100));
    EXPECT_EQ(gate.epoch, 2U);
}

TEST(TheiaViewAccumulationContract, CameraTargetChangeResets) {
    theia::ViewAccumulationGate gate;
    ASSERT_TRUE(gate.update(kPos, kTarget, kUp, kEv100));
    const sm::float3 looked{0.0f, 1.0f, 0.0f};
    EXPECT_TRUE(gate.viewChanged(kPos, looked, kUp, kEv100));
    EXPECT_TRUE(gate.update(kPos, looked, kUp, kEv100));
    EXPECT_EQ(gate.epoch, 2U);
}

TEST(TheiaViewAccumulationContract, CameraUpChangeResets) {
    theia::ViewAccumulationGate gate;
    ASSERT_TRUE(gate.update(kPos, kTarget, kUp, kEv100));
    const sm::float3 rolled{1.0f, 1.0f, 0.0f};
    EXPECT_TRUE(gate.viewChanged(kPos, kTarget, rolled, kEv100));
    EXPECT_TRUE(gate.update(kPos, kTarget, rolled, kEv100));
    EXPECT_EQ(gate.epoch, 2U);
}

TEST(TheiaViewAccumulationContract, Ev100ChangeResets) {
    theia::ViewAccumulationGate gate;
    ASSERT_TRUE(gate.update(kPos, kTarget, kUp, kEv100));
    constexpr float kOtherEv100 = 1.5f;
    EXPECT_TRUE(gate.viewChanged(kPos, kTarget, kUp, kOtherEv100));
    EXPECT_TRUE(gate.update(kPos, kTarget, kUp, kOtherEv100));
    EXPECT_EQ(gate.epoch, 2U);
}

// The consumption contract: the gate epoch IS the accumulation-reset token
// (Application feeds resetAccumulation() exactly on the true transitions), so
// it bumps once per view change and never on a static frame.
TEST(TheiaViewAccumulationContract, UpdateBumpsResetTokenExactlyOnViewChange) {
    theia::ViewAccumulationGate gate;
    EXPECT_TRUE(gate.update(kPos, kTarget, kUp, kEv100));                                 // first frame: reset
    EXPECT_FALSE(gate.update(kPos, kTarget, kUp, kEv100));                                // static: no reset
    EXPECT_FALSE(gate.update(kPos, kTarget, kUp, kEv100));                                // static: no reset
    EXPECT_TRUE(gate.update(kPos + sm::float3{0.0f, 1.0f, 0.0f}, kTarget, kUp, kEv100));  // move: reset
    EXPECT_FALSE(gate.update(kPos + sm::float3{0.0f, 1.0f, 0.0f}, kTarget, kUp, kEv100)); // parked: no reset
    EXPECT_EQ(gate.epoch, 2U);
}

// The pacing contract: --frames-per-flip throttles only the PRESENT half, so
// the flip cadence appears nowhere in this decision. A camera move resets
// accumulation wherever it lands in the cadence (on a flip frame or between
// flips); a static view never resets, even across flips.
TEST(TheiaViewAccumulationContract, MoveResetsAccumulationRegardlessOfFlipCadence) {
    theia::ViewAccumulationGate gate;
    const auto frame = [&gate](const sm::float3& pos) { return gate.update(pos, kTarget, kUp, kEv100); };

    // 4-frame flip cadence: '|' marks a swapchain flip on every 4th iteration.
    EXPECT_TRUE(frame(kPos));                                 // 1  (first frame)
    EXPECT_FALSE(frame(kPos));                                // 2
    EXPECT_FALSE(frame(kPos));                                // 3
    EXPECT_FALSE(frame(kPos));                                // 4 | converged view stays converged
    EXPECT_TRUE(frame(kPos + sm::float3{0.5f, 0.0f, 0.0f}));  // 5 | move ON a flip frame: reset
    EXPECT_TRUE(frame(kPos + sm::float3{1.0f, 0.0f, 0.0f}));  // 6 move BETWEEN flips: reset
    EXPECT_FALSE(frame(kPos + sm::float3{1.0f, 0.0f, 0.0f})); // 7
    EXPECT_FALSE(frame(kPos + sm::float3{1.0f, 0.0f, 0.0f})); // 8 | static across flip: no reset

    EXPECT_EQ(gate.epoch, 3U);
}
