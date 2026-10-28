#ifndef DEMO_VIEW_ACCUMULATION_HPP
#define DEMO_VIEW_ACCUMULATION_HPP

#include <cstdint>
#include <slang-math/slang-math.hpp>

namespace theia {

/// The move-resets-accumulation decision extracted from Application::onUpdate()
/// as a headless-testable seam (the demo host cannot be constructed without its
/// SDL window and Vulkan device: harmonia::app::App::bootstrap() owns both).
///
/// Contract: Application::onUpdate() calls harmonia::app::App::resetAccumulation()
/// -- which bumps App's private m_accumViewEpoch token -- exactly when update()
/// returns true. The epoch member below mirrors that token one-to-one, so CPU
/// tests can pin the reset semantics. The signature covers camera position,
/// target, up and EV100; the first frame (no stored signature yet) always reads
/// as changed. The decision consumes NO presentation-cadence input: the
/// --frames-per-flip gate sits in the PRESENT half (Harmonia's AppPresent) and
/// cannot suppress this reset -- rendering and this decision run every iteration.
struct ViewAccumulationGate {
    /// Previous view signature (defaults identical to the former Application members).
    sm::float3 prevPos{0.0f, 0.0f, 0.0f};
    sm::float3 prevTarget{0.0f, 0.0f, 0.0f};
    sm::float3 prevUp{0.0f, 0.0f, 0.0f};
    float prevEv100 = 0.0f;
    /// False until the first signature is stored; the first frame always resets.
    bool valid = false;

    /// Test observable: the accumulation-reset token, mirroring App's private
    /// m_accumViewEpoch. Bumped iff update() returns true (Application calls
    /// resetAccumulation() on exactly that transition).
    std::uint64_t epoch = 0;

    /// Pure decision: did the view change relative to the stored signature?
    [[nodiscard]] bool viewChanged(sm::float3 position, sm::float3 target, sm::float3 up, float ev100) const noexcept {
        return !valid || position != prevPos || target != prevTarget || up != prevUp || ev100 != prevEv100;
    }

    /// Decision + commit: returns viewChanged(); when true, stores the new
    /// signature and bumps the reset epoch. Behavior-identical to the inline
    /// block this replaced in Application::onUpdate().
    bool update(sm::float3 position, sm::float3 target, sm::float3 up, float ev100) noexcept {
        const bool changed = viewChanged(position, target, up, ev100);
        if (changed) {
            prevPos = position;
            prevTarget = target;
            prevUp = up;
            prevEv100 = ev100;
            valid = true;
            ++epoch;
        }
        return changed;
    }
};

} // namespace theia
#endif // DEMO_VIEW_ACCUMULATION_HPP
