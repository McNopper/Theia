#ifndef THEIA_RENDERER_PASSTIMER_HPP
#define THEIA_RENDERER_PASSTIMER_HPP

#include <volk/volk.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace harmonia {
struct DeviceContext;
}

namespace theia {

/// Strict parse of the `THEIA_PASS_TIMING` toggle: exactly "1" enables per-pass GPU
/// timing; unset, "0", or any other value keeps it off (same A/B-toggle family as
/// THEIA_FORCE_GD3 / THEIA_SINGLE_PASS). Pure so the contract is unit-testable
/// without a Vulkan device (tests/module/test_pass_timer.cpp).
[[nodiscard]] inline bool passTimingEnabled(const std::optional<std::string>& raw) {
    return raw.has_value() && *raw == "1";
}

/// PERF5 10a instrumentation baseline: minimal per-pass GPU wall-time logger.
///
/// Records one `vkCmdWriteTimestamp2` pair around each of the renderer's compute
/// stages (GI, motion vectors, TAA — exactly the passes the PERF5 wavefront split
/// will restructure) and, while `THEIA_PASS_TIMING=1`, emits one log line per
/// frame:
///
///     timing: gi 4.32 ms | motion 0.10 ms | taa 0.62 ms
///
/// (a pass whose dispatch did not run this frame is skipped). When the toggle is
/// unset there is no GPU work at all — every hook early-outs on one boolean — and
/// the pools are never created. Deliberately NOT a profiling framework: fixed
/// pass set, no scopes, no nesting; the wavefront stages (10b+) time themselves
/// by adding entries here.
///
/// Sync model: frames use 2 slots (harmonia::app::FrameSync::kFrameSlots), one
/// query pool per slot. `beginFrame(slot)` reads + logs + resets the slot's pool
/// — the data there is frame N-2's batch. That read is safe to block on
/// (VK_QUERY_RESULT_WAIT_BIT): the graphics slot fence was already waited by the
/// App frame loop, and in async-compute mode the GI timestamps' completion is
/// guaranteed by the same timeline-semaphore wait (value − 2) the async submit
/// performs later in this frame — WAIT_BIT merely parks the host there earlier.
/// Resetting after the read is legal because availability implies the writing
/// command buffer completed, and the reset happens before this frame records new
/// timestamps into the same pool.
///
/// The GI pair is recorded on whichever command buffer carries the dispatch that
/// frame (async-compute or graphics queue); per-pass elapsed times are within one
/// queue, so they are exact either way.
class PassTimer {
  public:
    /// Timestamp indices inside each slot's pool — one boundary each.
    enum Query : std::uint32_t {
        kGiBegin = 0,
        kGiEnd,
        kMotionBegin,
        kMotionEnd,
        kTaaBegin,
        kTaaEnd,
        kQueryCount,
    };

    PassTimer() = default;
    ~PassTimer();
    PassTimer(const PassTimer&) = delete;
    PassTimer& operator=(const PassTimer&) = delete;

    /// Reads the toggle, checks device timestamp support, creates the per-slot
    /// query pools. Never fails the app: any degradation (toggle off, timestamps
    /// unsupported, pool creation failure) logs once and stays a no-op.
    void initialize(const harmonia::DeviceContext& ctx);
    void shutdown();

    /// Host-side frame hook — call once per frame BEFORE recording any pass:
    /// logs the slot's completed batch (frame N-2), resets its pool, and arms
    /// the slot this frame's timestamp hooks write into. No-op when disabled.
    void beginFrame(std::uint32_t slot);

    // Timestamp hooks (no-ops when disabled). Record around the matching pass
    // dispatch in the same command buffer; all passes are compute dispatches, so
    // one stage (COMPUTE) brackets every pair exactly.
    void giBegin(VkCommandBuffer cmd) { write(cmd, kGiBegin); }
    void giEnd(VkCommandBuffer cmd) { write(cmd, kGiEnd); }
    void motionBegin(VkCommandBuffer cmd) { write(cmd, kMotionBegin); }
    void motionEnd(VkCommandBuffer cmd) { write(cmd, kMotionEnd); }
    void taaBegin(VkCommandBuffer cmd) { write(cmd, kTaaBegin); }
    void taaEnd(VkCommandBuffer cmd) { write(cmd, kTaaEnd); }

    [[nodiscard]] bool enabled() const noexcept { return m_enabled; }

  private:
    /// == harmonia::app::FrameSync::kFrameSlots (one query pool per frame slot).
    static constexpr std::uint32_t kFrameSlots = 2;

    void write(VkCommandBuffer cmd, Query query);
    void logSlot(std::uint32_t slot);

    VkDevice m_device = VK_NULL_HANDLE;
    bool m_enabled = false;
    float m_timestampPeriodNs = 1.0F; ///< device limits.timestampPeriod (ns per tick)
    std::uint32_t m_activeSlot = 0;   ///< slot armed by beginFrame(); the hooks write here
    std::array<VkQueryPool, kFrameSlots> m_pools{};
    /// Bitmask (1 << Query) of timestamps written since the slot's pool was reset.
    std::array<std::uint32_t, kFrameSlots> m_writtenMask{};
    /// True once a slot has completed a full frame batch — gates the first reads
    /// (before frame kFrameSlots the pool holds nothing to wait on).
    std::array<bool, kFrameSlots> m_slotHasData{};
};

} // namespace theia
#endif // THEIA_RENDERER_PASSTIMER_HPP
