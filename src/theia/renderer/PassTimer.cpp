// PassTimer.cpp — PERF5 10a per-pass GPU timing (THEIA_PASS_TIMING=1).
// See PassTimer.hpp for the sync model and log contract.
#include "theia/renderer/PassTimer.hpp"

#include <array>
#include <cstdint>
#include <format>
#include <string>
#include <utility>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Env.hpp"
#include "harmonia/core/Logger.hpp"

namespace theia {

namespace {
constexpr float kNsToMs = 1.0e-6F;
}

PassTimer::~PassTimer() {
    shutdown();
}

void PassTimer::initialize(const harmonia::DeviceContext& ctx) {
    if (!passTimingEnabled(harmonia::core::getEnvString("THEIA_PASS_TIMING"))) {
        return; // toggle off: zero cost, no pools
    }
    m_device = ctx.device;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
    if (props.limits.timestampComputeAndGraphics == VK_FALSE || props.limits.timestampPeriod <= 0.0F) {
        harmonia::core::Logger::warn("THEIA_PASS_TIMING=1 but the device has no timestamp queries — timing off");
        m_device = VK_NULL_HANDLE;
        return;
    }
    m_timestampPeriodNs = props.limits.timestampPeriod;

    for (auto& pool : m_pools) {
        const VkQueryPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = kQueryCount,
            .pipelineStatistics = 0,
        };
        if (const VkResult r = vkCreateQueryPool(m_device, &info, nullptr, &pool); r != VK_SUCCESS) {
            harmonia::core::Logger::error("PassTimer: query pool creation failed: VkResult {}", static_cast<int>(r));
            shutdown();
            return;
        }
    }
    m_enabled = true;
    harmonia::core::Logger::info("THEIA_PASS_TIMING=1 — per-pass GPU timing on (timestampPeriod {:.3f} ns)",
                                 m_timestampPeriodNs);
}

void PassTimer::shutdown() {
    if (m_device != VK_NULL_HANDLE) {
        for (auto& pool : m_pools) {
            if (pool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(m_device, pool, nullptr);
                pool = VK_NULL_HANDLE;
            }
        }
    }
    m_enabled = false;
    m_device = VK_NULL_HANDLE;
    m_writtenMask = {};
    m_slotHasData = {};
}

void PassTimer::beginFrame(const std::uint32_t slot) {
    if (!m_enabled) {
        return;
    }
    m_activeSlot = slot % kFrameSlots;
    if (m_slotHasData[m_activeSlot]) {
        logSlot(m_activeSlot);
    }
    m_writtenMask[m_activeSlot] = 0;
}

void PassTimer::write(const VkCommandBuffer cmd, const Query query) {
    if (!m_enabled) {
        return;
    }
    vkCmdWriteTimestamp2(
        cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_pools[m_activeSlot], static_cast<std::uint32_t>(query));
    m_writtenMask[m_activeSlot] |= 1U << static_cast<std::uint32_t>(query);
    m_slotHasData[m_activeSlot] = true;
}

void PassTimer::logSlot(const std::uint32_t slot) {
    const VkQueryPool pool = m_pools[slot];
    const std::uint32_t mask = m_writtenMask[slot];
    if (mask == 0U) {
        return; // nothing dispatched into this slot last round (e.g. GI-less frame)
    }

    // Elapsed ms per fully-written begin/end pair, in pipeline order. Each pair is
    // read separately (WAIT_BIT) — a pass that did not run has unwritten queries,
    // and waiting on those would block forever.
    std::string line;
    const std::pair<Query, Query> pairs[3] = {{kGiBegin, kGiEnd}, {kMotionBegin, kMotionEnd}, {kTaaBegin, kTaaEnd}};
    const char* names[3] = {"gi", "motion", "taa"};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto [begin, end] = pairs[i];
        const std::uint32_t pairMask =
            (1U << static_cast<std::uint32_t>(begin)) | (1U << static_cast<std::uint32_t>(end));
        if ((mask & pairMask) != pairMask) {
            continue; // pass did not run in this batch
        }
        std::array<std::uint64_t, 2> ticks{};
        const VkResult r = vkGetQueryPoolResults(m_device,
                                                 pool,
                                                 static_cast<std::uint32_t>(begin),
                                                 2,
                                                 sizeof(ticks),
                                                 ticks.data(),
                                                 sizeof(std::uint64_t),
                                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (r != VK_SUCCESS) {
            harmonia::core::Logger::warn("PassTimer: query read failed: VkResult {}", static_cast<int>(r));
            return;
        }
        const std::uint64_t delta = ticks[1] >= ticks[0] ? ticks[1] - ticks[0] : 0U;
        if (!line.empty()) {
            line += " |";
        }
        line += std::format(
            " {} {:.2f} ms", names[i], static_cast<double>(delta) * static_cast<double>(m_timestampPeriodNs) * kNsToMs);
    }
    if (line.empty()) {
        return;
    }
    harmonia::core::Logger::info("timing:{}", line);

    // Safe: availability (WAIT_BIT above) implies the writing command buffer completed,
    // and this frame's timestamps are recorded after this reset.
    vkResetQueryPool(m_device, pool, 0, kQueryCount);
}

} // namespace theia
