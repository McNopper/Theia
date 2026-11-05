#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

#include "harmonia/core/Env.hpp"
#include "harmonia/core/Logger.hpp"
#include "theia/gi/GiPass.hpp"
#include "theia/renderer/ShaderPath.hpp"

namespace theia {
namespace {
struct SmsProducerContext {
    VkDeviceAddress work, prefixes;
    std::uint32_t resolution, flags;
    VkDeviceAddress temporalReservoir, historyReservoir;
};
static_assert(sizeof(SmsProducerContext) == 40);
} // namespace

bool GiPass::createSmsResources(const harmonia::core::CommandPool& uploadPool) {
    static_cast<void>(uploadPool); // staging is handled by Buffer::uploadData
    SmsProducerContext context{};
    m_smsDiagEnabled = harmonia::core::getEnvString("THEIA_SMS_DIAGNOSTICS") == std::optional<std::string>{"1"};
    auto diagnostics =
        harmonia::core::Buffer::create(*m_ctx,
                                       64 * sizeof(std::uint32_t),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       VMA_MEMORY_USAGE_GPU_TO_CPU,
                                       "theia.sms.diagnostics");
    if (!diagnostics)
        return false;
    m_smsDiagnostics = std::move(*diagnostics);
    const std::array<std::uint32_t, 64> zeros{};
    m_smsDiagnostics.uploadData(zeros.data(), sizeof(zeros));
    if (m_cfg.smsEnabled) {
        const auto pixels = std::uint64_t{m_cfg.width} * m_cfg.height;
        if (pixels == 0 || pixels > harmonia::pipeline::SmsPass::kMaxCapacity || m_cfg.smsPartitionResolution == 0 ||
            m_cfg.smsPartitionResolution > 32 || m_cfg.smsTrialBudget == 0)
            return false;
        auto prefixes = harmonia::core::Buffer::create(*m_ctx,
                                                       pixels * sizeof(sm::float4),
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                       "theia.sms.prefixes");
        if (!prefixes)
            return false;
        m_smsPrefixes = std::move(*prefixes);
        const bool prior = m_cfg.smsPriorEnabled && m_cfg.smsPartitionResolution > 1;
        if (prior) {
            const auto blocksX = std::uint64_t{m_cfg.width} / 16 + (m_cfg.width % 16 != 0 ? 1 : 0);
            const auto blocksY = std::uint64_t{m_cfg.height} / 16 + (m_cfg.height % 16 != 0 ? 1 : 0);
            const auto entries = blocksX * blocksY * 128;
            if (entries > harmonia::pipeline::SmsPass::kMaxCapacity)
                return false;
            auto buffer = harmonia::core::Buffer::create(*m_ctx,
                                                         entries * 136,
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                         "theia.sms.prior");
            if (!buffer)
                return false;
            m_smsPrior = std::move(*buffer);
        }
        // Hong et al. 2025, section 5: persistent spatiotemporal reservoirs.
        // `temporal` holds this frame's PRE-SPATIAL merge output (the spatial
        // pass reads it); `history` holds last frame's copy (the temporal pass
        // reads it - copied at each frame start so the merge never races its
        // own output). The reference keeps the same split by swapping its
        // temporal buffer in before the spatial pass and clearing the spatial
        // output: history must be pre-spatial, or re-selected history
        // double-counts across frames. The 320B SmsReservoir stride is fixed
        // by the shader ABI.
        if (m_cfg.smsSpatialReuse) {
            // SmsReservoir: sizeof as computed by Slang = 320B (the struct tail
            // rounds up to 16-byte alignment). Must match the shader's pointer
            // stride exactly; a smaller allocation is OOB device-lost.
            constexpr VkDeviceSize kSmsReservoirStride = 320;
            if (pixels > (harmonia::pipeline::SmsPass::kMaxCapacity))
                return false;
            constexpr auto reservoirUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            auto temporal = harmonia::core::Buffer::create(*m_ctx,
                                                           pixels * kSmsReservoirStride,
                                                           reservoirUsage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                           VMA_MEMORY_USAGE_GPU_TO_CPU,
                                                           "theia.sms.reservoir.temporal");
            if (!temporal)
                return false;
            m_smsTemporalReservoir = std::move(*temporal);
            auto historyReservoir = harmonia::core::Buffer::create(*m_ctx,
                                                                   pixels * kSmsReservoirStride,
                                                                   reservoirUsage,
                                                                   VMA_MEMORY_USAGE_GPU_TO_CPU,
                                                                   "theia.sms.reservoir.history");
            if (!historyReservoir)
                return false;
            m_smsHistoryReservoir = std::move(*historyReservoir);
        }
        const std::filesystem::path shared{HARMONIA_SHADER_DIR};
        auto sms = harmonia::pipeline::SmsPass::create(
            *m_ctx,
            static_cast<std::uint32_t>(pixels),
            m_pipelineLayout,
            shared / "sms_queue_control.spv",
            shaderPath("sms_sample.spv"),
            shaderPath("sms_resolve.spv"),
            m_cfg.smsTrialBudget,
            VK_SHADER_STAGE_COMPUTE_BIT,
            prior ? shaderPath("sms_prior_build.spv") : std::filesystem::path{},
            prior ? m_smsPrior.deviceAddress() : 0,
            m_smsTemporalReservoir.isValid() ? shaderPath("sms_temporal.spv") : std::filesystem::path{});
        if (!sms)
            return false;
        m_sms = std::move(*sms);
        context = {m_sms.workBuffer().deviceAddress(),
                   m_smsPrefixes.deviceAddress(),
                   m_cfg.smsPartitionResolution,
                   (m_cfg.smsSpatialReuse ? 1U : 0U) | (m_smsDiagEnabled ? 2U : 0U) |
                       // Bit 2: the temporal dispatch is armed - the reuse
                       // population semantics follow the reference (no records
                       // fallback; see smsSpatialEstimate).
                       (m_smsTemporalReservoir.isValid() ? 4U : 0U),
                   m_smsTemporalReservoir.isValid() ? m_smsTemporalReservoir.deviceAddress() : 0,
                   m_smsHistoryReservoir.isValid() ? m_smsHistoryReservoir.deviceAddress() : 0};
    }
    // Immutable configuration, uploaded only at initialization/resize. A disabled
    // context is real configuration, never a host-written hot-path work counter.
    // Host-visible so the diagnostics readback can verify the GPU-visible
    // reservoir pointers during the temporal bring-up.
    auto buffer =
        harmonia::core::Buffer::create(*m_ctx,
                                       sizeof(SmsProducerContext),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VMA_MEMORY_USAGE_GPU_TO_CPU,
                                       "theia.sms.context");
    if (!buffer)
        return false;
    buffer->uploadData(&context, sizeof(context));
    m_smsContext = std::move(*buffer);
    for (auto& writer : m_descWriters) {
        writer.writeStorageBufferHandle(*m_ctx, 25, m_smsContext);
        writer.writeStorageBufferHandle(*m_ctx, 26, m_smsDiagnostics);
    }
    m_smsReservoirsCleared = false;
    return true;
}

void GiPass::reportSmsDiagnostics() const noexcept {
    if (!m_smsDiagEnabled)
        return;
    m_smsDiagnosticsReported = true; // TEMP: per-frame reporting
    std::array<std::uint32_t, 64> counters{};
    if (m_smsDiagnostics.readData(counters.data(), sizeof(counters)) != VK_SUCCESS) {
        harmonia::core::Logger::error("Theia SMS diagnostics readback failed (observation only)");
        return;
    }
    harmonia::core::Logger::info(
        "sms_diag eligible={} roots={} canonical={} domains={} candidates={} canonicalInv={}/{} neighborInv={}/{} "
        "selected={} topologyFail={} solverFail={} visibilityFail={} supportFail={} completedReciprocals={} "
        "maxTrials={} maxBase=({:.6g},{:.6g},{:.6g}) maxFinal={:.6g} maxConditionalMass={:.6g} "
        "maxReciprocalWeight={:.6g} temporal={} temporalCentral={} temporalCandidateErr={} spatialRgbErr={} "
        "spatialResampleErr={} spatialSelectErr={} maxReservoirWeight={:.6g} maxPrevMass={:.6g} "
        "maxCentralReservoirWeight={:.6g} maxSourceReservoirWeight={:.6g} maxCentralUcw={:.6g} "
        "maxCentralLum={:.6g} maxTemporalWeight={:.6g} maxCentralWeight={:.6g} overflowRoutes={} temporalSelects={} "
        "maxSelectedLum={:.6g} maxPrevMassEarly={} probe_weightSum={:.6g} probe_selLum={:.6g} probe_weight={:.6g} "
        "probe_mass={:.6g} probe_trials={} probe_PC={:.6g} rsl_weight={:.6g} rsl_mass={:.6g} rsl_estX={:.6g} "
        "rsl_trials={} "
        "hot_slot={} hot_weightSum={:.6g} hot_selLum={:.6g} hot_weight={:.6g} hot_mis={:.6g} hot_ucw={:.6g} "
        "hot_lum={:.6g} hot_tMis={:.6g} hot_pMass={:.6g} hot_tUcw={:.6g} recordsErr={} dimErr={} reservoirErr={} "
        "hdrNonfinite={} partitionMax={} partitionSum={} partitionN={}",
        counters[0],
        counters[1],
        counters[2],
        counters[3],
        counters[4],
        counters[6],
        counters[5],
        counters[8],
        counters[7],
        counters[9],
        counters[10],
        counters[11],
        counters[12],
        counters[13],
        counters[15],
        counters[14],
        std::bit_cast<float>(counters[16]),
        std::bit_cast<float>(counters[17]),
        std::bit_cast<float>(counters[18]),
        std::bit_cast<float>(counters[19]),
        std::bit_cast<float>(counters[20]),
        std::bit_cast<float>(counters[21]),
        counters[22],
        counters[23],
        counters[24],
        counters[29],
        counters[30],
        counters[31],
        std::bit_cast<float>(counters[25]),
        std::bit_cast<float>(counters[26]),
        std::bit_cast<float>(counters[27]),
        std::bit_cast<float>(counters[28]),
        std::bit_cast<float>(counters[33]),
        std::bit_cast<float>(counters[34]),
        std::bit_cast<float>(counters[35]),
        std::bit_cast<float>(counters[36]),
        counters[32],
        counters[37],
        std::bit_cast<float>(counters[38]),
        counters[39],
        std::bit_cast<float>(counters[40]),
        std::bit_cast<float>(counters[41]),
        std::bit_cast<float>(counters[42]),
        std::bit_cast<float>(counters[43]),
        counters[44],
        std::bit_cast<float>(counters[45]),
        std::bit_cast<float>(counters[46]),
        std::bit_cast<float>(counters[47]),
        std::bit_cast<float>(counters[48]),
        counters[49],
        counters[40],
        std::bit_cast<float>(counters[41]),
        std::bit_cast<float>(counters[42]),
        std::bit_cast<float>(counters[43]),
        std::bit_cast<float>(counters[44]),
        std::bit_cast<float>(counters[45]),
        std::bit_cast<float>(counters[46]),
        std::bit_cast<float>(counters[47]),
        std::bit_cast<float>(counters[48]),
        std::bit_cast<float>(counters[49]),
        counters[50],
        counters[51],
        counters[52],
        counters[53],
        std::bit_cast<float>(counters[57]),
        counters[60],
        counters[61]);
    // Ground truth: direct host readback of the stored reservoirs. Locate the
    // temporal merge magic (reserved0 = 0x12345678) to establish the stride
    // the shader actually used, then dump that entry's fields.
    if (m_smsTemporalReservoir.isValid() && m_smsHistoryReservoir.isValid()) {
        const std::uint64_t slots = std::uint64_t{m_cfg.width} * m_cfg.height;
        const std::uint64_t words = slots * 80;
        std::vector<std::uint32_t> temporal(words);
        std::vector<std::uint32_t> history(words);
        if (m_smsTemporalReservoir.readData(temporal.data(), temporal.size() * sizeof(std::uint32_t)) == VK_SUCCESS &&
            m_smsHistoryReservoir.readData(history.data(), history.size() * sizeof(std::uint32_t)) == VK_SUCCESS) {
            for (auto* table : {&temporal, &history}) {
                // Full-buffer magic scan: every occurrence of the merge magic
                // (reserved0) regardless of stride. The spacing between hits IS
                // the stride the shader actually used; zero hits prove the
                // stores landed outside this buffer entirely.
                std::array<std::uint64_t, 8> magicPos{};
                std::uint64_t magicCount = 0;
                for (std::uint64_t word = 0; word < words; ++word) {
                    if (table->at(word) == 0x12345678u) {
                        if (magicCount < magicPos.size())
                            magicPos[magicCount] = word;
                        ++magicCount;
                    }
                }
                harmonia::core::Logger::info(
                    "sms_reservoir_magicscan buffer={} count={} p0={} p1={} p2={} p3={} p4={} p5={}",
                    table == &temporal ? "temporal" : "history",
                    magicCount,
                    magicPos[0],
                    magicPos[1],
                    magicPos[2],
                    magicPos[3],
                    magicPos[4],
                    magicPos[5]);
                for (auto [name, stride] :
                     {std::pair{"s77", 77ULL}, {std::pair{"s76", 76ULL}}, {std::pair{"s80", 80ULL}}}) {
                    float bestWeight = 0.0f;
                    std::uint64_t bestSlot = 0;
                    std::uint64_t fills = 0;
                    std::uint64_t magics = 0;
                    for (std::uint64_t slot = 0; slot < slots; ++slot) {
                        if (table->at(slot * stride + 74) == 0xDEADBEEFu)
                            ++fills;
                        if (table->at(slot * stride + 74) == 0x12345678u)
                            ++magics;
                    }
                    for (std::uint64_t slot = 0; slot < slots; ++slot) {
                        const float weight = std::bit_cast<float>(table->at(slot * stride + 71));
                        if (std::isfinite(weight) && weight > bestWeight) {
                            bestWeight = weight;
                            bestSlot = slot;
                        }
                    }
                    const std::uint64_t base = bestSlot * stride;
                    // Dump the top entries by weight at this stride, with their
                    // full derivation fields (trials, P(C), base sum).
                    std::vector<std::pair<float, std::uint64_t>> ranked;
                    for (std::uint64_t slot = 0; slot < slots; ++slot) {
                        const float weight = std::bit_cast<float>(table->at(slot * stride + 71));
                        if (std::isfinite(weight) && weight > 0.0f)
                            ranked.emplace_back(weight, slot);
                    }
                    std::sort(
                        ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                    for (std::size_t i = 0; i < ranked.size() && i < 3; ++i) {
                        const std::uint64_t e = ranked[i].second * stride;
                        harmonia::core::Logger::info(
                            "sms_reservoir_top buffer={} stride={} rank={} slot={} weight={:.6g} mass={:.6g} "
                            "count={} trials={} status={} baseSum={:.6g} probC={:.6g} estX={:.6g} reserved={:#x}",
                            table == &temporal ? "temporal" : "history",
                            name,
                            i,
                            ranked[i].second,
                            ranked[i].first,
                            std::bit_cast<float>(table->at(e + 72)),
                            table->at(e + 73),
                            table->at(e + 48),
                            table->at(e + 50),
                            std::bit_cast<float>(table->at(e + 44)),
                            std::bit_cast<float>(table->at(e + 47)),
                            std::bit_cast<float>(table->at(e + 68)),
                            table->at(e + 74));
                    }
                    harmonia::core::Logger::info(
                        "sms_reservoir_dump buffer={} stride={} bestSlot={} weight={:.6g} mass={:.6g} count={} "
                        "trials={} baseSum={:.6g} probC={:.6g} estX={:.6g} reserved={} fills={} magics={}",
                        table == &temporal ? "temporal" : "history",
                        name,
                        bestSlot,
                        std::bit_cast<float>(table->at(base + 71)),
                        std::bit_cast<float>(table->at(base + 72)),
                        table->at(base + 73),
                        table->at(base + 48),
                        std::bit_cast<float>(table->at(base + 44)),
                        std::bit_cast<float>(table->at(base + 47)),
                        std::bit_cast<float>(table->at(base + 68)),
                        table->at(base + 74),
                        fills,
                        magics);
                }
            }
        }
        // Context pointer audit: read back the uploaded context buffer and
        // compare its reservoir pointers with the allocations' addresses.
        std::array<std::uint32_t, 10> context{};
        if (m_smsContext.readData(context.data(), sizeof(context)) == VK_SUCCESS) {
            harmonia::core::Logger::info(
                "sms_context_dump ctxWork={:#x} ctxPrefixes={:#x} ctxResolution={} ctxFlags={} "
                "ctxTemporal={:#x} ctxHistory={:#x} allocTemporal={:#x} allocHistory={:#x}",
                context[0] | std::uint64_t(context[1]) << 32,
                context[2] | std::uint64_t(context[3]) << 32,
                context[4],
                context[5],
                context[6] | std::uint64_t(context[7]) << 32,
                context[8] | std::uint64_t(context[9]) << 32,
                m_smsTemporalReservoir.deviceAddress(),
                m_smsHistoryReservoir.deviceAddress());
        }
    }
}

std::expected<bool, VkResult> GiPass::frameReady() const noexcept {
    if (m_frameError != VK_SUCCESS)
        return std::unexpected(m_frameError);
    if (!m_frameRecorded)
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    if (!m_smsFrameActive) {
        reportSmsDiagnostics();
        return true;
    }
    const auto completion = m_sms.completion();
    if (!completion) {
        m_frameError = completion.error();
        return std::unexpected(m_frameError);
    }
    if (completion->status == harmonia::pipeline::SmsQueueCompletion::Failed) {
        // Observation-only counters persist across submissions; report them
        // before failing so the error site is identifiable.
        if (m_smsDiagEnabled)
            reportSmsDiagnostics();
        harmonia::core::Logger::error("Theia SMS logical frame failed: GPU error bits {}", completion->error);
        m_frameError = VK_ERROR_UNKNOWN;
        return std::unexpected(m_frameError);
    }
    const bool ready = completion->status == harmonia::pipeline::SmsQueueCompletion::Ready;
    if (ready)
        reportSmsDiagnostics();
    return ready;
}

VkResult GiPass::recordContinuation(VkCommandBuffer cmd) noexcept {
    if (m_frameError != VK_SUCCESS)
        return m_frameError;
    if (!m_frameRecorded || !m_smsFrameActive || cmd == VK_NULL_HANDLE) {
        m_frameError = VK_ERROR_INITIALIZATION_FAILED;
        return m_frameError;
    }
    m_descWriters[m_activeSlot].bind(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout);
    m_frameError = m_sms.recordBatch(cmd);
    return m_frameError;
}
} // namespace theia
