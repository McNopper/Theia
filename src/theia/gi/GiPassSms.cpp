#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>

#include "harmonia/core/Env.hpp"
#include "harmonia/core/Logger.hpp"
#include "theia/gi/GiPass.hpp"
#include "theia/renderer/ShaderPath.hpp"

namespace theia {
namespace {
struct SmsProducerContext {
    VkDeviceAddress work, prefixes;
    std::uint32_t resolution, flags;
    VkDeviceAddress temporalReservoir, finalReservoir;
};
static_assert(sizeof(SmsProducerContext) == 40);
} // namespace

bool GiPass::createSmsResources(const harmonia::core::CommandPool& uploadPool) {
    SmsProducerContext context{};
    m_smsDiagEnabled = harmonia::core::getEnvString("THEIA_SMS_DIAGNOSTICS") == std::optional<std::string>{"1"};
    auto diagnostics =
        harmonia::core::Buffer::create(*m_ctx,
                                       40 * sizeof(std::uint32_t),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       VMA_MEMORY_USAGE_GPU_TO_CPU,
                                       "theia.sms.diagnostics");
    if (!diagnostics)
        return false;
    m_smsDiagnostics = std::move(*diagnostics);
    const std::array<std::uint32_t, 40> zeros{};
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
        // Final reservoirs survive across frames; temporal reservoirs are the
        // per-frame merge output consumed by resolve. The 320B SmsReservoir
        // stride is fixed by the shader ABI. The temporal dispatch is held
        // out until the residual dark/bright boundary weight amplification is
        // fixed (see PLAN); the reuse path then runs spatially only.
        if (m_cfg.smsSpatialReuse) {
            // SmsReservoir: 256B record + 16-byte-aligned tail = 320B stride,
            // mirrored from the shader ABI in Harmonia/shaders/sms_types.slang.
            constexpr VkDeviceSize kSmsReservoirStride = 320;
            if (pixels > (harmonia::pipeline::SmsPass::kMaxCapacity))
                return false;
            constexpr auto reservoirUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            auto temporal = harmonia::core::Buffer::create(*m_ctx,
                                                           pixels * kSmsReservoirStride,
                                                           reservoirUsage,
                                                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                           "theia.sms.reservoir.temporal");
            if (!temporal)
                return false;
            m_smsTemporalReservoir = std::move(*temporal);
            auto finalReservoir = harmonia::core::Buffer::create(*m_ctx,
                                                                 pixels * kSmsReservoirStride,
                                                                 reservoirUsage,
                                                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                                 "theia.sms.reservoir.final");
            if (!finalReservoir)
                return false;
            m_smsFinalReservoir = std::move(*finalReservoir);
        }
        const std::filesystem::path shared{HARMONIA_SHADER_DIR};
        auto sms =
            harmonia::pipeline::SmsPass::create(*m_ctx,
                                                static_cast<std::uint32_t>(pixels),
                                                m_pipelineLayout,
                                                shared / "sms_queue_control.spv",
                                                shaderPath("sms_sample.spv"),
                                                shaderPath("sms_resolve.spv"),
                                                m_cfg.smsTrialBudget,
                                                VK_SHADER_STAGE_COMPUTE_BIT,
                                                prior ? shaderPath("sms_prior_build.spv") : std::filesystem::path{},
                                                prior ? m_smsPrior.deviceAddress() : 0,
                                                // Dispatch held out: the temporal merge stores reservoir weights
                                                // contradicting its bounded inputs on frame one; root-cause first.
                                                std::filesystem::path{});
        if (!sms)
            return false;
        m_sms = std::move(*sms);
        context = {m_sms.workBuffer().deviceAddress(),
                   m_smsPrefixes.deviceAddress(),
                   m_cfg.smsPartitionResolution,
                   (m_cfg.smsSpatialReuse ? 1U : 0U) | (m_smsDiagEnabled ? 2U : 0U),
                   m_smsTemporalReservoir.isValid() ? m_smsTemporalReservoir.deviceAddress() : 0,
                   m_smsFinalReservoir.isValid() ? m_smsFinalReservoir.deviceAddress() : 0};
    }
    // Immutable configuration, uploaded only at initialization/resize. A disabled
    // context is real configuration, never a host-written hot-path work counter.
    auto buffer =
        harmonia::core::Buffer::upload(*m_ctx,
                                       uploadPool,
                                       std::as_bytes(std::span(&context, 1)),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       "theia.sms.context");
    if (!buffer)
        return false;
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
    std::array<std::uint32_t, 40> counters{};
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
        "maxSelectedLum={:.6g} maxPrevMassEarly={}",
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
        counters[39]);
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
