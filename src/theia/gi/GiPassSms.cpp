#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>

#include "harmonia/core/Logger.hpp"
#include "theia/gi/GiPass.hpp"
#include "theia/renderer/ShaderPath.hpp"

namespace theia {
namespace {
struct SmsProducerContext {
    VkDeviceAddress work, prefixes;
    std::uint32_t resolution, flags;
};
static_assert(sizeof(SmsProducerContext) == 24);
} // namespace

bool GiPass::createSmsResources(const harmonia::core::CommandPool& uploadPool) {
    SmsProducerContext context{};
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
                                                prior ? m_smsPrior.deviceAddress() : 0);
        if (!sms)
            return false;
        m_sms = std::move(*sms);
        context = {m_sms.workBuffer().deviceAddress(),
                   m_smsPrefixes.deviceAddress(),
                   m_cfg.smsPartitionResolution,
                   m_cfg.smsSpatialReuse ? 1U : 0U};
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
    for (auto& writer : m_descWriters)
        writer.writeStorageBufferHandle(*m_ctx, 25, m_smsContext);
    return true;
}

std::expected<bool, VkResult> GiPass::frameReady() const noexcept {
    if (m_frameError != VK_SUCCESS)
        return std::unexpected(m_frameError);
    if (!m_frameRecorded)
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    if (!m_smsFrameActive)
        return true;
    const auto completion = m_sms.completion();
    if (!completion) {
        m_frameError = completion.error();
        return std::unexpected(m_frameError);
    }
    if (completion->status == harmonia::pipeline::SmsQueueCompletion::Failed) {
        harmonia::core::Logger::error("Theia SMS logical frame failed: GPU error bits {}", completion->error);
        m_frameError = VK_ERROR_UNKNOWN;
        return std::unexpected(m_frameError);
    }
    return completion->status == harmonia::pipeline::SmsQueueCompletion::Ready;
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
