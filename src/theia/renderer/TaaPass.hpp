#ifndef THEIA_RENDERER_TAAPASS_HPP
#define THEIA_RENDERER_TAAPASS_HPP

#include <volk/volk.h>

#include <cstdint>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/core/VulkanHandle.hpp"

namespace theia {

/// Temporal Anti-Aliasing compute pass (cross-vendor KHR/EXT only).
///
/// Reads the current-frame HDR output + per-pixel motion vectors produced by
/// MotionVectorPass (A1b), applies YCoCg-space 3x3 neighbourhood AABB clamping
/// against the local colour neighbourhood, and blends with a history buffer
/// (alpha=0.1 → 90 % history, 10 % current).  The resolved result is copied
/// back into the shared HDR image so the Harmonia A-SVGF denoiser receives a
/// temporally stable, shimmer-free input.
///
/// Must run AFTER MotionVectorPass and BEFORE the Harmonia denoiser.
class TaaPass {
  public:
    struct Config {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        VkImage hdrImage = VK_NULL_HANDLE; ///< shared HDR image (R32G32B32A32_SFLOAT)
        VkImageView hdrView = VK_NULL_HANDLE;
        VkImageView motionVecView = VK_NULL_HANDLE; ///< from MotionVectorPass (R32G32_SFLOAT)
    };

    struct FrameParams {
        float alpha = 0.1f;      ///< current-frame blend weight (default 0.1)
        bool firstFrame = false; ///< when true the shader writes current directly
    };

    TaaPass() = default;
    ~TaaPass();
    TaaPass(const TaaPass&) = delete;
    TaaPass& operator=(const TaaPass&) = delete;

    [[nodiscard]] bool initialize(const harmonia::DeviceContext& ctx, const Config& cfg, const char* spvName = "taa.comp.spv");
    void shutdown();

    /// Dispatch TAA.
    /// Pre-condition:  hdrImage GENERAL (GiPass wrote), motionVecImage GENERAL.
    /// Post-condition: hdrImage GENERAL (TAA-resolved, visible to COMPUTE_SHADER).
    void record(VkCommandBuffer cmd, const FrameParams& params) noexcept;

    [[nodiscard]] bool isInitialized() const noexcept { return m_pipeline != VK_NULL_HANDLE; }

  private:
    struct TaaPushConstants {
        std::uint32_t screenWidth = 0;
        std::uint32_t screenHeight = 0;
        float alpha = 0.1f;
        std::uint32_t firstFrame = 0;
    };
    static_assert(sizeof(TaaPushConstants) == 16);

    [[nodiscard]] bool createImages() noexcept;
    [[nodiscard]] bool createSampler() noexcept;
    [[nodiscard]] bool createPipeline(const char* spvName) noexcept;

    const harmonia::DeviceContext* m_ctx = nullptr;
    Config m_cfg{};

    harmonia::UniqueSampler m_sampler;
    harmonia::Image m_history{};   ///< previous TAA output  (R32G32B32A32, SAMPLED | TRANSFER_DST)
    harmonia::Image m_taaOutput{}; ///< current TAA result   (R32G32B32A32, STORAGE | TRANSFER_SRC)
    bool m_firstUse = true;

    harmonia::UniqueDescriptorSetLayout m_setLayout;
    harmonia::UniquePipelineLayout m_pipeLayout;
    harmonia::UniquePipeline m_pipeline;

    std::uint32_t m_frameCount = 0;
};

} // namespace theia
#endif // THEIA_RENDERER_TAAPASS_HPP
