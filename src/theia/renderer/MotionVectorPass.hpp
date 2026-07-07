#pragma once

#include <volk/volk.h>

#include <slang-math/slang-math.hpp>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Image.hpp"

namespace theia {

/// Compute pass that produces per-pixel motion vectors in screen space.
///
/// Reads world-space primary hit positions from the forward GI G-buffer and
/// transforms them with the current and previous view-projection matrices to
/// compute pixel-space (dx, dy) deltas.  The output is consumed by the shared
/// Harmonia denoiser for temporal reprojection so camera motion no longer
/// ghosts / smears the accumulated history.
///
/// Must run AFTER GiPass (the GI G-buffer must be in SHADER_READ_ONLY_OPTIMAL)
/// and BEFORE the Harmonia denoiser pass.
class MotionVectorPass {
  public:
    struct Config {
        uint32_t width = 0;
        uint32_t height = 0;
        VkImage giBufferImage = VK_NULL_HANDLE;    ///< GI G-buffer: xyz = worldPos, w = matIdx+1
        VkImageView giBufferView = VK_NULL_HANDLE;
    };

    /// Per-frame matrices and buffer handles passed each frame.
    struct FrameParams {
        /// row-major view-projection matrix for the current frame.
        sm::float4x4 curViewProj{1.0f};
        /// row-major view-projection matrix for the previous frame.
        sm::float4x4 prevViewProj{1.0f};
        /// Per-instance previous-frame world transforms (one sm::float4x4 per instance).
        /// Must remain valid until record() returns.  May be VK_NULL_HANDLE on the first
        /// frame before the scene is loaded (record() is a no-op in that case).
        VkBuffer prevInstanceTransformBuffer = VK_NULL_HANDLE;
    };

    MotionVectorPass() = default;
    ~MotionVectorPass();
    MotionVectorPass(const MotionVectorPass&) = delete;
    MotionVectorPass& operator=(const MotionVectorPass&) = delete;

    [[nodiscard]] bool initialize(const DeviceContext& ctx, const Config& cfg,
                                  const char* spvName = "motion_vector.comp.spv");
    void shutdown();

    /// Record the motion-vector dispatch into cmd.
    /// Pre-condition: giBuffer is in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    /// Post-condition: motionVectorImage is in VK_IMAGE_LAYOUT_GENERAL (writes
    ///                 made visible to subsequent COMPUTE_SHADER reads).
    void record(VkCommandBuffer cmd, const FrameParams& params) noexcept;

    [[nodiscard]] bool isInitialized() const noexcept { return m_pipeline != VK_NULL_HANDLE; }

    /// View of the R32G32_SFLOAT motion vector image (VK_IMAGE_LAYOUT_GENERAL).
    [[nodiscard]] VkImageView motionVectorImageView() const noexcept {
        return m_motionVectorImage.isValid() ? m_motionVectorImage.view() : VK_NULL_HANDLE;
    }
    /// Raw image handle, used by Application for cross-queue ownership transfers.
    [[nodiscard]] VkImage motionVectorImageHandle() const noexcept {
        return m_motionVectorImage.isValid() ? m_motionVectorImage.handle() : VK_NULL_HANDLE;
    }

  private:
    struct alignas(16) MotionVectorPC {
        sm::float4x4 curViewProj{1.0f};   // 64 bytes — row-major view-projection matrix
        sm::float4x4 prevViewProj{1.0f};  // 64 bytes — row-major view-projection matrix
        // Total: 128 bytes (Vulkan minimum push-constant size)
    };
    static_assert(sizeof(MotionVectorPC) == 128);

    [[nodiscard]] bool createImage() noexcept;
    [[nodiscard]] bool createPipeline(const char* spvName) noexcept;

    const DeviceContext* m_ctx = nullptr;
    Config m_cfg{};

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    Image m_motionVectorImage{};
    bool m_firstUse = true;
};

} // namespace theia
