#ifndef THEIA_RENDERER_HIZPASS_HPP
#define THEIA_RENDERER_HIZPASS_HPP

#include <volk/volk.h>

#include <cstdint>
#include <vector>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/core/VulkanHandle.hpp"

namespace theia {

/// Hierarchical-Z (Hi-Z) depth pyramid builder for two-pass occlusion culling (B4).
///
/// After render pass 1 (which draws only meshlets that were visible last frame) has
/// produced the current frame's initial depth buffer, this pass reduces that depth into a
/// full max-depth mip chain stored in a separate R32F image. Render pass 2's mesh shader
/// then tests each remaining meshlet's bounding sphere against this CURRENT-frame Hi-Z and
/// skips meshlets fully behind already-rendered geometry.
///
/// Max reduction is the conservative choice: a sphere is culled only when its nearest depth
/// is behind the farthest occluder over its entire screen footprint, so visible geometry is
/// never wrongly culled.
///
/// Cross-vendor: core Vulkan compute + storage images only (no vendor extensions). If the
/// compute pipeline fails to initialize the pass reports !isInitialized() and callers fall
/// back to drawing all meshlets (backward compatible).
class HiZPass {
  public:
    HiZPass() = default;
    ~HiZPass();
    HiZPass(const HiZPass&) = delete;
    HiZPass& operator=(const HiZPass&) = delete;

    /// Create the Hi-Z image (mip chain sized to width/height) and the reduction pipeline.
    /// @param spvName SPIR-V filename resolved against THEIA_SHADER_DIR.
    [[nodiscard]] bool initialize(const DeviceContext& ctx,
                                  std::uint32_t width,
                                  std::uint32_t height,
                                  const char* spvName = "hiz_build.comp.spv");
    void shutdown();

    /// Ensure the Hi-Z image is in SHADER_READ_ONLY_OPTIMAL so the mesh shader may sample
    /// it during BOTH render passes. On first use this transitions from UNDEFINED (contents
    /// are undefined but pass 1 never samples them — it uses hiZMipCount = 0). No-op after.
    void prepareForSampling(VkCommandBuffer cmd) noexcept;

    /// Build the pyramid from the current depth buffer.
    /// Pre : depthView is a DEPTH-aspect view in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    ///       and the Hi-Z image is in SHADER_READ_ONLY_OPTIMAL (see prepareForSampling()).
    /// Post: the Hi-Z image is back in SHADER_READ_ONLY_OPTIMAL, all mips written and
    ///       visible to subsequent mesh-shader sampled reads.
    void build(VkCommandBuffer cmd, VkImageView depthView) noexcept;

    [[nodiscard]] bool isInitialized() const noexcept { return m_pipeline != VK_NULL_HANDLE; }
    [[nodiscard]] VkImageView sampledView() const noexcept {
        return m_image.isValid() ? m_image.view() : VK_NULL_HANDLE;
    }
    [[nodiscard]] VkImage image() const noexcept { return m_image.handle(); }
    [[nodiscard]] std::uint32_t mipLevels() const noexcept { return m_mipLevels; }
    [[nodiscard]] std::uint32_t width() const noexcept { return m_width; }
    [[nodiscard]] std::uint32_t height() const noexcept { return m_height; }

  private:
    struct HiZPC {
        std::uint32_t srcSize[2];
        std::uint32_t dstSize[2];
        std::uint32_t copyMode;
        std::uint32_t _pad;
    };

    [[nodiscard]] bool createImage() noexcept;
    [[nodiscard]] bool createPipeline(const char* spvName) noexcept;

    const DeviceContext* m_ctx = nullptr;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;
    std::uint32_t m_mipLevels = 1;

    Image m_image{};                                       ///< R32F max-depth pyramid (STORAGE + SAMPLED)
    std::vector<harmonia::UniqueImageView> m_mipViews;     ///< one single-level storage view per mip

    harmonia::UniqueDescriptorSetLayout m_setLayout;
    harmonia::UniquePipelineLayout m_pipelineLayout;
    harmonia::UniquePipeline m_pipeline;

    bool m_firstUse = true;
};

} // namespace theia
#endif // THEIA_RENDERER_HIZPASS_HPP
