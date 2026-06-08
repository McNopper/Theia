#pragma once

#include <volk/volk.h>

#include <glm/glm.hpp>

#include <expected>
#include <filesystem>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/core/Image.hpp"

/// Image-based lighting probe loaded from an equirectangular HDR panorama (EXR).
///
/// The panorama is stored on the GPU as a 2D RGBA32F texture in linear Rec.2020.
/// Input EXR files are assumed to be in linear sRGB (Rec.709 primaries, D65)
/// and are converted to linear Rec.2020 at load time.
///
/// A 2D separable CDF (256×128) is also built from the panorama luminance for
/// environment map importance sampling (env NEE + MIS).
/// Ref: Pharr, Jakob & Humphreys — "Physically Based Rendering" 4th ed. §12.5 & §13.4.3
class IblProbe {
  public:
    IblProbe() = default;
    IblProbe(const IblProbe&) = delete;
    IblProbe& operator=(const IblProbe&) = delete;
    IblProbe(IblProbe&& other) noexcept;
    IblProbe& operator=(IblProbe&& other) noexcept;
    ~IblProbe();

    /// Load an equirectangular EXR panorama, convert to linear Rec.2020, and upload to GPU.
    /// Requires HYPERION_HAS_OPENEXR; returns VK_ERROR_FEATURE_NOT_PRESENT otherwise.
    [[nodiscard]] static std::expected<IblProbe, VkResult>
    loadFromEXR(const DeviceContext& ctx, const CommandPool& pool, const std::filesystem::path& path);

    [[nodiscard]] VkImageView imageView() const noexcept { return m_image.view(); }
    [[nodiscard]] VkSampler sampler() const noexcept { return m_sampler; }
    [[nodiscard]] bool isValid() const noexcept { return m_sampler != VK_NULL_HANDLE; }

    /// CDF buffers for 2D env importance sampling (valid when cdfWidth() > 0).
    [[nodiscard]] const Buffer& marginalCdfBuffer() const noexcept { return m_marginalCdf; }
    [[nodiscard]] const Buffer& conditionalCdfBuffer() const noexcept { return m_conditionalCdf; }
    [[nodiscard]] uint32_t cdfWidth() const noexcept { return m_cdfWidth; }
    [[nodiscard]] uint32_t cdfHeight() const noexcept { return m_cdfHeight; }

    /// Dominant ("sun") direction extracted from the brightest region of the panorama,
    /// expressed as a normalised world-space direction pointing *towards* the light.
    /// Used to drive ray-traced directional shadows for IBL-lit scenes.
    [[nodiscard]] glm::vec3 sunDirection() const noexcept { return m_sunDirection; }

    /// Relative strength of the dominant light in [0,1]: 0 for a uniform/overcast
    /// panorama (no crisp shadows), approaching 1 for a clear, concentrated sun.
    [[nodiscard]] float sunStrength() const noexcept { return m_sunStrength; }

  private:
    void reset() noexcept;

    Image m_image{};
    VkSampler m_sampler{VK_NULL_HANDLE};
    const DeviceContext* m_ctx{};

    Buffer m_marginalCdf{};    ///< (H+1) floats: normalised marginal CDF over rows
    Buffer m_conditionalCdf{}; ///< H*(W+1) floats: normalised conditional CDF per row
    uint32_t m_cdfWidth{0};
    uint32_t m_cdfHeight{0};

    glm::vec3 m_sunDirection{0.0f, 1.0f, 0.0f}; ///< world-space direction towards the dominant light
    float m_sunStrength{0.0f};                  ///< [0,1] concentration of the dominant light
};
