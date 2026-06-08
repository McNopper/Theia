#pragma once

#include <volk/volk.h>

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/core/Image.hpp"

/// Source color space of a texture asset.
/// Names follow the OCIO / OpenEXR IIF color space registry.
/// On load, all color data is converted to linear Rec.2020 (the render color space).
/// Data maps (normal, ORM, roughness) use Raw — no color conversion is applied.
enum class TextureColorSpace : uint8_t {
    Raw = 0,     ///< "raw"          — uninterpreted data; no conversion (normal, ORM, roughness)
    SrgbTexture, ///< "srgb_texture" — sRGB OETF + Rec.709 primaries   → linear Rec.2020
    LinSrgb,     ///< "lin_srgb"     — linear Rec.709 primaries         → linear Rec.2020
    LinRec2020,  ///< "lin_rec2020"  — already in render color space; no conversion
    AcesCg,      ///< "acescg"       — ACEScg / lin_ap1                 → linear Rec.2020
};

/// Parse an OCIO/OpenEXR color space name string.
/// Returns Raw for unrecognised strings (safe fallback for data maps).
[[nodiscard]] inline TextureColorSpace parseTextureColorSpace(std::string_view name) noexcept {
    if (name == "srgb_texture")
        return TextureColorSpace::SrgbTexture;
    if (name == "lin_srgb")
        return TextureColorSpace::LinSrgb;
    if (name == "lin_rec2020")
        return TextureColorSpace::LinRec2020;
    if (name == "acescg" || name == "lin_ap1")
        return TextureColorSpace::AcesCg;
    return TextureColorSpace::Raw;
}

class Texture {
  public:
    Texture() = default;
    ~Texture() noexcept;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    /// Upload pre-decoded pixels (RGBA8, already in render color space).
    [[nodiscard]] static std::expected<Texture, VkResult> create(const DeviceContext& ctx,
                                                                 const CommandPool& cmdPool,
                                                                 std::span<const std::byte> pixels,
                                                                 uint32_t width,
                                                                 uint32_t height,
                                                                 std::string_view name = "");

    /// Load a texture from a file and convert it to linear Rec.2020 at load time.
    /// colorSpace describes how the source data is encoded; the correct decode +
    /// primaries conversion is applied on the CPU before GPU upload.
    [[nodiscard]] static std::expected<Texture, VkResult>
    loadFromFile(const DeviceContext& ctx,
                 const CommandPool& cmdPool,
                 const std::filesystem::path& path,
                 TextureColorSpace colorSpace = TextureColorSpace::SrgbTexture,
                 std::string_view name = "");

    [[nodiscard]] const Image& image() const noexcept { return m_image; }
    [[nodiscard]] VkSampler sampler() const noexcept { return m_sampler; }
    [[nodiscard]] uint32_t width() const noexcept { return m_width; }
    [[nodiscard]] uint32_t height() const noexcept { return m_height; }
    [[nodiscard]] uint32_t mipLevels() const noexcept { return m_mipLevels; }

  private:
    void reset() noexcept;

    const DeviceContext* m_ctx{};
    Image m_image{};
    VkSampler m_sampler{VK_NULL_HANDLE};
    uint32_t m_width{};
    uint32_t m_height{};
    uint32_t m_mipLevels{1};
};
