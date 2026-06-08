#pragma once

#include <cstdint>

/// Display output color space / HDR transfer function.
///
/// Priority order used by Swapchain for automatic selection:
///   eHDR10 > eHLG > eScRGB > eDisplayP3 > eSDRLinear > eSDR
///
/// Mapping to Vulkan surface format pairs:
///   eHDR10     — A2B10G10R10_UNORM  + VK_COLOR_SPACE_HDR10_ST2084_EXT
///   eHLG       — A2B10G10R10_UNORM  + VK_COLOR_SPACE_HDR10_HLG_EXT
///   eScRGB     — R16G16B16A16_SFLOAT + VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT
///   eDisplayP3 — B8G8R8A8_UNORM     + VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT
///   eSDRLinear — R16G16B16A16_SFLOAT + VK_COLOR_SPACE_BT709_LINEAR_EXT
///   eSDR       — B8G8R8A8_UNORM     + VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
enum class OutputColorSpace : uint32_t {
    eHDR10 = 0,     ///< ST.2084 PQ OETF, Rec.2020 primaries, 10 000 nit absolute reference
    eHLG = 1,       ///< BT.2100 HLG OETF, Rec.2020 primaries, relative (1 000 nit peak)
    eScRGB = 2,     ///< Extended linear scRGB, Rec.709 primaries, no OETF (Windows HDR)
    eDisplayP3 = 3, ///< Display P3 (DCI-P3 D65 primaries, sRGB-like OETF)
    eSDRLinear = 4, ///< Linear BT.709, clamped [0, 1]
    eSDR = 5,       ///< sRGB / BT.709, gamma-encoded — always available fallback
};
