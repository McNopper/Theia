#pragma once

#include <volk/volk.h>

#include "hyperion/utils/OutputColorSpace.hpp"

class Image; // forward — avoids pulling in the full header

/// Per-frame context passed to every IRenderPass::record() call.
/// Each pass reads only the fields it needs.
struct PassContext {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    uint32_t frameIndex = 0;
    VkExtent2D extent = {};

    /// Outputs from PathTracer — all in VK_IMAGE_LAYOUT_GENERAL.
    const Image* hdrBuffer = nullptr; ///< accumulated HDR radiance
    const Image* gNormal = nullptr;   ///< world-space normal G-buffer
    const Image* gDepth = nullptr;    ///< ray-hit distance G-buffer

    /// Denoiser output (null if denoiser was not run; ToneMapper falls back to hdrBuffer).
    const Image* denoised = nullptr;

    /// Swapchain image view for the current frame (only needed by ToneMapper).
    VkImageView swapchainView = VK_NULL_HANDLE;
    OutputColorSpace colorSpace = OutputColorSpace::eSDR;
};
