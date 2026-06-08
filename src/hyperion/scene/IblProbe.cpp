#include "hyperion/scene/IblProbe.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/Logger.hpp"

#ifdef THEIA_HAS_OPENEXR
#include <Imath/ImathBox.h>
#include <OpenEXR/ImfRgbaFile.h>
#endif

IblProbe::IblProbe(IblProbe&& other) noexcept
    : m_image(std::move(other.m_image)),
      m_sampler(std::exchange(other.m_sampler, VK_NULL_HANDLE)),
      m_ctx(other.m_ctx),
      m_marginalCdf(std::move(other.m_marginalCdf)),
      m_conditionalCdf(std::move(other.m_conditionalCdf)),
      m_cdfWidth(std::exchange(other.m_cdfWidth, 0u)),
      m_cdfHeight(std::exchange(other.m_cdfHeight, 0u)),
      m_sunDirection(other.m_sunDirection),
      m_sunStrength(std::exchange(other.m_sunStrength, 0.0f)) {
    other.m_ctx = nullptr;
}

IblProbe& IblProbe::operator=(IblProbe&& other) noexcept {
    if (this != &other) {
        reset();
        m_image = std::move(other.m_image);
        m_sampler = std::exchange(other.m_sampler, VK_NULL_HANDLE);
        m_ctx = other.m_ctx;
        m_marginalCdf = std::move(other.m_marginalCdf);
        m_conditionalCdf = std::move(other.m_conditionalCdf);
        m_cdfWidth = std::exchange(other.m_cdfWidth, 0u);
        m_cdfHeight = std::exchange(other.m_cdfHeight, 0u);
        m_sunDirection = other.m_sunDirection;
        m_sunStrength = std::exchange(other.m_sunStrength, 0.0f);
        other.m_ctx = nullptr;
    }
    return *this;
}

IblProbe::~IblProbe() {
    reset();
}

void IblProbe::reset() noexcept {
    m_image = {};
    m_marginalCdf = {};
    m_conditionalCdf = {};
    m_cdfWidth = 0;
    m_cdfHeight = 0;
    if (m_ctx != nullptr && m_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_ctx->device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
    m_ctx = nullptr;
}

std::expected<IblProbe, VkResult>
IblProbe::loadFromEXR(const DeviceContext& ctx, const CommandPool& pool, const std::filesystem::path& path) {
#ifndef THEIA_HAS_OPENEXR
    (void)ctx;
    (void)pool;
    (void)path;
    Logger::error("IblProbe: OpenEXR support is not compiled in; cannot load '{}'", path.string());
    return std::unexpected(VK_ERROR_FEATURE_NOT_PRESENT);
#else
    // ── Load EXR pixels ──────────────────────────────────────────────────────
    int width = 0, height = 0;
    std::vector<float> rgba32f;

    try {
        using namespace OPENEXR_IMF_NAMESPACE;
        RgbaInputFile file(path.string().c_str());
        const IMATH_NAMESPACE::Box2i dw = file.dataWindow();
        width = dw.max.x - dw.min.x + 1;
        height = dw.max.y - dw.min.y + 1;

        std::vector<Rgba> halfs(static_cast<size_t>(width * height));
        file.setFrameBuffer(halfs.data() - dw.min.x - dw.min.y * width, 1, width);
        file.readPixels(dw.min.y, dw.max.y);

        // ── Convert half RGBA → float RGBA, with lin_srgb → lin_rec2020 ──────
        // Rec.709 → Rec.2020 primary transform (D65 white point, IEC 61966 / BT.2087)
        const float m00 = 0.6274040f, m01 = 0.3292820f, m02 = 0.0433140f;
        const float m10 = 0.0690970f, m11 = 0.9195400f, m12 = 0.0113630f;
        const float m20 = 0.0163916f, m21 = 0.0880132f, m22 = 0.8955950f;

        // Half-float max (65504): clamp before matrix multiply to stay finite.
        // EXR panoramas routinely store the sun disc as half-float inf (exponent=31,
        // mantissa=0) when the captured radiance exceeds the half-float range.
        // Clamping to kHalfMax keeps the sun visible and correctly weighted in the CDF
        // rather than zeroing it out, which would remove it from importance sampling.
        auto safeHalf = [](float v) -> float {
            constexpr float kHalfMax = 65504.0f;
            return std::isfinite(v) ? std::min(v, kHalfMax) : kHalfMax;
        };

        rgba32f.resize(static_cast<size_t>(width * height) * 4u);
        for (int i = 0; i < width * height; ++i) {
            const float r = safeHalf(static_cast<float>(halfs[i].r));
            const float g = safeHalf(static_cast<float>(halfs[i].g));
            const float b = safeHalf(static_cast<float>(halfs[i].b));
            rgba32f[i * 4 + 0] = m00 * r + m01 * g + m02 * b;
            rgba32f[i * 4 + 1] = m10 * r + m11 * g + m12 * b;
            rgba32f[i * 4 + 2] = m20 * r + m21 * g + m22 * b;
            rgba32f[i * 4 + 3] = safeHalf(static_cast<float>(halfs[i].a));
        }
    } catch (const std::exception& e) {
        Logger::error("IblProbe: failed to read '{}': {}", path.string(), e.what());
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    // ── Upload to GPU ────────────────────────────────────────────────────────
    const VkExtent2D extent{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(width * height) * 4u * sizeof(float);

    auto staging = Buffer::create(
        ctx, byteSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "ibl.staging");
    if (!staging) {
        return std::unexpected(staging.error());
    }
    staging->uploadData(rgba32f.data(), byteSize);

    auto image = Image::create(ctx,
                               extent,
                               VK_FORMAT_R32G32B32A32_SFLOAT,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               "ibl.env");
    if (!image) {
        return std::unexpected(image.error());
    }

    auto cmd = pool.beginOneShot();
    if (!cmd) {
        return std::unexpected(cmd.error());
    }

    image->transition(*cmd,
                      VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_NONE,
                      0,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT);

    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {extent.width, extent.height, 1u},
    };
    vkCmdCopyBufferToImage(*cmd, staging->handle(), image->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    image->transition(*cmd,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                      VK_ACCESS_2_SHADER_READ_BIT);

    if (const VkResult result = pool.endOneShot(*cmd); result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    // ── Create sampler (REPEAT on U, CLAMP_TO_EDGE on V to avoid pole artefacts) ──
    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler sampler = VK_NULL_HANDLE;
    if (const VkResult result = vkCreateSampler(ctx.device, &samplerInfo, nullptr, &sampler); result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    IblProbe probe;
    probe.m_ctx = &ctx;
    probe.m_image = std::move(*image);
    probe.m_sampler = sampler;

    // ── Build 2D separable CDF for env importance sampling ───────────────────
    // Ref: PBR Book 4th ed §12.5 "Infinite Area Lights" — 2D separable CDF construction
    // Resolution: 256×128 (each cell covers ~16×16 source pixels for a 4K panorama)
    static constexpr int kCdfW = 256;
    static constexpr int kCdfH = 128;
    const float kPiCpu = 3.14159265358979f;
    const float srcToGridU = static_cast<float>(kCdfW) / static_cast<float>(width);
    const float srcToGridV = static_cast<float>(kCdfH) / static_cast<float>(height);

    // Luminance grid: weighted by sin(θ) to account for equirectangular → solid-angle mapping
    std::vector<float> lumGrid(static_cast<size_t>(kCdfW * kCdfH));
    // Track the brightest (raw, unweighted) cell to extract a dominant "sun" direction
    // for ray-traced directional shadows, plus the mean to gauge how concentrated it is.
    float sunBestAvg = -1.0f;
    int sunBestU = 0, sunBestV = 0;
    double sunAvgSum = 0.0;
    int sunAvgCount = 0;
    for (int v = 0; v < kCdfH; ++v) {
        const float sinTheta = std::sin(kPiCpu * (static_cast<float>(v) + 0.5f) / static_cast<float>(kCdfH));
        for (int u = 0; u < kCdfW; ++u) {
            const int srcX0 = static_cast<int>(static_cast<float>(u) / srcToGridU);
            const int srcX1 = std::max(srcX0 + 1, static_cast<int>(static_cast<float>(u + 1) / srcToGridU));
            const int srcY0 = static_cast<int>(static_cast<float>(v) / srcToGridV);
            const int srcY1 = std::max(srcY0 + 1, static_cast<int>(static_cast<float>(v + 1) / srcToGridV));
            const int cX1 = std::min(srcX1, width);
            const int cY1 = std::min(srcY1, height);

            float sumLum = 0.0f;
            int count = 0;
            for (int sy = srcY0; sy < cY1; ++sy) {
                for (int sx = srcX0; sx < cX1; ++sx) {
                    const size_t idx = static_cast<size_t>(sy * width + sx) * 4u;
                    const float r = rgba32f[idx + 0];
                    const float g = rgba32f[idx + 1];
                    const float b = rgba32f[idx + 2];
                    // Rec.2020 luminance coefficients (ITU-R BT.2020)
                    const float lum = 0.2627f * r + 0.6780f * g + 0.0593f * b;
                    sumLum += (std::isfinite(lum) && lum > 0.0f) ? lum : 0.0f;
                    ++count;
                }
            }
            const float avgLum = (count > 0 ? sumLum / static_cast<float>(count) : 0.0f);
            sunAvgSum += avgLum;
            ++sunAvgCount;
            if (avgLum > sunBestAvg) {
                sunBestAvg = avgLum;
                sunBestU = u;
                sunBestV = v;
            }
            lumGrid[static_cast<size_t>(v * kCdfW + u)] = avgLum * sinTheta;
        }
    }

    // Convert the brightest grid cell into a world-space direction toward the sun.
    // Inverts the lat-long convention used by the shaders (env.slang):
    //   u = atan2(z,x)/(2π) + 0.5,  v = acos(y)/π  (v=0 at top, y=+1)
    glm::vec3 domSunDir{0.0f, 1.0f, 0.0f};
    float domSunStrength = 0.0f;
    {
        const float uNorm = (static_cast<float>(sunBestU) + 0.5f) / static_cast<float>(kCdfW);
        const float vNorm = (static_cast<float>(sunBestV) + 0.5f) / static_cast<float>(kCdfH);
        const float phi = (uNorm - 0.5f) * 2.0f * kPiCpu;
        const float theta = vNorm * kPiCpu;
        const float sinT = std::sin(theta);
        domSunDir = glm::normalize(glm::vec3(sinT * std::cos(phi), std::cos(theta), sinT * std::sin(phi)));
        // Concentration: ratio of the brightest cell to the mean. Uniform/overcast skies
        // give a ratio near 1 (no harsh shadows); a clear sun gives a very large ratio.
        const float meanAvg =
            (sunAvgCount > 0) ? static_cast<float>(sunAvgSum / static_cast<double>(sunAvgCount)) : 0.0f;
        const float ratio = (meanAvg > 1e-8f) ? (sunBestAvg / meanAvg) : 0.0f;
        domSunStrength = std::clamp((ratio - 4.0f) / 16.0f, 0.0f, 1.0f);
        Logger::info("IblProbe: dominant light dir ({:.2f}, {:.2f}, {:.2f}), strength {:.2f} (peak/mean {:.1f})",
                     domSunDir.x,
                     domSunDir.y,
                     domSunDir.z,
                     domSunStrength,
                     ratio);
    }
    probe.m_sunDirection = domSunDir;
    probe.m_sunStrength = domSunStrength;

    // Per-row conditional CDFs: conditionalCdf[v*(W+1)..(v+1)*(W+1)] for each row v
    std::vector<float> conditionalCdf(static_cast<size_t>(kCdfH * (kCdfW + 1)));
    std::vector<float> rowIntegrals(static_cast<size_t>(kCdfH), 0.0f);
    for (int v = 0; v < kCdfH; ++v) {
        float* rowCdf = conditionalCdf.data() + static_cast<ptrdiff_t>(v * (kCdfW + 1));
        rowCdf[0] = 0.0f;
        for (int u = 0; u < kCdfW; ++u) {
            rowCdf[u + 1] = rowCdf[u] + lumGrid[static_cast<size_t>(v * kCdfW + u)];
        }
        rowIntegrals[static_cast<size_t>(v)] = rowCdf[kCdfW];
        if (rowIntegrals[static_cast<size_t>(v)] > 0.0f) {
            const float invRow = 1.0f / rowIntegrals[static_cast<size_t>(v)];
            for (int u = 1; u <= kCdfW; ++u) {
                rowCdf[u] *= invRow;
            }
        } else {
            // Degenerate dark row: uniform distribution so binary search returns valid indices
            for (int u = 1; u <= kCdfW; ++u) {
                rowCdf[u] = static_cast<float>(u) / static_cast<float>(kCdfW);
            }
        }
        rowCdf[kCdfW] = 1.0f; // ensure exact 1.0
    }

    // Marginal CDF over rows: marginalCdf[H+1]
    std::vector<float> marginalCdf(static_cast<size_t>(kCdfH + 1));
    marginalCdf[0] = 0.0f;
    for (int v = 0; v < kCdfH; ++v) {
        marginalCdf[static_cast<size_t>(v + 1)] =
            marginalCdf[static_cast<size_t>(v)] + rowIntegrals[static_cast<size_t>(v)];
    }
    const float totalWeight = marginalCdf[static_cast<size_t>(kCdfH)];
    if (totalWeight > 0.0f) {
        const float invTotal = 1.0f / totalWeight;
        for (int v = 1; v <= kCdfH; ++v) {
            marginalCdf[static_cast<size_t>(v)] *= invTotal;
        }
        marginalCdf[static_cast<size_t>(kCdfH)] = 1.0f;

        // Upload marginal CDF buffer
        const VkDeviceSize margSize = static_cast<VkDeviceSize>(kCdfH + 1) * sizeof(float);
        auto mBuf = Buffer::create(
            ctx, margSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, "ibl.marginalCdf");
        if (mBuf) {
            mBuf->uploadData(marginalCdf.data(), margSize);
            probe.m_marginalCdf = std::move(*mBuf);
        }

        // Upload conditional CDF buffer
        const VkDeviceSize condSize = static_cast<VkDeviceSize>(kCdfH * (kCdfW + 1)) * sizeof(float);
        auto cBuf = Buffer::create(ctx,
                                   condSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                   "ibl.conditionalCdf");
        if (cBuf) {
            cBuf->uploadData(conditionalCdf.data(), condSize);
            probe.m_conditionalCdf = std::move(*cBuf);
        }

        if (probe.m_marginalCdf.isValid() && probe.m_conditionalCdf.isValid()) {
            probe.m_cdfWidth = static_cast<uint32_t>(kCdfW);
            probe.m_cdfHeight = static_cast<uint32_t>(kCdfH);
            Logger::info("IblProbe: built {}×{} importance CDF (total weight {:.2f})", kCdfW, kCdfH, totalWeight);
        }
    } else {
        Logger::warn("IblProbe: env map is completely dark — importance sampling disabled");
    }

    Logger::info("IblProbe: loaded '{}' ({}×{})", path.filename().string(), width, height);
    return probe;
#endif
}
