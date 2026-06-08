#include "hyperion/scene/Texture.hpp"

#include <volk/volk.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stb_image.h>
#include <utility>
#include <vma/vk_mem_alloc.h>

#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/Logger.hpp"
#include "hyperion/utils/ColorSpace.hpp"

namespace {
[[nodiscard]] std::expected<Buffer, VkResult>
createStagingBuffer(const DeviceContext& ctx, std::span<const std::byte> pixels, std::string_view name) {
    auto staging = Buffer::create(ctx,
                                  static_cast<VkDeviceSize>(pixels.size()),
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  std::string(name).append(".staging"));
    if (!staging) {
        return std::unexpected(staging.error());
    }

    staging->uploadData(pixels.data(), pixels.size(), 0);
    return std::move(*staging);
}
} // namespace

Texture::~Texture() noexcept {
    reset();
}

Texture::Texture(Texture&& other) noexcept
    : m_ctx(other.m_ctx),
      m_image(std::move(other.m_image)),
      m_sampler(std::exchange(other.m_sampler, VK_NULL_HANDLE)),
      m_width(std::exchange(other.m_width, 0)),
      m_height(std::exchange(other.m_height, 0)),
      m_mipLevels(std::exchange(other.m_mipLevels, 1)) {
    other.m_ctx = nullptr;
}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this != &other) {
        reset();
        m_ctx = other.m_ctx;
        m_image = std::move(other.m_image);
        m_sampler = std::exchange(other.m_sampler, VK_NULL_HANDLE);
        m_width = std::exchange(other.m_width, 0);
        m_height = std::exchange(other.m_height, 0);
        m_mipLevels = std::exchange(other.m_mipLevels, 1);
        other.m_ctx = nullptr;
    }
    return *this;
}

std::expected<Texture, VkResult> Texture::create(const DeviceContext& ctx,
                                                 const CommandPool& cmdPool,
                                                 std::span<const std::byte> pixels,
                                                 uint32_t width,
                                                 uint32_t height,
                                                 std::string_view name) {
    if (width == 0 || height == 0 || pixels.empty()) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    auto image = Image::create(ctx,
                               VkExtent2D{width, height},
                               format,
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               name);
    if (!image) {
        return std::unexpected(image.error());
    }

    auto staging = createStagingBuffer(ctx, pixels, name);
    if (!staging) {
        return std::unexpected(staging.error());
    }

    auto cmd = cmdPool.beginOneShot();
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
        .imageSubresource =
            VkImageSubresourceLayers{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = VkOffset3D{0, 0, 0},
        .imageExtent = VkExtent3D{width, height, 1},
    };
    vkCmdCopyBufferToImage(*cmd, staging->handle(), image->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    image->transition(*cmd,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    if (const VkResult submitResult = cmdPool.endOneShot(*cmd); submitResult != VK_SUCCESS) {
        return std::unexpected(submitResult);
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);

    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = std::min(16.0f, props.limits.maxSamplerAnisotropy),
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    Texture texture;
    texture.m_ctx = &ctx;
    texture.m_image = std::move(*image);
    texture.m_width = width;
    texture.m_height = height;

    if (const VkResult result = vkCreateSampler(ctx.device, &samplerInfo, nullptr, &texture.m_sampler);
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    if (!name.empty()) {
        ctx.setDebugName(VK_OBJECT_TYPE_SAMPLER,
                         reinterpret_cast<uint64_t>(texture.m_sampler),
                         std::string(name).append(".sampler").c_str());
    }

    return texture;
}

std::expected<Texture, VkResult> Texture::loadFromFile(const DeviceContext& ctx,
                                                       const CommandPool& cmdPool,
                                                       const std::filesystem::path& path,
                                                       TextureColorSpace colorSpace,
                                                       std::string_view name) {
    const std::string pathStr = path.string();
    int w = 0, h = 0, srcChannels = 0;

    // Always load as 4-channel RGBA, 8 bits per channel.
    stbi_uc* raw = stbi_load(pathStr.c_str(), &w, &h, &srcChannels, 4);
    if (!raw) {
        Logger::error("Texture::loadFromFile: stbi_load failed for '{}': {}", pathStr, stbi_failure_reason());
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    const auto pixelCount = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<uint8_t> converted(pixelCount * 4);

    const bool needsConversion = (colorSpace != TextureColorSpace::Raw && colorSpace != TextureColorSpace::LinRec2020);

    if (!needsConversion) {
        // Raw data (normal/ORM/roughness) or already Rec.2020 — copy verbatim.
        std::memcpy(converted.data(), raw, pixelCount * 4);
    } else {
        // Convert each pixel to linear Rec.2020.
        for (size_t i = 0; i < pixelCount; ++i) {
            const float r = raw[i * 4 + 0] / 255.0f;
            const float g = raw[i * 4 + 1] / 255.0f;
            const float b = raw[i * 4 + 2] / 255.0f;
            const uint8_t a = raw[i * 4 + 3];

            glm::vec3 linear{r, g, b};
            switch (colorSpace) {
            case TextureColorSpace::SrgbTexture:
                // sRGB OETF decode + Rec.709 → Rec.2020 primaries.
                linear = ColorSpace::srgbAssetToRec2020(linear);
                break;
            case TextureColorSpace::LinSrgb:
                // Already linear Rec.709 — only primaries conversion needed.
                linear = ColorSpace::rec709ToRec2020(linear);
                break;
            case TextureColorSpace::AcesCg:
                linear = ColorSpace::acesCgToRec2020(linear);
                break;
            default:
                break;
            }

            converted[i * 4 + 0] = static_cast<uint8_t>(std::clamp(linear.r, 0.0f, 1.0f) * 255.0f + 0.5f);
            converted[i * 4 + 1] = static_cast<uint8_t>(std::clamp(linear.g, 0.0f, 1.0f) * 255.0f + 0.5f);
            converted[i * 4 + 2] = static_cast<uint8_t>(std::clamp(linear.b, 0.0f, 1.0f) * 255.0f + 0.5f);
            converted[i * 4 + 3] = a;
        }
    }

    stbi_image_free(raw);

    const auto bytes = std::as_bytes(std::span<const uint8_t>(converted));
    return create(ctx,
                  cmdPool,
                  bytes,
                  static_cast<uint32_t>(w),
                  static_cast<uint32_t>(h),
                  name.empty() ? path.filename().string() : std::string(name));
}

void Texture::reset() noexcept {
    if (m_ctx != nullptr && m_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_ctx->device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
    m_ctx = nullptr;
    m_width = 0;
    m_height = 0;
    m_mipLevels = 1;
}
