#include "theia/renderer/IblPrecompute.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "harmonia/core/Logger.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "theia/renderer/ShaderPath.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace {

constexpr VkExtent2D kSheenExtent{512, 512};
constexpr VkExtent2D kSpecularExtent{1024, 512};
constexpr std::uint32_t kSpecularMipLevels = 8;

VkShaderModule loadShaderModule(const DeviceContext& ctx, const char* filename) {
    auto module = harmonia::createShaderModule(ctx.device, theia::shaderPath(filename));
    if (!module) {
        Logger::error("IblPrecompute: failed to load shader '{}'", filename);
        return VK_NULL_HANDLE;
    }
    return *module;
}

VkImageView createMipView(const DeviceContext& ctx, const Image& image, std::uint32_t mipLevel) {
    const VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image.handle(),
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = image.format(),
        .components =
            VkComponentMapping{
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
        .subresourceRange =
            VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = mipLevel,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };

    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create mip image view {}", mipLevel);
        return VK_NULL_HANDLE;
    }
    return view;
}

// Fill an image with black when no env_map is provided.
// Direct-light-only scenes must remain free of synthetic ambient terms so
// Hyperion/Theia parity comparisons stay physically consistent.
void clearNeutralAmbientImage(VkCommandBuffer cmd, const Image& image) {
    image.transition(cmd,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE,
                     0,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);

    static constexpr float kAmbient = 0.0f;
    const VkClearColorValue clearValue{{kAmbient, kAmbient, kAmbient, 1.0f}};
    const VkImageSubresourceRange range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = image.mipLevels(),
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    vkCmdClearColorImage(cmd, image.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);

    image.transition(cmd,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT);
}

std::uint32_t dispatchCount(std::uint32_t extent) {
    return (extent + 7u) / 8u;
}

} // namespace

namespace theia {

IblPrecompute::~IblPrecompute() {
    shutdown();
}

bool IblPrecompute::initialize(const DeviceContext& ctx,
                               const CommandPool& pool,
                               VkImageView envImageView,
                               VkSampler envSampler,
                               float envUnitNits,
                               VkExtent2D diffuseExtent,
                               VkBuffer marginalCdf,
                               VkBuffer conditionalCdf,
                               std::uint32_t cdfWidth,
                               std::uint32_t cdfHeight) {
    shutdown();

    m_ctx = &ctx;
    m_pool = &pool;
    m_envImageView = envImageView;
    m_envSampler = envSampler;
    m_envUnitNits = (envUnitNits > 0.0f) ? envUnitNits : 1.0f;
    m_diffuseExtent = diffuseExtent;
    m_marginalCdf = marginalCdf;
    m_conditionalCdf = conditionalCdf;
    m_cdfWidth = cdfWidth;
    m_cdfHeight = cdfHeight;

    if (!createTextures() || !createSamplers()) {
        shutdown();
        return false;
    }

    const auto runPass = [this, &pool](bool (IblPrecompute::*pass)(VkCommandBuffer), const char* label) -> bool {
        auto cmdResult = pool.beginOneShot();
        if (!cmdResult) {
            Logger::error("IblPrecompute: failed to allocate command buffer for {}", label);
            return false;
        }

        if (!(this->*pass)(*cmdResult)) {
            pool.free(*cmdResult);
            destroyTemporaryObjects();
            Logger::error("IblPrecompute: {} pass failed", label);
            return false;
        }

        if (const VkResult result = pool.endOneShot(*cmdResult); result != VK_SUCCESS) {
            destroyTemporaryObjects();
            Logger::error("IblPrecompute: {} submission failed ({})", label, static_cast<int>(result));
            return false;
        }

        destroyTemporaryObjects();
        return true;
    };

    if (!runPass(&IblPrecompute::runBrdfLutPass, "BRDF LUT") ||
        !runPass(&IblPrecompute::runSheenLutPass, "sheen LUT") ||
        !runPass(&IblPrecompute::runDiffusePass, "diffuse irradiance") ||
        !runPass(&IblPrecompute::runSpecularPass, "specular prefilter")) {
        shutdown();
        return false;
    }

    m_initialized = true;
    Logger::info("IBL precompute complete");
    return true;
}

void IblPrecompute::shutdown() {
    destroyTemporaryObjects();

    m_res.lutSampler.reset();
    m_res.envSampler.reset();

    m_res.sheenLut = {};
    m_res.brdfLut = {};
    m_res.diffuseIrrad = {};
    m_res.specularMipped = {};

    m_ctx = nullptr;
    m_pool = nullptr;
    m_envImageView = VK_NULL_HANDLE;
    m_envSampler = VK_NULL_HANDLE;
    m_initialized = false;
}

bool IblPrecompute::createTextures() {
    auto sheen = Image::create(*m_ctx,
                               kSheenExtent,
                               VK_FORMAT_R16_SFLOAT,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               "theia.ibl.sheen");
    if (!sheen) {
        Logger::error("IblPrecompute: failed to create sheen LUT image");
        return false;
    }
    m_res.sheenLut = std::move(*sheen);

    auto brdf = Image::create(*m_ctx,
                              kSheenExtent,
                              VK_FORMAT_R16G16_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT,
                              "theia.ibl.brdf");
    if (!brdf) {
        Logger::error("IblPrecompute: failed to create BRDF LUT image");
        return false;
    }
    m_res.brdfLut = std::move(*brdf);

    auto diffuse =
        Image::create(*m_ctx,
                      m_diffuseExtent,
                      VK_FORMAT_R16G16B16A16_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT,
                      "theia.ibl.diffuse");
    if (!diffuse) {
        Logger::error("IblPrecompute: failed to create diffuse irradiance image");
        return false;
    }
    m_res.diffuseIrrad = std::move(*diffuse);

    auto specular =
        Image::create(*m_ctx,
                      kSpecularExtent,
                      VK_FORMAT_R16G16B16A16_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT,
                      "theia.ibl.specular",
                      kSpecularMipLevels);
    if (!specular) {
        Logger::error("IblPrecompute: failed to create specular prefilter image");
        return false;
    }
    m_res.specularMipped = std::move(*specular);

    return true;
}

bool IblPrecompute::createSamplers() {
    const VkSamplerCreateInfo lutSamplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
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
    VkSampler lutSampler = VK_NULL_HANDLE;
    if (vkCreateSampler(m_ctx->device, &lutSamplerInfo, nullptr, &lutSampler) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create LUT sampler");
        return false;
    }
    m_res.lutSampler = harmonia::UniqueSampler{m_ctx->device, lutSampler};

    const VkSamplerCreateInfo envSamplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
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
        .maxLod = static_cast<float>(kSpecularMipLevels - 1),
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler envSampler = VK_NULL_HANDLE;
    if (vkCreateSampler(m_ctx->device, &envSamplerInfo, nullptr, &envSampler) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create environment sampler");
        return false;
    }
    m_res.envSampler = harmonia::UniqueSampler{m_ctx->device, envSampler};

    return true;
}

bool IblPrecompute::runBrdfLutPass(VkCommandBuffer cmd) {
    return runLutPass(cmd, m_res.brdfLut, "ibl_brdf_lut.comp.spv", "BRDF LUT");
}

bool IblPrecompute::runSheenLutPass(VkCommandBuffer cmd) {
    return runLutPass(cmd, m_res.sheenLut, "ibl_sheen_lut.comp.spv", "sheen");
}

bool IblPrecompute::runLutPass(VkCommandBuffer cmd, Image& targetImage, const char* shaderName, const char* logLabel) {
    const VkDescriptorSetLayoutBinding binding{
        0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(m_ctx->device, &layoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create {} set layout", logLabel);
        return false;
    }
    m_tempSetLayouts.emplace_back(m_ctx->device, setLayout);

    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create {} descriptor pool", logLabel);
        return false;
    }
    m_tempDescriptorPools.emplace_back(m_ctx->device, descriptorPool);

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (!createComputePipeline(shaderName, setLayout, 0, pipeline, pipelineLayout)) {
        return false;
    }

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &setLayout,
    };
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_ctx->device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to allocate {} descriptor set", logLabel);
        return false;
    }

    targetImage.transition(cmd,
                           VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_PIPELINE_STAGE_2_NONE,
                           0,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT);

    const VkDescriptorImageInfo outputInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = targetImage.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptorSet,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &outputInfo,
    };
    vkUpdateDescriptorSets(m_ctx->device, 1, &write, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(cmd, dispatchCount(kSheenExtent.width), dispatchCount(kSheenExtent.height), 1);

    targetImage.transition(cmd,
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_2_SHADER_READ_BIT);
    return true;
}

bool IblPrecompute::runDiffusePass(VkCommandBuffer cmd) {
    if (m_envImageView == VK_NULL_HANDLE || m_envSampler == VK_NULL_HANDLE) {
        clearNeutralAmbientImage(cmd, m_res.diffuseIrrad);
        return true;
    }

    // Bindings: 0=envMap (SAMPLED_IMAGE), 1=envSampler (SAMPLER),
    //           2=outIrrad (STORAGE_IMAGE),
    //           3=marginalCdf (STORAGE_BUFFER), 4=conditionalCdf (STORAGE_BUFFER)
    const std::array<VkDescriptorSetLayoutBinding, 5> bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(m_ctx->device, &layoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create diffuse set layout");
        return false;
    }
    m_tempSetLayouts.emplace_back(m_ctx->device, setLayout);

    const std::array<VkDescriptorPoolSize, 4> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        // Storage buffers (marginal + conditional CDF) counted together:
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create diffuse descriptor pool");
        return false;
    }
    m_tempDescriptorPools.emplace_back(m_ctx->device, descriptorPool);

    struct DiffusePC {
        float envScale{};
        std::uint32_t cdfWidth{};
        std::uint32_t cdfHeight{};
        std::uint32_t _pad{0};
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (!createComputePipeline("ibl_diffuse.comp.spv", setLayout, sizeof(DiffusePC), pipeline, pipelineLayout)) {
        return false;
    }

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &setLayout,
    };
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_ctx->device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to allocate diffuse descriptor set");
        return false;
    }

    m_res.diffuseIrrad.transition(cmd,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_NONE,
                                  0,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_WRITE_BIT);

    const VkDescriptorImageInfo envImageInfo{VK_NULL_HANDLE, m_envImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo envSamplerInfo{m_envSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    const VkDescriptorImageInfo outIrradInfo{VK_NULL_HANDLE, m_res.diffuseIrrad.view(), VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorBufferInfo marginalInfo{m_marginalCdf, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo conditionalInfo{m_conditionalCdf, 0, VK_WHOLE_SIZE};

    const std::array<VkWriteDescriptorSet, 5> writes{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             descriptorSet,
                             0,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &envImageInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             descriptorSet,
                             1,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLER,
                             &envSamplerInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             descriptorSet,
                             2,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                             &outIrradInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             descriptorSet,
                             3,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &marginalInfo,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             descriptorSet,
                             4,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             nullptr,
                             &conditionalInfo,
                             nullptr},
    };
    vkUpdateDescriptorSets(m_ctx->device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    const DiffusePC pc{m_envUnitNits, m_cdfWidth, m_cdfHeight, 0};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, dispatchCount(m_diffuseExtent.width), dispatchCount(m_diffuseExtent.height), 1);

    m_res.diffuseIrrad.transition(cmd,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_READ_BIT);
    return true;
}

bool IblPrecompute::runSpecularPass(VkCommandBuffer cmd) {
    if (m_envImageView == VK_NULL_HANDLE || m_envSampler == VK_NULL_HANDLE) {
        clearNeutralAmbientImage(cmd, m_res.specularMipped);
        return true;
    }

    const std::uint32_t mipCount = m_res.specularMipped.mipLevels();

    const std::array<VkDescriptorSetLayoutBinding, 3> bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(m_ctx->device, &layoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create specular set layout");
        return false;
    }
    m_tempSetLayouts.emplace_back(m_ctx->device, setLayout);

    // One descriptor set per mip level — all sets are fully written before
    // any CB recording begins, so no descriptor set is ever updated while bound.
    const std::array<VkDescriptorPoolSize, 3> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, mipCount},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, mipCount},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, mipCount},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = mipCount,
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create specular descriptor pool");
        return false;
    }
    m_tempDescriptorPools.emplace_back(m_ctx->device, descriptorPool);

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (!createComputePipeline("ibl_specular.comp.spv", setLayout, sizeof(float) * 2, pipeline, pipelineLayout)) {
        return false;
    }

    // Allocate one descriptor set per mip and fully write all descriptors
    // before recording any commands — no descriptor may be updated while bound.
    std::vector<VkDescriptorSetLayout> layouts(mipCount, setLayout);
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = mipCount,
        .pSetLayouts = layouts.data(),
    };
    std::vector<VkDescriptorSet> mipSets(mipCount, VK_NULL_HANDLE);
    if (vkAllocateDescriptorSets(m_ctx->device, &allocInfo, mipSets.data()) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to allocate specular descriptor sets");
        return false;
    }

    const VkDescriptorImageInfo envImageInfo{VK_NULL_HANDLE, m_envImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo envSamplerInfo{m_envSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};

    for (std::uint32_t mip = 0; mip < mipCount; ++mip) {
        VkImageView mipView = createMipView(*m_ctx, m_res.specularMipped, mip);
        if (mipView == VK_NULL_HANDLE) {
            return false;
        }
        m_tempImageViews.emplace_back(m_ctx->device, mipView);

        const VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, mipView, VK_IMAGE_LAYOUT_GENERAL};
        const std::array<VkWriteDescriptorSet, 3> writes{
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 nullptr,
                                 mipSets[mip],
                                 0,
                                 0,
                                 1,
                                 VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                 &envImageInfo,
                                 nullptr,
                                 nullptr},
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 nullptr,
                                 mipSets[mip],
                                 1,
                                 0,
                                 1,
                                 VK_DESCRIPTOR_TYPE_SAMPLER,
                                 &envSamplerInfo,
                                 nullptr,
                                 nullptr},
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 nullptr,
                                 mipSets[mip],
                                 2,
                                 0,
                                 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                 &outputInfo,
                                 nullptr,
                                 nullptr},
        };
        vkUpdateDescriptorSets(m_ctx->device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // All descriptors are now written — safe to record commands.
    m_res.specularMipped.transition(cmd,
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    VK_PIPELINE_STAGE_2_NONE,
                                    0,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    for (std::uint32_t mip = 0; mip < mipCount; ++mip) {
        const float roughness = (mipCount > 1) ? static_cast<float>(mip) / static_cast<float>(mipCount - 1) : 0.0f;
        const std::uint32_t mipWidth = std::max(1u, kSpecularExtent.width >> mip);
        const std::uint32_t mipHeight = std::max(1u, kSpecularExtent.height >> mip);

        // cppcheck-suppress unusedStructMember; both fields are read by the GPU via vkCmdPushConstants
        const struct {
            float roughness;
            float envScale;
        } pc{roughness, m_envUnitNits};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &mipSets[mip], 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, dispatchCount(mipWidth), dispatchCount(mipHeight), 1);
    }

    m_res.specularMipped.transition(cmd,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_READ_BIT);
    return true;
}

bool IblPrecompute::createComputePipeline(const char* spirvPath,
                                          VkDescriptorSetLayout layout,
                                          std::uint32_t pushConstantSize,
                                          VkPipeline& outPipeline,
                                          VkPipelineLayout& outLayout) {
    VkShaderModule module = loadShaderModule(*m_ctx, spirvPath);
    if (module == VK_NULL_HANDLE) {
        return false;
    }

    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = pushConstantSize,
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &layout,
        .pushConstantRangeCount = pushConstantSize > 0 ? 1u : 0u,
        .pPushConstantRanges = pushConstantSize > 0 ? &pushConstantRange : nullptr,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(m_ctx->device, module, nullptr);
        Logger::error("IblPrecompute: failed to create compute pipeline layout for '{}'", spirvPath);
        return false;
    }
    m_tempPipelineLayouts.emplace_back(m_ctx->device, outLayout);

    const VkPipelineShaderStageCreateInfo stageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = module,
        .pName = "main",
    };
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = outLayout,
    };
    if (vkCreateComputePipelines(m_ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &outPipeline) !=
        VK_SUCCESS) {
        vkDestroyShaderModule(m_ctx->device, module, nullptr);
        Logger::error("IblPrecompute: failed to create compute pipeline for '{}'", spirvPath);
        return false;
    }
    m_tempPipelines.emplace_back(m_ctx->device, outPipeline);

    vkDestroyShaderModule(m_ctx->device, module, nullptr);
    return true;
}

void IblPrecompute::destroyTemporaryObjects() noexcept {
    m_tempImageViews.clear();
    m_tempPipelines.clear();
    m_tempPipelineLayouts.clear();
    m_tempDescriptorPools.clear();
    m_tempSetLayouts.clear();
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
