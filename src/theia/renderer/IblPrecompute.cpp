#include "theia/renderer/IblPrecompute.hpp"

#include <algorithm>
#include <array>
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
constexpr VkExtent2D kDiffuseExtent{256, 128};
constexpr VkExtent2D kSpecularExtent{1024, 512};
constexpr uint32_t kSpecularMipLevels = 8;

VkShaderModule loadShaderModule(const DeviceContext& ctx, const char* filename) {
    auto module = harmonia::createShaderModule(ctx.device, theia::shaderPath(filename));
    if (!module) {
        Logger::error("IblPrecompute: failed to load shader '{}'", filename);
        return VK_NULL_HANDLE;
    }
    return *module;
}

VkImageView createMipView(const DeviceContext& ctx, const Image& image, uint32_t mipLevel) {
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

// Fill an image with a small neutral ambient value instead of black.
// Used when no env_map is provided so scenes without an environment get a
// subtle ambient contribution rather than pure darkness.
void clearNeutralAmbientImage(VkCommandBuffer cmd, const Image& image) {
    image.transition(cmd,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE,
                     0,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Faint neutral ambient as a cheap stand-in for the single indirect bounce
    // that a path tracer (Hyperion) resolves for free.  Kept deliberately low so
    // it does NOT flatten directional contrast: in Hyperion's non-IBL scenes
    // surfaces that face away from the area light stay very dark (the front face
    // of the Cornell box is nearly black).  A large constant fill washes that
    // contrast out, so only a token amount of fill is applied here.
    static constexpr float kAmbient = 0.015f;
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

uint32_t dispatchCount(uint32_t extent) {
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
                               VkBuffer marginalCdf,
                               VkBuffer conditionalCdf,
                               uint32_t cdfWidth,
                               uint32_t cdfHeight) {
    shutdown();

    m_ctx = &ctx;
    m_pool = &pool;
    m_envImageView = envImageView;
    m_envSampler = envSampler;
    m_envUnitNits = (envUnitNits > 0.0f) ? envUnitNits : 1.0f;
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

    if (!runPass(&IblPrecompute::runSheenLutPass, "sheen LUT") ||
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

    if (m_ctx != nullptr && m_ctx->device != VK_NULL_HANDLE) {
        if (m_res.lutSampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_ctx->device, m_res.lutSampler, nullptr);
            m_res.lutSampler = VK_NULL_HANDLE;
        }
        if (m_res.envSampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_ctx->device, m_res.envSampler, nullptr);
            m_res.envSampler = VK_NULL_HANDLE;
        }
    }

    m_res.sheenLut = {};
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

    auto diffuse =
        Image::create(*m_ctx,
                      kDiffuseExtent,
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
    if (vkCreateSampler(m_ctx->device, &lutSamplerInfo, nullptr, &m_res.lutSampler) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create LUT sampler");
        return false;
    }

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
    if (vkCreateSampler(m_ctx->device, &envSamplerInfo, nullptr, &m_res.envSampler) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create environment sampler");
        return false;
    }

    return true;
}

bool IblPrecompute::runSheenLutPass(VkCommandBuffer cmd) {
    const VkDescriptorSetLayoutBinding binding{
        0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(m_ctx->device, &layoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create sheen set layout");
        return false;
    }
    m_tempSetLayouts.push_back(setLayout);

    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create sheen descriptor pool");
        return false;
    }
    m_tempDescriptorPools.push_back(descriptorPool);

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (!createComputePipeline("ibl_sheen_lut.comp.spv", setLayout, 0, pipeline, pipelineLayout)) {
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
        Logger::error("IblPrecompute: failed to allocate sheen descriptor set");
        return false;
    }

    m_res.sheenLut.transition(cmd,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_NONE,
                              0,
                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                              VK_ACCESS_2_SHADER_WRITE_BIT);

    const VkDescriptorImageInfo outputInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = m_res.sheenLut.view(),
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

    m_res.sheenLut.transition(cmd,
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
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(m_ctx->device, &layoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create diffuse set layout");
        return false;
    }
    m_tempSetLayouts.push_back(setLayout);

    const std::array<VkDescriptorPoolSize, 4> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER,        1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1},
        // Storage buffers (marginal + conditional CDF) counted together:
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create diffuse descriptor pool");
        return false;
    }
    m_tempDescriptorPools.push_back(descriptorPool);

    struct DiffusePC {
        float    envScale;
        uint32_t cdfWidth;
        uint32_t cdfHeight;
        uint32_t _pad{0};
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
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                             0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &envImageInfo,    nullptr, nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                             1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER,        &envSamplerInfo,  nullptr, nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                             2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &outIrradInfo,    nullptr, nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                             3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &marginalInfo,    nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptorSet,
                             4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &conditionalInfo, nullptr},
    };
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    const DiffusePC pc{m_envUnitNits, m_cdfWidth, m_cdfHeight, 0};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, dispatchCount(kDiffuseExtent.width), dispatchCount(kDiffuseExtent.height), 1);

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

    const std::array<VkDescriptorSetLayoutBinding, 3> bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(m_ctx->device, &layoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create specular set layout");
        return false;
    }
    m_tempSetLayouts.push_back(setLayout);

    const std::array<VkDescriptorPoolSize, 3> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        Logger::error("IblPrecompute: failed to create specular descriptor pool");
        return false;
    }
    m_tempDescriptorPools.push_back(descriptorPool);

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (!createComputePipeline("ibl_specular.comp.spv", setLayout, sizeof(float) * 2, pipeline, pipelineLayout)) {
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
        Logger::error("IblPrecompute: failed to allocate specular descriptor set");
        return false;
    }

    m_res.specularMipped.transition(cmd,
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    VK_PIPELINE_STAGE_2_NONE,
                                    0,
                                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_WRITE_BIT);

    const std::array<VkDescriptorImageInfo, 2> staticInfos{
        VkDescriptorImageInfo{VK_NULL_HANDLE, m_envImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        VkDescriptorImageInfo{m_envSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},
    };
    const std::array<VkWriteDescriptorSet, 2> staticWrites{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             descriptorSet,
                             0,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &staticInfos[0],
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             descriptorSet,
                             1,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLER,
                             &staticInfos[1],
                             nullptr,
                             nullptr},
    };
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(staticWrites.size()), staticWrites.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    for (uint32_t mip = 0; mip < m_res.specularMipped.mipLevels(); ++mip) {
        VkImageView mipView = createMipView(*m_ctx, m_res.specularMipped, mip);
        if (mipView == VK_NULL_HANDLE) {
            return false;
        }
        m_tempImageViews.push_back(mipView);

        const VkDescriptorImageInfo outputInfo{
            .sampler = VK_NULL_HANDLE,
            .imageView = mipView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkWriteDescriptorSet outputWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &outputInfo,
        };
        vkUpdateDescriptorSets(m_ctx->device, 1, &outputWrite, 0, nullptr);

        const float roughness = (m_res.specularMipped.mipLevels() > 1)
                                    ? static_cast<float>(mip) / static_cast<float>(m_res.specularMipped.mipLevels() - 1)
                                    : 0.0f;
        const uint32_t mipWidth = std::max(1u, kSpecularExtent.width >> mip);
        const uint32_t mipHeight = std::max(1u, kSpecularExtent.height >> mip);

        const struct {
            float roughness;
            float envScale;
        } pc{roughness, m_envUnitNits};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
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
                                          uint32_t pushConstantSize,
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
    m_tempPipelineLayouts.push_back(outLayout);

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
    m_tempPipelines.push_back(outPipeline);

    vkDestroyShaderModule(m_ctx->device, module, nullptr);
    return true;
}

void IblPrecompute::destroyTemporaryObjects() noexcept {
    if (m_ctx == nullptr || m_ctx->device == VK_NULL_HANDLE) {
        m_tempDescriptorPools.clear();
        m_tempSetLayouts.clear();
        m_tempPipelineLayouts.clear();
        m_tempPipelines.clear();
        m_tempImageViews.clear();
        return;
    }

    for (VkImageView view : m_tempImageViews) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_ctx->device, view, nullptr);
        }
    }
    m_tempImageViews.clear();

    for (VkPipeline pipeline : m_tempPipelines) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_ctx->device, pipeline, nullptr);
        }
    }
    m_tempPipelines.clear();

    for (VkPipelineLayout layout : m_tempPipelineLayouts) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_ctx->device, layout, nullptr);
        }
    }
    m_tempPipelineLayouts.clear();

    for (VkDescriptorPool pool : m_tempDescriptorPools) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_ctx->device, pool, nullptr);
        }
    }
    m_tempDescriptorPools.clear();

    for (VkDescriptorSetLayout setLayout : m_tempSetLayouts) {
        if (setLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_ctx->device, setLayout, nullptr);
        }
    }
    m_tempSetLayouts.clear();
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
