#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "theia/renderer/ForwardRenderer.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <harmonia/core/Logger.hpp>
#include <harmonia/core/ShaderModule.hpp>
#include <theia/renderer/ShaderPath.hpp>
#include <theia/scene/Scene.hpp>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

namespace theia {

ForwardRenderer::~ForwardRenderer() {
    shutdown();
}

bool ForwardRenderer::initialize(const DeviceContext& ctx, const Config& config) {
    m_config = config;
    m_ctx = &ctx;

    if (!createDepthTarget()) {
        Logger::error("Failed to create depth target");
        return false;
    }

    if (!createPipeline()) {
        Logger::error("Failed to create pipeline");
        return false;
    }

    // Create 1-element dummy buffers for tile light slots (fallback when LightCuller hasn't run).
    constexpr VkDeviceSize kDummySize = sizeof(uint32_t) * 128; // >= kMaxLightsPerTile
    auto dummyCounts = Buffer::create(ctx,
                                      kDummySize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      "theia.dummyTileCounts");
    auto dummyIndices = Buffer::create(ctx,
                                       kDummySize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                       "theia.dummyTileIndices");
    if (dummyCounts && dummyIndices) {
        m_dummyTileCounts = std::move(*dummyCounts);
        m_dummyTileIndices = std::move(*dummyIndices);
    }

    m_initialized = true;
    m_hdrFirstUse = true;
    Logger::info("GPU-driven forward renderer initialized ({}x{})", config.width, config.height);
    return true;
}

void ForwardRenderer::shutdown() {
    if (!m_ctx) {
        return;
    }

    vkDeviceWaitIdle(m_ctx->device);

    if (m_graphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx->device, m_graphicsPipeline, nullptr);
        m_graphicsPipeline = VK_NULL_HANDLE;
    }
    if (m_skyPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx->device, m_skyPipeline, nullptr);
        m_skyPipeline = VK_NULL_HANDLE;
    }
    if (m_skyPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_ctx->device, m_skyPipelineLayout, nullptr);
        m_skyPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_ctx->device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_meshSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
    }
    if (m_matSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
    }
    if (m_iblSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
    }
    if (m_textureSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
    }
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }

    m_iblDiffuseInfo = {};
    m_iblSpecularInfo = {};
    m_sheenLutInfo = {};
    m_iblEnvSamplerInfo = {};

    m_dummyTileCounts = {};
    m_dummyTileIndices = {};

    m_meshSet = VK_NULL_HANDLE;
    m_matSet = VK_NULL_HANDLE;
    m_iblSet = VK_NULL_HANDLE;
    m_textureSet = VK_NULL_HANDLE;
    m_texturesBoundFor = nullptr;
    m_depthTarget = {};
    m_gbufferTarget = {};
    m_ctx = nullptr;

    m_initialized = false;
}

void ForwardRenderer::setTileBuffers(VkBuffer tileLightCounts,
                                     VkBuffer tileLightIndices,
                                     uint32_t tilesX,
                                     uint32_t tilesY) {
    m_tileLightCountsBuf = tileLightCounts;
    m_tileLightIndicesBuf = tileLightIndices;
    m_tilesX = tilesX;
    m_tilesY = tilesY;
}

void ForwardRenderer::setIbl(const IblResources& res, VkImageView rawEnvView, float envUnitNits) {
    if (!m_ctx || m_iblSet == VK_NULL_HANDLE) {
        return;
    }

    m_iblDiffuseInfo =
        VkDescriptorImageInfo{VK_NULL_HANDLE, res.diffuseIrrad.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_iblSpecularInfo =
        VkDescriptorImageInfo{VK_NULL_HANDLE, res.specularMipped.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_sheenLutInfo =
        VkDescriptorImageInfo{VK_NULL_HANDLE, res.sheenLut.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_iblEnvSamplerInfo = VkDescriptorImageInfo{res.envSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    // Raw env panorama for the sky background. Fall back to the specular view (a valid
    // SAMPLED_IMAGE) when no raw env is supplied, so the descriptor stays valid.
    const VkImageView envView = (rawEnvView != VK_NULL_HANDLE) ? rawEnvView : res.specularMipped.view();
    m_iblEnvRawInfo = VkDescriptorImageInfo{VK_NULL_HANDLE, envView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    m_envUnitNits = envUnitNits;

    const std::array<VkWriteDescriptorSet, 5> writes{
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             0,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_iblDiffuseInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             1,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_iblSpecularInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             2,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_sheenLutInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             3,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLER,
                             &m_iblEnvSamplerInfo,
                             nullptr,
                             nullptr},
        VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                             nullptr,
                             m_iblSet,
                             4,
                             0,
                             1,
                             VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                             &m_iblEnvRawInfo,
                             nullptr,
                             nullptr},
    };
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool ForwardRenderer::createDepthTarget() {
    VkExtent2D extent{m_config.width, m_config.height};
    // SAMPLED_BIT so SSR can read depth after the forward pass.
    auto depthResult = Image::create(*m_ctx,
                                     extent,
                                     VK_FORMAT_D32_SFLOAT,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                     VK_IMAGE_ASPECT_DEPTH_BIT,
                                     "theia.depth");
    if (!depthResult) {
        Logger::error("Failed to create depth target");
        return false;
    }
    m_depthTarget = std::move(*depthResult);

    // GBuffer: RGBA16F — view-space normal (xyz * 0.5 + 0.5) + roughness (w).
    // SAMPLED_BIT so SSR compute can sample it; STORAGE_BIT reserved for future post-process.
    auto gbufResult = Image::create(*m_ctx,
                                    extent,
                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                    "theia.gbuffer");
    if (!gbufResult) {
        Logger::error("Failed to create GBuffer target");
        return false;
    }
    m_gbufferTarget = std::move(*gbufResult);
    return true;
}

bool ForwardRenderer::createPipeline() {
    auto loadShaderModule = [this](const char* filename) -> VkShaderModule {
        auto module = harmonia::createShaderModule(m_ctx->device, shaderPath(filename));
        if (!module) {
            Logger::error("Failed to load shader: {}", filename);
            return VK_NULL_HANDLE;
        }
        return *module;
    };

    VkShaderModule taskModule = loadShaderModule("forward_render.task.spv");
    VkShaderModule meshModule = loadShaderModule("forward_render.mesh.spv");
    VkShaderModule fragModule = loadShaderModule("forward_render.frag.spv");
    if (!taskModule || !meshModule || !fragModule) {
        if (taskModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        }
        if (meshModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        }
        if (fragModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        }
        return false;
    }

    constexpr VkShaderStageFlags kTaskAndMeshStages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(ForwardRenderer::MeshPushConstants),
    };

    // Set 0: geometry buffers (vertices, instances, indices, meshlets, meshletVerts, meshletTris)
    const std::array<VkDescriptorSetLayoutBinding, 6> meshBindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskAndMeshStages, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskAndMeshStages, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskAndMeshStages, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo meshSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(meshBindings.size()),
        .pBindings = meshBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &meshSetLayoutInfo, nullptr, &m_meshSetLayout) != VK_SUCCESS) {
        Logger::error("Failed to create mesh descriptor set layout");
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // Set 1: material/lighting buffers (materials, lights, emissive triangles, tile lists — fragment stage)
    // Binding 5 = scene TLAS for ray-traced shadows (ray query in the fragment shader).
    const std::array<VkDescriptorSetLayoutBinding, 6> matBindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{
            3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // tileLightCounts
        VkDescriptorSetLayoutBinding{
            4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // tileLightIndices
        VkDescriptorSetLayoutBinding{
            5, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, // scene TLAS
    };
    const VkDescriptorSetLayoutCreateInfo matSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(matBindings.size()),
        .pBindings = matBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &matSetLayoutInfo, nullptr, &m_matSetLayout) != VK_SUCCESS) {
        Logger::error("Failed to create material descriptor set layout");
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // Set 2: IBL textures + sampler (fragment stage)
    // binding 0: t_iblDiffuse, 1: t_iblSpecular, 2: t_sheenLut, 3: s_iblLinear, 4: t_envRaw (sky)
    const std::array<VkDescriptorSetLayoutBinding, 5> iblBindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo iblSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(iblBindings.size()),
        .pBindings = iblBindings.data(),
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &iblSetLayoutInfo, nullptr, &m_iblSetLayout) != VK_SUCCESS) {
        Logger::error("Failed to create IBL descriptor set layout");
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // Set 3: bindless material textures (variable array of combined image samplers, fragment stage).
    // Uses partially-bound descriptors so only the slots actually populated per scene must be valid.
    const VkDescriptorSetLayoutBinding textureBinding{
        0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    const VkDescriptorBindingFlags textureBindingFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    const VkDescriptorSetLayoutBindingFlagsCreateInfo textureBindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = 1,
        .pBindingFlags = &textureBindingFlags,
    };
    const VkDescriptorSetLayoutCreateInfo textureSetLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &textureBindingFlagsInfo,
        .bindingCount = 1,
        .pBindings = &textureBinding,
    };
    if (vkCreateDescriptorSetLayout(m_ctx->device, &textureSetLayoutInfo, nullptr, &m_textureSetLayout) != VK_SUCCESS) {
        Logger::error("Failed to create bindless texture descriptor set layout");
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // Single pool for all four sets.
    const std::array<VkDescriptorPoolSize, 5> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 4,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    if (vkCreateDescriptorPool(m_ctx->device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        Logger::error("Failed to create descriptor pool");
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    const std::array<VkDescriptorSetLayout, 4> setLayouts{
        m_meshSetLayout, m_matSetLayout, m_iblSetLayout, m_textureSetLayout};
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts = setLayouts.data(),
    };
    std::array<VkDescriptorSet, 4> sets{};
    if (vkAllocateDescriptorSets(m_ctx->device, &allocInfo, sets.data()) != VK_SUCCESS) {
        Logger::error("Failed to allocate descriptor sets");
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }
    m_meshSet = sets[0];
    m_matSet = sets[1];
    m_iblSet = sets[2];
    m_textureSet = sets[3];

    const std::array<VkDescriptorSetLayout, 4> pipelineSetLayouts{
        m_meshSetLayout, m_matSetLayout, m_iblSetLayout, m_textureSetLayout};
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(pipelineSetLayouts.size()),
        .pSetLayouts = pipelineSetLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        Logger::error("Failed to create graphics pipeline layout");
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_textureSetLayout, nullptr);
        m_textureSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    const std::array<VkPipelineShaderStageCreateInfo, 3> graphicsStages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
            .module = taskModule,
            .pName = "main",
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
            .module = meshModule,
            .pName = "main",
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragModule,
            .pName = "main",
        },
    };

    const VkPipelineViewportStateCreateInfo viewport{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineColorBlendAttachmentState colorAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    // GBuffer (target 1) blend state — write all, no blend
    const VkPipelineColorBlendAttachmentState gbufferAttachment = colorAttachment;
    const std::array<VkPipelineColorBlendAttachmentState, 2> colorAttachments{colorAttachment, gbufferAttachment};
    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(colorAttachments.size()),
        .pAttachments = colorAttachments.data(),
    };
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    const std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };
    constexpr VkFormat kGbufferFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    const std::array<VkFormat, 2> colorFormats{m_config.outputFormat, kGbufferFormat};
    const VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()),
        .pColorAttachmentFormats = colorFormats.data(),
        .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
    };

    const VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering,
        .stageCount = static_cast<uint32_t>(graphicsStages.size()),
        .pStages = graphicsStages.data(),
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = nullptr,
        .pViewportState = &viewport,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = m_pipelineLayout,
    };
    if (vkCreateGraphicsPipelines(m_ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) !=
        VK_SUCCESS) {
        Logger::error("Failed to create graphics pipeline");
        vkDestroyPipelineLayout(m_ctx->device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_ctx->device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_iblSetLayout, nullptr);
        m_iblSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_matSetLayout, nullptr);
        m_matSetLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_ctx->device, m_meshSetLayout, nullptr);
        m_meshSetLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
        return false;
    }

    // ── Sky/background pipeline ───────────────────────────────────────────────
    // Fullscreen triangle that samples the environment panorama (or outputs black).
    // Reuses the IBL descriptor set layout (set 0 == specular prefilter + sampler).
    {
        VkShaderModule skyVert = loadShaderModule("sky.vert.spv");
        VkShaderModule skyFrag = loadShaderModule("sky.frag.spv");
        if (skyVert == VK_NULL_HANDLE || skyFrag == VK_NULL_HANDLE) {
            Logger::error("Failed to load sky shader modules");
            if (skyVert)
                vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
            if (skyFrag)
                vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
            vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
            return false;
        }

        const VkPushConstantRange skyPcRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(SkyPushConstants),
        };
        const VkPipelineLayoutCreateInfo skyLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_iblSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &skyPcRange,
        };
        if (vkCreatePipelineLayout(m_ctx->device, &skyLayoutInfo, nullptr, &m_skyPipelineLayout) != VK_SUCCESS) {
            Logger::error("Failed to create sky pipeline layout");
            vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
            vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
            vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
            return false;
        }

        const std::array<VkPipelineShaderStageCreateInfo, 2> skyStages{
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = skyVert,
                .pName = "main",
            },
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = skyFrag,
                .pName = "main",
            },
        };
        const VkPipelineInputAssemblyStateCreateInfo skyInputAsm{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        // No depth test/write — drawn first; geometry overwrites where present.
        const VkPipelineDepthStencilStateCreateInfo skyDepth{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_FALSE,
            .depthWriteEnable = VK_FALSE,
            .depthCompareOp = VK_COMPARE_OP_ALWAYS,
        };
        // Color 0 (HDR) writes RGBA; color 1 (GBuffer) is left untouched (keeps clear).
        const VkPipelineColorBlendAttachmentState skyHdrBlend{
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT,
        };
        const VkPipelineColorBlendAttachmentState skyGbufBlend{.blendEnable = VK_FALSE, .colorWriteMask = 0};
        const std::array<VkPipelineColorBlendAttachmentState, 2> skyBlends{skyHdrBlend, skyGbufBlend};
        const VkPipelineColorBlendStateCreateInfo skyColorBlend{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = static_cast<uint32_t>(skyBlends.size()),
            .pAttachments = skyBlends.data(),
        };

        const VkGraphicsPipelineCreateInfo skyPipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering,
            .stageCount = static_cast<uint32_t>(skyStages.size()),
            .pStages = skyStages.data(),
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &skyInputAsm,
            .pViewportState = &viewport,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &skyDepth,
            .pColorBlendState = &skyColorBlend,
            .pDynamicState = &dynamicState,
            .layout = m_skyPipelineLayout,
        };
        const VkResult skyRes =
            vkCreateGraphicsPipelines(m_ctx->device, VK_NULL_HANDLE, 1, &skyPipelineInfo, nullptr, &m_skyPipeline);
        vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
        vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
        if (skyRes != VK_SUCCESS) {
            Logger::error("Failed to create sky pipeline");
            vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
            vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
            return false;
        }
    }

    vkDestroyShaderModule(m_ctx->device, taskModule, nullptr);
    vkDestroyShaderModule(m_ctx->device, meshModule, nullptr);
    vkDestroyShaderModule(m_ctx->device, fragModule, nullptr);
    return true;
}

void ForwardRenderer::recordFrame(VkCommandBuffer cmd) {
    if (!m_scene || m_config.hdrImage == VK_NULL_HANDLE) {
        return;
    }

    const std::array imageBarriers{
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = m_hdrFirstUse ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = m_hdrFirstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_config.hdrImage,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        },
        // Depth: always discard (UNDEFINED) — depth is fully cleared each frame.
        // SAMPLED_BIT added so SSR can read it after the pass.
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_depthTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
        },
        // GBuffer: always discard (UNDEFINED) — fully written each frame.
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_gbufferTarget.handle(),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        },
    };
    const VkDependencyInfo hdrDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &hdrDep);
    m_hdrFirstUse = false;

    // Create buffer info structures for descriptor updates
    VkDescriptorBufferInfo vertexBufferInfo{
        .buffer = m_scene->vertexBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo instanceBufferInfo{
        .buffer = m_scene->instanceBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo indexBufferInfo{
        .buffer = m_scene->indexBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo meshletBufferInfo{
        .buffer = m_scene->meshletBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo meshletVertexBufferInfo{
        .buffer = m_scene->meshletVertexBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo meshletTriangleBufferInfo{
        .buffer = m_scene->meshletTriangleBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    VkDescriptorBufferInfo materialBufferInfo{
        .buffer = m_scene->materialBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo lightBufferInfo{
        .buffer = m_scene->lightBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo emissiveTriangleBufferInfo{
        .buffer = m_scene->emissiveTriangleBuffer().handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    // Tile light lists: use real buffers if LightCuller ran, else dummy fallback.
    const VkBuffer tileCntBuf = (m_tileLightCountsBuf != VK_NULL_HANDLE && m_dummyTileCounts.handle() != VK_NULL_HANDLE)
                                    ? m_tileLightCountsBuf
                                    : m_dummyTileCounts.handle();
    const VkBuffer tileIdxBuf =
        (m_tileLightIndicesBuf != VK_NULL_HANDLE && m_dummyTileIndices.handle() != VK_NULL_HANDLE)
            ? m_tileLightIndicesBuf
            : m_dummyTileIndices.handle();
    const bool hasTileData = (tileCntBuf != VK_NULL_HANDLE && tileIdxBuf != VK_NULL_HANDLE);
    VkDescriptorBufferInfo tileLightCountsInfo{
        .buffer = hasTileData ? tileCntBuf : m_dummyTileCounts.handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    VkDescriptorBufferInfo tileLightIndicesInfo{
        .buffer = hasTileData ? tileIdxBuf : m_dummyTileIndices.handle(),
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    const VkAccelerationStructureKHR sceneTlas = m_scene->tlas();
    const VkWriteDescriptorSetAccelerationStructureKHR tlasWriteInfo{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &sceneTlas,
    };

    std::array<VkWriteDescriptorSet, 12> writes{
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &vertexBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &instanceBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &indexBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &meshletBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &meshletVertexBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_meshSet,
            .dstBinding = 5,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &meshletTriangleBufferInfo,
        },
        // Set 1: materials / lights / emissive triangles
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &materialBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &lightBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &emissiveTriangleBufferInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &tileLightCountsInfo,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_matSet,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &tileLightIndicesInfo,
        },
        // Set 1, binding 5: scene TLAS for ray-traced shadows.
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = &tlasWriteInfo,
            .dstSet = m_matSet,
            .dstBinding = 5,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        },
    };
    vkUpdateDescriptorSets(m_ctx->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Set 3: bindless material textures. Written only when the scene changes (textures are
    // immutable per scene). Each scene texture carries its own view + sampler.
    if (m_scene != m_texturesBoundFor) {
        const auto& sceneTextures = m_scene->textures();
        const uint32_t texCount = std::min(static_cast<uint32_t>(sceneTextures.size()), kMaxBindlessTextures);
        if (texCount > 0) {
            std::vector<VkDescriptorImageInfo> imageInfos(texCount);
            for (uint32_t i = 0; i < texCount; ++i) {
                imageInfos[i] = VkDescriptorImageInfo{
                    .sampler = sceneTextures[i].sampler(),
                    .imageView = sceneTextures[i].image().view(),
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                };
            }
            const VkWriteDescriptorSet texWrite{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = m_textureSet,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = texCount,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = imageInfos.data(),
            };
            vkUpdateDescriptorSets(m_ctx->device, 1, &texWrite, 0, nullptr);
        }
        m_texturesBoundFor = m_scene;
    }

    // Set dynamic state
    VkViewport vp{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(m_config.width),
        .height = static_cast<float>(m_config.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{{0, 0}, {m_config.width, m_config.height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Begin dynamic rendering (Vulkan 1.4 - no render pass!)
    VkClearColorValue clearCol{{0.0f, 0.0f, 0.0f, 1.0f}};
    // GBuffer clear: encoded normal (0.5,0.5,0.5) = view-space (0,0,0), roughness=1 → background
    VkClearColorValue clearGbuf{{0.5f, 0.5f, 0.5f, 1.0f}};
    VkClearDepthStencilValue clearDepth{1.0f, 0};

    VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = m_config.hdrImageView, // externally owned HDR image
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue{.color = clearCol},
    };

    VkRenderingAttachmentInfo gbufferAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = m_gbufferTarget.view(),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE, // must store for SSR to read
        .clearValue{.color = clearGbuf},
    };

    VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = m_depthTarget.view(),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE, // must store for SSR depth ray march
        .clearValue{.depthStencil = clearDepth},
    };

    const std::array<VkRenderingAttachmentInfo, 2> colorAttachments{colorAttachment, gbufferAttachment};
    VkRenderingInfo rendering{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea{{0, 0}, {m_config.width, m_config.height}},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
        .pColorAttachments = colorAttachments.data(),
        .pDepthAttachment = &depthAttachment,
    };

    vkCmdBeginRendering(cmd, &rendering);

    // Draw the environment sky background first (only when the scene has an env map).
    // It fills every pixel at the far plane with no depth write; opaque geometry then
    // overwrites it. Scenes without an env keep the black color clear.
    if (m_hasEnv && m_skyPipeline != VK_NULL_HANDLE) {
        glm::mat4 skyProj = glm::perspective(glm::radians(m_camera.vfovDeg),
                                             static_cast<float>(m_config.width) / static_cast<float>(m_config.height),
                                             m_camera.nearPlane,
                                             m_camera.farPlane);
        skyProj[1][1] *= -1.0f;
        const glm::mat4 skyView = glm::lookAt(m_camera.position, m_camera.target, m_camera.up);
        const glm::mat4 skyViewProj = skyProj * skyView;
        const SkyPushConstants skyPc{
            .invViewProj = glm::transpose(glm::inverse(skyViewProj)),
            .cameraPos = glm::vec4(m_camera.position, 1.0f),
            .exposure = 1.0f / (1.2f * std::pow(2.0f, m_camera.ev100)),
            .hasEnv = 1u,
            .envScale = m_envUnitNits,
            ._pad1 = 0u,
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyPipelineLayout, 0, 1, &m_iblSet, 0, nullptr);
        vkCmdPushConstants(cmd,
                           m_skyPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(skyPc),
                           &skyPc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

    // Bind descriptor sets: set 0 = geometry, set 1 = materials/lights, set 2 = IBL, set 3 = bindless textures.
    const std::array<VkDescriptorSet, 4> descSets{m_meshSet, m_matSet, m_iblSet, m_textureSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 4, descSets.data(), 0, nullptr);

    // Push constants: camera matrices + OpenPBR lighting params
    glm::mat4 proj = glm::perspective(glm::radians(m_camera.vfovDeg),
                                      static_cast<float>(m_config.width) / static_cast<float>(m_config.height),
                                      m_camera.nearPlane,
                                      m_camera.farPlane);
    proj[1][1] *= -1.0f;
    const glm::mat4 view = glm::lookAt(m_camera.position, m_camera.target, m_camera.up);
    const glm::mat4 viewProj = proj * view;
    // exposure = 1 / (1.2 * 2^EV100) — matches Hyperion's PhysicalCamera calc
    const float exposure = 1.0f / (1.2f * std::pow(2.0f, m_camera.ev100));

    const ForwardRenderer::MeshPushConstants pc{
        .viewProj = glm::transpose(viewProj), // transposed for Slang mul(pos, mat)
        .view = glm::transpose(view),
        .cameraPos = glm::vec4(m_camera.position, 1.0f),
        .exposure = exposure,
        .lightCount = m_scene ? m_scene->lightCount() : 0u,
        .emissiveTriangleCount = m_scene ? m_scene->emissiveTriangleCount() : 0u,
        .tilesX = m_tilesX,
        .tilesY = m_tilesY,
        .screenWidth = m_config.width,
        .screenHeight = m_config.height,
        ._pad = 0u,
        // Ray-traced sun shadow: direction toward the dominant IBL light + strength.
        // shadowParams: x = ray tMin (scene-scale bias from camera near plane), y = sky ambient floor.
        .sunDirection = glm::vec4(m_sunDir, m_hasEnv ? m_sunStrength : 0.0f),
        .shadowParams = glm::vec4(std::max(m_camera.nearPlane, 1e-4f), 0.35f, 0.0f, 0.0f),
    };
    vkCmdPushConstants(cmd,
                       m_pipelineLayout,
                       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(ForwardRenderer::MeshPushConstants),
                       &pc);

    const uint32_t instanceCount = m_scene->instanceCount();
    if (instanceCount > 0) {
        if (vkCmdDrawMeshTasksEXT == nullptr) {
            Logger::error("vkCmdDrawMeshTasksEXT is NULL - mesh shader extension not loaded!");
        } else {
            vkCmdDrawMeshTasksEXT(cmd, instanceCount, 1, 1);
        }
    }

    vkCmdEndRendering(cmd);
}

} // namespace theia

#ifdef __clang__
#pragma clang diagnostic pop
#endif
