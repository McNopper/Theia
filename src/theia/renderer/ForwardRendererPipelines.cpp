// ForwardRendererPipelines.cpp - the pipeline-creation cluster of
// theia::renderer::ForwardRenderer: graphics pipelines, descriptor set
// layouts and pipeline layouts. Methods of theia::renderer::ForwardRenderer
// split out of ForwardRenderer.cpp (TU split, no behavior change).
#include "theia/renderer/ForwardRenderer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <harmonia/core/Barrier.hpp>
#include <harmonia/core/Env.hpp>
#include <harmonia/core/Logger.hpp>
#include <harmonia/core/ShaderModule.hpp>
#include <slang-math/slang-math.hpp>
#include <theia/renderer/CameraJitter.hpp>
#include <theia/renderer/ShaderPath.hpp>
#include <theia/scene/Scene.hpp>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

namespace theia {

bool ForwardRenderer::createPipeline() {
    m_pipelineBuild.taskModule = loadShaderModule("forward_render.task.spv");
    m_pipelineBuild.meshModule = loadShaderModule("forward_render.mesh.spv");
    m_pipelineBuild.fragModule = loadShaderModule("forward_render.frag.spv");
    if (!m_pipelineBuild.taskModule || !m_pipelineBuild.meshModule || !m_pipelineBuild.fragModule) {
        vkDestroyShaderModule(m_ctx->device, m_pipelineBuild.taskModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, m_pipelineBuild.meshModule, nullptr);
        vkDestroyShaderModule(m_ctx->device, m_pipelineBuild.fragModule, nullptr);
        m_pipelineBuild = {};
        return false;
    }

    auto& b = m_pipelineBuild;
    b.viewport = VkPipelineViewportStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    b.rasterization = VkPipelineRasterizationStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    b.multisample = VkPipelineMultisampleStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    b.vertexInput = VkPipelineVertexInputStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    b.dynamicState = VkPipelineDynamicStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<std::uint32_t>(b.dynamicStates.size()),
        .pDynamicStates = b.dynamicStates.data(),
    };
    constexpr VkFormat kGbufferFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr VkFormat kGiBufferFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    b.colorFormats[0] = m_config.outputFormat;
    b.colorFormats[1] = kGbufferFormat;
    b.colorFormats[2] = kGiBufferFormat;
    b.rendering = VkPipelineRenderingCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = static_cast<std::uint32_t>(b.colorFormats.size()),
        .pColorAttachmentFormats = b.colorFormats.data(),
        .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
    };

    const bool ok = createDescriptorSetLayouts() && createPipelineLayouts() && createOpaquePipeline() &&
                    createTransparentPipeline() && createSkyPipeline();

    vkDestroyShaderModule(m_ctx->device, m_pipelineBuild.taskModule, nullptr);
    vkDestroyShaderModule(m_ctx->device, m_pipelineBuild.meshModule, nullptr);
    vkDestroyShaderModule(m_ctx->device, m_pipelineBuild.fragModule, nullptr);
    m_pipelineBuild = {};
    return ok;
}

bool ForwardRenderer::createOneSetLayout(std::span<const VkDescriptorSetLayoutBinding> bindings,
                                         std::span<const VkDescriptorBindingFlags> bindingFlags,
                                         const char* errorMessage,
                                         harmonia::core::UniqueDescriptorSetLayout& out) {
    // An all-zero pBindingFlags is spec-equivalent to omitting the flags chain, so the
    // chain is only included when per-binding flags are actually requested (MOD1: no
    // UPDATE_AFTER_BIND — incompatible with DESCRIPTOR_BUFFER layouts).
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = bindingFlags.empty() ? nullptr : &flagsInfo,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    VkDescriptorSetLayout layout{};
    if (vkCreateDescriptorSetLayout(m_ctx->device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
        harmonia::core::Logger::error("{}", errorMessage);
        return false;
    }
    out = harmonia::core::UniqueDescriptorSetLayout{m_ctx->device, layout};
    return true;
}

bool ForwardRenderer::createDescriptorSetLayouts() {
    constexpr VkShaderStageFlags kTaskAndMeshStages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    constexpr VkShaderStageFlags kTaskMeshFragStages =
        VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
    constexpr VkShaderStageFlags kTaskStage = VK_SHADER_STAGE_TASK_BIT_EXT;
    // MOD1: no UPDATE_AFTER_BIND — incompatible with DESCRIPTOR_BUFFER layouts.

    const std::array<VkDescriptorSetLayoutBinding, 12> meshBindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskMeshFragStages, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskMeshFragStages, nullptr},
        VkDescriptorSetLayoutBinding{2,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     1,
                                     VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                     nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskAndMeshStages, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskStage, nullptr},
        VkDescriptorSetLayoutBinding{7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        VkDescriptorSetLayoutBinding{10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTaskStage, nullptr},
        VkDescriptorSetLayoutBinding{11,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     1,
                                     VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                     nullptr},
    };
    if (!createOneSetLayout(meshBindings, {}, "Failed to create mesh descriptor set layout",
                            m_desc.meshSetLayout)) {
        return false;
    }

    const std::array<VkDescriptorSetLayoutBinding, 6> matBindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{
            5, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    if (!createOneSetLayout(matBindings, {}, "Failed to create material descriptor set layout",
                            m_desc.matSetLayout)) {
        m_desc.meshSetLayout.reset();
        return false;
    }

    // Set 2: environment resources. Indirect lighting is supplied by the RT-GI compute
    // stage; the forward pass keeps only the linear sampler (3), the raw env panorama (4)
    // and the env-importance CDFs (6/7) for the sky background / transparent env path.
    const std::array<VkDescriptorSetLayoutBinding, 4> iblBindings{
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    if (!createOneSetLayout(iblBindings, {}, "Failed to create IBL descriptor set layout",
                            m_desc.iblSetLayout)) {
        m_desc.matSetLayout.reset();
        m_desc.meshSetLayout.reset();
        return false;
    }

    const std::array<VkDescriptorSetLayoutBinding, 1> textureBindings{
        VkDescriptorSetLayoutBinding{
            0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    const std::array<VkDescriptorBindingFlags, 1> textureBindingFlags{
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}; // MOD1: no UPDATE_AFTER_BIND (incompatible with DESCRIPTOR_BUFFER)
    if (!createOneSetLayout(textureBindings, textureBindingFlags,
                            "Failed to create bindless texture descriptor set layout",
                            m_desc.textureSetLayout)) {
        m_desc.iblSetLayout.reset();
        m_desc.matSetLayout.reset();
        m_desc.meshSetLayout.reset();
        return false;
    }

    // MOD1: descriptor buffers (replaces the pool + allocated sets).
    if (!m_desc.mesh.init(*m_ctx, m_desc.meshSetLayout.get(), 12, "theia.fwd.mesh.descBuf") ||
        !m_desc.mat.init(*m_ctx, m_desc.matSetLayout.get(), 6, "theia.fwd.mat.descBuf") ||
        !m_desc.ibl.init(*m_ctx, m_desc.iblSetLayout.get(), 8, "theia.fwd.ibl.descBuf") ||
        !m_desc.texture.init(*m_ctx, m_desc.textureSetLayout.get(), 1, "theia.fwd.texture.descBuf")) {
        harmonia::core::Logger::error("Failed to create descriptor buffers");
        m_desc.mesh.shutdown();
        m_desc.mat.shutdown();
        m_desc.ibl.shutdown();
        m_desc.texture.shutdown();
        m_desc.textureSetLayout.reset();
        m_desc.iblSetLayout.reset();
        m_desc.matSetLayout.reset();
        m_desc.meshSetLayout.reset();
        return false;
    }
    return true;
}

bool ForwardRenderer::createPipelineLayouts() {
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(ForwardRenderer::MeshPushConstants),
    };
    const std::array<VkDescriptorSetLayout, 4> pipelineSetLayouts{
        m_desc.meshSetLayout, m_desc.matSetLayout, m_desc.iblSetLayout, m_desc.textureSetLayout};
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(pipelineSetLayouts.size()),
        .pSetLayouts = pipelineSetLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    VkPipelineLayout pipelineLayout{};
    if (vkCreatePipelineLayout(m_ctx->device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        harmonia::core::Logger::error("Failed to create graphics pipeline layout");
        m_desc.mesh.shutdown();
    m_desc.mat.shutdown();
    m_desc.ibl.shutdown();
    m_desc.texture.shutdown();
        m_desc.textureSetLayout.reset();
        m_desc.iblSetLayout.reset();
        m_desc.matSetLayout.reset();
        m_desc.meshSetLayout.reset();
        return false;
    }
    m_pipelineLayout = harmonia::core::UniquePipelineLayout{m_ctx->device, pipelineLayout};
    return true;
}

bool ForwardRenderer::createOpaquePipeline() {
    const auto& b = m_pipelineBuild;
    const std::array<VkPipelineShaderStageCreateInfo, 3> graphicsStages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
            .module = b.taskModule,
            .pName = "main",
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
            .module = b.meshModule,
            .pName = "main",
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = b.fragModule,
            .pName = "main",
        },
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
    const VkPipelineColorBlendAttachmentState gbufferAttachment = colorAttachment;
    const VkPipelineColorBlendAttachmentState giBufferAttachment = colorAttachment;
    const std::array<VkPipelineColorBlendAttachmentState, 3> colorAttachments{
        colorAttachment, gbufferAttachment, giBufferAttachment};
    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<std::uint32_t>(colorAttachments.size()),
        .pAttachments = colorAttachments.data(),
    };
    const VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &b.rendering,
        .flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT, // MOD1
        .stageCount = static_cast<std::uint32_t>(graphicsStages.size()),
        .pStages = graphicsStages.data(),
        .pVertexInputState = &b.vertexInput,
        .pInputAssemblyState = nullptr,
        .pViewportState = &b.viewport,
        .pRasterizationState = &b.rasterization,
        .pMultisampleState = &b.multisample,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &b.dynamicState,
        .layout = m_pipelineLayout,
    };
    VkPipeline graphicsPipeline{};
    if (vkCreateGraphicsPipelines(m_ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) !=
        VK_SUCCESS) {
        harmonia::core::Logger::error("Failed to create graphics pipeline");
        m_pipelineLayout.reset();
        m_desc.mesh.shutdown();
    m_desc.mat.shutdown();
    m_desc.ibl.shutdown();
    m_desc.texture.shutdown();
        m_desc.iblSetLayout.reset();
        m_desc.matSetLayout.reset();
        m_desc.meshSetLayout.reset();
        return false;
    }
    m_graphicsPipeline = harmonia::core::UniquePipeline{m_ctx->device, graphicsPipeline};
    return true;
}

bool ForwardRenderer::createTransparentPipeline() {
    const auto& b = m_pipelineBuild;
    const std::array<VkPipelineShaderStageCreateInfo, 3> graphicsStages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_TASK_BIT_EXT,
            .module = b.taskModule,
            .pName = "main",
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_MESH_BIT_EXT,
            .module = b.meshModule,
            .pName = "main",
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = b.fragModule,
            .pName = "main",
        },
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencilTransparent{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineColorBlendAttachmentState colorAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendAttachmentState giBufferAttachment = colorAttachment;
    const VkPipelineColorBlendAttachmentState transparentGbufferAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask = 0,
    };
    const std::array<VkPipelineColorBlendAttachmentState, 3> transparentColorAttachments{
        colorAttachment, transparentGbufferAttachment, giBufferAttachment};
    const VkPipelineColorBlendStateCreateInfo transparentColorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<std::uint32_t>(transparentColorAttachments.size()),
        .pAttachments = transparentColorAttachments.data(),
    };
    const VkGraphicsPipelineCreateInfo transparentPipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &b.rendering,
        .flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT, // MOD1
        .stageCount = static_cast<std::uint32_t>(graphicsStages.size()),
        .pStages = graphicsStages.data(),
        .pVertexInputState = &b.vertexInput,
        .pInputAssemblyState = nullptr,
        .pViewportState = &b.viewport,
        .pRasterizationState = &b.rasterization,
        .pMultisampleState = &b.multisample,
        .pDepthStencilState = &depthStencilTransparent,
        .pColorBlendState = &transparentColorBlend,
        .pDynamicState = &b.dynamicState,
        .layout = m_pipelineLayout,
    };
    VkPipeline graphicsPipelineTransparent{};
    if (vkCreateGraphicsPipelines(
            m_ctx->device, VK_NULL_HANDLE, 1, &transparentPipelineInfo, nullptr, &graphicsPipelineTransparent) !=
        VK_SUCCESS) {
        harmonia::core::Logger::error("Failed to create transparent graphics pipeline");
        m_graphicsPipeline.reset();
        m_pipelineLayout.reset();
        m_desc.mesh.shutdown();
    m_desc.mat.shutdown();
    m_desc.ibl.shutdown();
    m_desc.texture.shutdown();
        m_desc.textureSetLayout.reset();
        m_desc.iblSetLayout.reset();
        m_desc.matSetLayout.reset();
        m_desc.meshSetLayout.reset();
        return false;
    }
    m_graphicsPipelineTransparent = harmonia::core::UniquePipeline{m_ctx->device, graphicsPipelineTransparent};
    return true;
}

bool ForwardRenderer::createSkyPipeline() {
    const auto& b = m_pipelineBuild;
    VkShaderModule skyVert = loadShaderModule("sky.vert.spv");
    VkShaderModule skyFrag = loadShaderModule("sky.frag.spv");
    if (skyVert == VK_NULL_HANDLE || skyFrag == VK_NULL_HANDLE) {
        harmonia::core::Logger::error("Failed to load sky shader modules");
        if (skyVert)
            vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
        if (skyFrag)
            vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
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
        .pSetLayouts = m_desc.iblSetLayout.ptr(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &skyPcRange,
    };
    VkPipelineLayout skyPipelineLayout{};
    if (vkCreatePipelineLayout(m_ctx->device, &skyLayoutInfo, nullptr, &skyPipelineLayout) != VK_SUCCESS) {
        harmonia::core::Logger::error("Failed to create sky pipeline layout");
        vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
        vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
        return false;
    }
    m_skyPipelineLayout = harmonia::core::UniquePipelineLayout{m_ctx->device, skyPipelineLayout};

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
    const VkPipelineDepthStencilStateCreateInfo skyDepth{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_ALWAYS,
    };
    const VkPipelineColorBlendAttachmentState skyHdrBlend{
        .blendEnable = VK_FALSE,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendAttachmentState skyGbufBlend{.blendEnable = VK_FALSE, .colorWriteMask = 0};
    const std::array<VkPipelineColorBlendAttachmentState, 3> skyBlends{skyHdrBlend, skyGbufBlend, skyGbufBlend};
    const VkPipelineColorBlendStateCreateInfo skyColorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<std::uint32_t>(skyBlends.size()),
        .pAttachments = skyBlends.data(),
    };

    const VkGraphicsPipelineCreateInfo skyPipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &b.rendering,
        .flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT, // MOD1
        .stageCount = static_cast<std::uint32_t>(skyStages.size()),
        .pStages = skyStages.data(),
        .pVertexInputState = &b.vertexInput,
        .pInputAssemblyState = &skyInputAsm,
        .pViewportState = &b.viewport,
        .pRasterizationState = &b.rasterization,
        .pMultisampleState = &b.multisample,
        .pDepthStencilState = &skyDepth,
        .pColorBlendState = &skyColorBlend,
        .pDynamicState = &b.dynamicState,
        .layout = m_skyPipelineLayout,
    };
    VkPipeline skyPipeline{};
    const VkResult skyRes =
        vkCreateGraphicsPipelines(m_ctx->device, VK_NULL_HANDLE, 1, &skyPipelineInfo, nullptr, &skyPipeline);
    vkDestroyShaderModule(m_ctx->device, skyVert, nullptr);
    vkDestroyShaderModule(m_ctx->device, skyFrag, nullptr);
    if (skyRes != VK_SUCCESS) {
        harmonia::core::Logger::error("Failed to create sky pipeline");
        return false;
    }
    m_skyPipeline = harmonia::core::UniquePipeline{m_ctx->device, skyPipeline};
    return true;
}


} // namespace theia
