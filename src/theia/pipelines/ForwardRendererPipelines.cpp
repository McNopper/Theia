// ForwardRendererPipelines.cpp - the pipeline-creation cluster of
// theia::renderer::ForwardRenderer: graphics pipelines, descriptor set
// layouts and pipeline layouts. Methods of theia::renderer::ForwardRenderer
// split out of ForwardRenderer.cpp (TU split, no behavior change).
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
#include <theia/camera/CameraJitter.hpp>
#include <theia/renderer/ShaderPath.hpp>
#include <theia/scene/Scene.hpp>
#include <vector>

#include "theia/renderer/ForwardRenderer.hpp"

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
    if (!createOneSetLayout(meshBindings, {}, "Failed to create mesh descriptor set layout", m_desc.meshSetLayout)) {
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
    if (!createOneSetLayout(matBindings, {}, "Failed to create material descriptor set layout", m_desc.matSetLayout)) {
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
    if (!createOneSetLayout(iblBindings, {}, "Failed to create IBL descriptor set layout", m_desc.iblSetLayout)) {
        m_desc.matSetLayout.reset();
        m_desc.meshSetLayout.reset();
        return false;
    }

    const std::array<VkDescriptorSetLayoutBinding, 1> textureBindings{VkDescriptorSetLayoutBinding{
        0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxBindlessTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    const std::array<VkDescriptorBindingFlags, 1> textureBindingFlags{
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}; // MOD1: no UPDATE_AFTER_BIND (incompatible with DESCRIPTOR_BUFFER)
    if (!createOneSetLayout(textureBindings,
                            textureBindingFlags,
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

bool ForwardRenderer::initialize(const harmonia::DeviceContext& ctx, const Config& config) {
    m_config = config;
    m_ctx = &ctx;

    if (!createDepthTarget()) {
        harmonia::core::Logger::error("Failed to create depth target");
        return false;
    }

    // Hi-Z depth pyramid for two-pass occlusion culling. A pipeline failure (e.g. missing
    // shader) is non-fatal: the image is still created, the occlusion test disabled, and the
    // renderer falls back to drawing all meshlets. Only a failure to create the image itself
    // (catastrophic / OOM) is fatal.
    (void)m_gpu.hiZPass.initialize(ctx, config.width, config.height);
    if (m_gpu.hiZPass.sampledView() == VK_NULL_HANDLE) {
        harmonia::core::Logger::error("Failed to create Hi-Z image");
        return false;
    }

    // GPU-driven frustum cull pass (GD2/GD3/GD6). VK_EXT_mesh_shader is a hard requirement
    // (the rasterizer is mesh-shader based), so the indirect draw (GD3) is always available
    // and the cull pass must succeed — there is no CPU-count draw to fall back to.
    if (!m_gpu.gpuCullPass.initialize(ctx)) {
        harmonia::core::Logger::error("GpuCullPass: initialization failed");
        return false;
    }

    // Debug/A-B toggle: set THEIA_FORCE_GD3 to skip DGC layout creation and force the
    // vkCmdDrawMeshTasksIndirectEXT (GD3) draw path. Used to isolate DGC-specific regressions.
    bool forceGd3 = false;
    {
        if (const auto forceGd3Raw = harmonia::core::getEnvString("THEIA_FORCE_GD3")) {
            forceGd3 = ((!forceGd3Raw->empty() && (*forceGd3Raw)[0] != '0'));
            if (forceGd3) {
                harmonia::core::Logger::info("THEIA_FORCE_GD3 set — DGC disabled, using indirect draw fallback");
            }
        }
    }

    // GD6: create DGC indirect commands layout when VK_EXT_device_generated_commands is available.
    // One token: DRAW_MESH_TASKS_EXT (stride=12 = VkDrawMeshTasksIndirectCommandEXT).
    // sequenceCountAddress will point to indirectDrawBuf[0..3] (visible instance count).
    if (ctx.dgcSupported && m_gpu.gpuCullPass.isInitialized() && !forceGd3) {
        const VkIndirectCommandsLayoutTokenEXT dgcToken{
            .sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,
            .type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_EXT,
            .offset = 0,
        };
        const VkIndirectCommandsLayoutCreateInfoEXT dgcLayoutCI{
            .sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
            .flags = 0,
            .shaderStages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .indirectStride = kMeshTaskIndirectCmdSize,
            .pipelineLayout = VK_NULL_HANDLE, // no push-constant token
            .tokenCount = 1,
            .pTokens = &dgcToken,
        };
        VkIndirectCommandsLayoutEXT dgcLayout{};
        if (vkCreateIndirectCommandsLayoutEXT(ctx.device, &dgcLayoutCI, nullptr, &dgcLayout) != VK_SUCCESS) {
            harmonia::core::Logger::warn("ForwardRenderer: vkCreateIndirectCommandsLayoutEXT failed — GD3 fallback");
        } else {
            m_gpu.dgcLayout = harmonia::core::UniqueIndirectCommandsLayout{ctx.device, dgcLayout};
        }
    }

    if (!createPipeline()) {
        harmonia::core::Logger::error("Failed to create pipeline");
        return false;
    }

    // GD6: query and allocate preprocess buffer after pipeline creation (requires pipeline handle).
    // VkBufferUsageFlags2CreateInfo is needed because VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT
    // is a 64-bit flag not representable in the 32-bit VkBufferUsageFlags enum.
    if (m_gpu.dgcLayout != VK_NULL_HANDLE) {
        const VkGeneratedCommandsPipelineInfoEXT memReqPipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,
            .pipeline = m_graphicsPipeline,
        };
        const VkGeneratedCommandsMemoryRequirementsInfoEXT memReqInfo{
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT,
            .pNext = &memReqPipelineInfo,
            .indirectExecutionSet = VK_NULL_HANDLE,
            .indirectCommandsLayout = m_gpu.dgcLayout,
            .maxSequenceCount = 1u, // single GPU-generated draw (see drawOpaque/drawTransparent)
            .maxDrawCount = 1,
        };
        VkMemoryRequirements2 memReq{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        vkGetGeneratedCommandsMemoryRequirementsEXT(ctx.device, &memReqInfo, &memReq);
        m_gpu.dgcPreprocessSize = memReq.memoryRequirements.size;

        if (m_gpu.dgcPreprocessSize > 0) {
            constexpr VkBufferUsageFlags2KHR kPreprocessUsage =
                VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR;
            const VkBufferUsageFlags2CreateInfo usageFlags2{
                .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
                .usage = kPreprocessUsage,
            };
            const VkBufferCreateInfo ppBufCI{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = &usageFlags2,
                .size = m_gpu.dgcPreprocessSize,
                .usage = 0, // usage provided via VkBufferUsageFlags2CreateInfo
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            const VmaAllocationCreateInfo ppAllocCI{
                .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            };
            if (vmaCreateBuffer(
                    ctx.allocator, &ppBufCI, &ppAllocCI, &m_gpu.dgcPreprocessBuf, &m_gpu.dgcPreprocessAlloc, nullptr) ==
                VK_SUCCESS) {
                const VkBufferDeviceAddressInfo addrInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                    .buffer = m_gpu.dgcPreprocessBuf,
                };
                m_gpu.dgcPreprocessAddr = vkGetBufferDeviceAddress(ctx.device, &addrInfo);
            } else {
                harmonia::core::Logger::warn(
                    "ForwardRenderer: failed to allocate DGC preprocess buffer — GD3 fallback");
                m_gpu.dgcLayout.reset();
                m_gpu.dgcPreprocessSize = 0;
            }
        }
    }

    if (const auto debugModeRaw = harmonia::core::getEnvString("THEIA_DEBUG_RAY_HIT_MODE")) {
        m_render.debugRayHitMode = std::clamp(std::strtof(debugModeRaw->c_str(), nullptr), 0.0f, 6.0f);
        if (m_render.debugRayHitMode > 0.0f) {
            harmonia::core::Logger::info("THEIA_DEBUG_RAY_HIT_MODE = {:.1f}", m_render.debugRayHitMode);
        }
    }

    // Debug/A-B toggle: set THEIA_DISABLE_HIZ to draw all meshlets (two passes, no occlusion
    // test). Used to verify Hi-Z culling is visually equivalent to the uncullered path.
    if (const auto hiZDisableRaw = harmonia::core::getEnvString("THEIA_DISABLE_HIZ")) {
        m_gpu.hiZDebugDisabled = ((!hiZDisableRaw->empty() && (*hiZDisableRaw)[0] != '0'));
        if (m_gpu.hiZDebugDisabled) {
            harmonia::core::Logger::info("THEIA_DISABLE_HIZ set — Hi-Z occlusion test disabled");
        }
    }

    if (const auto singlePassRaw = harmonia::core::getEnvString("THEIA_SINGLE_PASS")) {
        m_gpu.forceSinglePass = ((!singlePassRaw->empty() && (*singlePassRaw)[0] != '0'));
        if (m_gpu.forceSinglePass) {
            harmonia::core::Logger::info("THEIA_SINGLE_PASS set — two-pass Hi-Z bypassed (single draw pass)");
        }
    }

    // Create 1-element dummy buffers for tile light slots (fallback when LightCuller hasn't run).
    // VK14: nullDescriptor — tile light buffers bind VK_NULL_HANDLE when absent (no dummy).

    // Env sampler (linear, U-repeat / V-clamp, mip-linear) for the raw equirectangular
    // panorama bound at set 2 binding 3 (sky background + transparent env path).
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
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler envSampler = VK_NULL_HANDLE;
    if (vkCreateSampler(ctx.device, &envSamplerInfo, nullptr, &envSampler) != VK_SUCCESS) {
        harmonia::core::Logger::error("ForwardRenderer: failed to create env sampler");
        return false;
    }
    m_envSampler = harmonia::core::UniqueSampler{ctx.device, envSampler};

    // VK14: nullDescriptor — env view binds VK_NULL_HANDLE when absent (no 1×1 placeholder).

    m_initialized = true;
    m_hdrFirstUse = true;
    const char* drawPath = m_gpu.dgcLayout != VK_NULL_HANDLE ? "DGC" : "indirect";
    harmonia::core::Logger::info(
        "GPU-driven forward renderer initialized ({}x{}) [{}]", config.width, config.height, drawPath);
    return true;
}

bool ForwardRenderer::resize(std::uint32_t width,
                             std::uint32_t height,
                             VkImage hdrImage,
                             VkImageView hdrImageView) noexcept {
    if (!m_ctx) {
        return false;
    }
    m_config.width = width;
    m_config.height = height;
    m_config.hdrImage = hdrImage;
    m_config.hdrImageView = hdrImageView;
    m_depthTarget = {};
    m_gbufferTarget = {};
    m_giBufferTarget = {};
    m_hdrFirstUse = true;
    // Recreate the Hi-Z pyramid at the new resolution. Visibility buffers are
    // resolution-independent (one entry per meshlet) and are kept across resize.
    (void)m_gpu.hiZPass.initialize(*m_ctx, width, height);
    return createDepthTarget();
}

bool ForwardRenderer::createDepthTarget() {
    VkExtent2D extent{m_config.width, m_config.height};
    // SAMPLED_BIT so SSR can read depth after the forward pass.
    auto depthResult =
        harmonia::core::Image::create(*m_ctx,
                                      extent,
                                      VK_FORMAT_D32_SFLOAT,
                                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                      VK_IMAGE_ASPECT_DEPTH_BIT,
                                      "theia.depth");
    if (!depthResult) {
        harmonia::core::Logger::error("Failed to create depth target");
        return false;
    }
    m_depthTarget = std::move(*depthResult);

    // GBuffer: RGBA16F — view-space normal (xyz * 0.5 + 0.5) + roughness (w).
    // SAMPLED_BIT so SSR compute can sample it; STORAGE_BIT reserved for future post-process.
    auto gbufResult = harmonia::core::Image::create(*m_ctx,
                                                    extent,
                                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                                    "theia.gbuffer");
    if (!gbufResult) {
        harmonia::core::Logger::error("Failed to create GBuffer target");
        return false;
    }
    m_gbufferTarget = std::move(*gbufResult);

    // GI GBuffer: RGBA32F — world-space primary hit position (xyz) + material index (w,
    // encoded as asfloat(materialIdx + 1); 0 = background). RGBA32F keeps the integer
    // material index bit-exact and the world position at full precision for ray seeding.
    // SAMPLED_BIT so the GI compute stage can fetch it via texelFetch (Texture2D.Load).
    auto giBufResult = harmonia::core::Image::create(*m_ctx,
                                                     extent,
                                                     VK_FORMAT_R32G32B32A32_SFLOAT,
                                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                     VK_IMAGE_ASPECT_COLOR_BIT,
                                                     "theia.gibuffer");
    if (!giBufResult) {
        harmonia::core::Logger::error("Failed to create GI GBuffer target");
        return false;
    }
    m_giBufferTarget = std::move(*giBufResult);
    return true;
}

void ForwardRenderer::transitionGbuffersForRecord(VkCommandBuffer cmd, VkImage hdrImage, bool giEnabled) noexcept {
    if (!giEnabled) {
        const std::array hdrToGeneral{harmonia::core::imageBarrier(hdrImage,
                                                                   VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                                                   VK_IMAGE_LAYOUT_GENERAL,
                                                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                                   VK_ACCESS_2_SHADER_READ_BIT)};
        harmonia::core::pipelineBarrier(cmd, hdrToGeneral);
    }

    VkImageLayout gbufferOld = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    VkPipelineStageFlags2 gbufferSrcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkAccessFlags2 gbufferSrcAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    VkImageLayout depthOld = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    VkPipelineStageFlags2 depthSrcStage =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    VkAccessFlags2 depthSrcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (giEnabled) {
        gbufferOld = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        gbufferSrcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        gbufferSrcAccess = VK_ACCESS_2_SHADER_READ_BIT;
    }

    const VkImageMemoryBarrier2 gbufferToGeneral = harmonia::core::imageBarrier(gbufferImage(),
                                                                                gbufferOld,
                                                                                VK_IMAGE_LAYOUT_GENERAL,
                                                                                gbufferSrcStage,
                                                                                gbufferSrcAccess,
                                                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                                VK_ACCESS_2_SHADER_READ_BIT);
    const VkImageMemoryBarrier2 depthToGeneral = harmonia::core::imageBarrier(depthImage(),
                                                                              depthOld,
                                                                              VK_IMAGE_LAYOUT_GENERAL,
                                                                              depthSrcStage,
                                                                              depthSrcAccess,
                                                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                              VK_ACCESS_2_SHADER_READ_BIT,
                                                                              VK_IMAGE_ASPECT_DEPTH_BIT);
    const std::array guideToGeneral{gbufferToGeneral, depthToGeneral};
    harmonia::core::pipelineBarrier(cmd, guideToGeneral);
}

void ForwardRenderer::prepareAttachments(VkCommandBuffer cmd) {
    const std::array imageBarriers{
        harmonia::core::imageBarrier(m_config.hdrImage,
                                     m_hdrFirstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     m_hdrFirstUse ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     0,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
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
        harmonia::core::imageBarrier(m_gbufferTarget.handle(),
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     0,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
        harmonia::core::imageBarrier(m_giBufferTarget.handle(),
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     0,
                                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
    };
    const VkDependencyInfo hdrDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &hdrDep);
    m_hdrFirstUse = false;
}

void ForwardRenderer::updateSceneDescriptors(VkCommandBuffer /*cmd*/) {
    // MOD1: fallback buffer handles for optional resources (env CDFs, tile lights, visibility).
    // VK14 nullDescriptor: bind a null descriptor when buffers are absent (shader gates on hasTileData).
    const harmonia::core::Buffer* tileLightCountsBuf = m_tileLightCountsBuf;
    const harmonia::core::Buffer* tileLightIndicesBuf = m_tileLightIndicesBuf;
    const harmonia::core::Buffer* envMarginalCdfBuf = m_envMarginalCdf;
    const harmonia::core::Buffer* envConditionalCdfBuf = m_envConditionalCdf;

    const VkAccelerationStructureKHR sceneTlas = m_scene->tlas();

    const harmonia::core::Buffer& visFallbackBuf = m_scene->meshletBuffer();
    const harmonia::core::Buffer& prevVisBuf = m_gpu.meshletVisibility[m_gpu.visFrame];
    const harmonia::core::Buffer& currVisBuf = m_gpu.meshletVisibility[m_gpu.visFrame ^ 1u];
    const harmonia::core::Buffer& prevVisBufFinal = prevVisBuf.isValid() ? prevVisBuf : visFallbackBuf;
    const harmonia::core::Buffer& currVisBufFinal = currVisBuf.isValid() ? currVisBuf : visFallbackBuf;
    const harmonia::core::Buffer& compactInstBuf = m_gpu.gpuCullPass.compactInstanceListBuffer();

    // MOD1: write all descriptors via the descriptor buffer writers.
    // Optional buffers (tile lights, env CDFs) may be absent — bind a null descriptor
    // (VK14 nullDescriptor; the shaders gate reads on hasTileData/hasEnvMap).
    const auto writeOptional =
        [this](harmonia::core::DescriptorBufferWriter& w, std::uint32_t binding, const harmonia::core::Buffer* buf) {
            if (buf != nullptr) {
                w.writeStorageBufferHandle(*m_ctx, binding, *buf);
            } else {
                w.writeStorageBufferHandle(*m_ctx, binding, VK_NULL_HANDLE, 0U);
            }
        };
    // Mesh set (set 0)
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 0, m_scene->vertexBuffer());
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 1, m_scene->instanceBuffer());
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 2, m_scene->indexBuffer());
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 3, m_scene->meshletBuffer());
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 4, m_scene->meshletVertexBuffer());
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 5, m_scene->meshletTriangleBuffer());
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 6, m_scene->materialBuffer());
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 7, prevVisBufFinal);
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 8, currVisBufFinal);
    m_desc.mesh.writeSampledImage(*m_ctx, 9, m_gpu.hiZPass.sampledView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 10, compactInstBuf);
    m_desc.mesh.writeStorageBufferHandle(*m_ctx, 11, m_scene->instanceTransformBuffer());

    // Mat set (set 1)
    m_desc.mat.writeStorageBufferHandle(*m_ctx, 0, m_scene->materialBuffer());
    m_desc.mat.writeStorageBufferHandle(*m_ctx, 1, m_scene->lightBuffer());
    m_desc.mat.writeStorageBufferHandle(*m_ctx, 2, m_scene->emissiveTriangleBuffer());
    writeOptional(m_desc.mat, 3, tileLightCountsBuf);
    writeOptional(m_desc.mat, 4, tileLightIndicesBuf);
    m_desc.mat.writeAccelerationStructure(*m_ctx, 5, sceneTlas);

    // IBL set (set 2) — bindings 3,4 written in setEnvironment(); 6,7 here.
    writeOptional(m_desc.ibl, 6, envMarginalCdfBuf);
    writeOptional(m_desc.ibl, 7, envConditionalCdfBuf);

    if (m_scene != m_desc.texturesBoundFor) {
        const auto& sceneTextures = m_scene->textures();
        const std::uint32_t texCount = std::min(static_cast<std::uint32_t>(sceneTextures.size()), kMaxBindlessTextures);
        for (std::uint32_t i = 0; i < texCount; ++i) {
            // MOD1: write bindless combined image samplers into the descriptor buffer.
            m_desc.texture.writeCombinedImageSampler(*m_ctx,
                                                     0,
                                                     i,
                                                     sceneTextures[i].sampler(),
                                                     sceneTextures[i].image().view(),
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        m_desc.texturesBoundFor = m_scene;
    }
}

void ForwardRenderer::beginSceneRendering(VkCommandBuffer cmd, VkAttachmentLoadOp loadOp) {
    VkClearColorValue clearCol{{0.0f, 0.0f, 0.0f, 1.0f}};
    VkClearColorValue clearGbuf{{0.5f, 0.5f, 0.5f, 1.0f}};
    VkClearColorValue clearGi{{0.0f, 0.0f, 0.0f, 0.0f}};
    VkClearDepthStencilValue clearDepth{1.0f, 0};

    auto colorAtt = [&](VkImageView v, VkAttachmentLoadOp op, VkClearColorValue c) {
        return VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = v,
            .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .loadOp = op,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{.color = c},
        };
    };
    auto depthAtt = [&](VkAttachmentLoadOp op) {
        return VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = m_depthTarget.view(),
            .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .loadOp = op,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue{.depthStencil = clearDepth},
        };
    };
    const std::array<VkRenderingAttachmentInfo, 3> atts{
        colorAtt(m_config.hdrImageView, loadOp, clearCol),
        colorAtt(m_gbufferTarget.view(), loadOp, clearGbuf),
        colorAtt(m_giBufferTarget.view(), loadOp, clearGi),
    };
    const VkRenderingAttachmentInfo depth = depthAtt(loadOp);
    const VkRenderingInfo info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea{{0, 0}, {m_config.width, m_config.height}},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<std::uint32_t>(atts.size()),
        .pColorAttachments = atts.data(),
        .pDepthAttachment = &depth,
    };
    vkCmdBeginRendering(cmd, &info);
}

void ForwardRenderer::bindMeshSets(VkCommandBuffer cmd) {
    // MOD1: bind descriptor buffers (replaces vkCmdBindDescriptorSets).
    {
        const std::array<harmonia::core::DescriptorBufferWriter*, 4> writers{
            &m_desc.mesh, &m_desc.mat, &m_desc.ibl, &m_desc.texture};
        harmonia::core::DescriptorBufferWriter::bindSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, writers);
    }
}

} // namespace theia
