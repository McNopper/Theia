#include "demo/presentation/ToneMapper.hpp"

#include <array>
#include <fstream>
#include <utility>
#include <vector>

#include "hyperion/GpuTypes.hpp"

namespace {
[[nodiscard]] std::expected<std::vector<uint32_t>, VkResult> readSpirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    const auto size = file.tellg();
    if (size <= 0 || (static_cast<size_t>(size) % sizeof(uint32_t)) != 0U) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(code.data()), size);
    if (!file) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }
    return code;
}

[[nodiscard]] std::expected<VkShaderModule, VkResult> createShaderModule(VkDevice device,
                                                                         const std::filesystem::path& path) {
    auto spirv = readSpirv(path);
    if (!spirv) {
        return std::unexpected(spirv.error());
    }
    const VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = spirv->size() * sizeof(uint32_t),
        .pCode = spirv->data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (const VkResult r = vkCreateShaderModule(device, &info, nullptr, &module); r != VK_SUCCESS) {
        return std::unexpected(r);
    }
    return module;
}
} // namespace

std::expected<ToneMapper, VkResult> ToneMapper::create(const DeviceContext& ctx,
                                                       VkPipelineLayout pipelineLayout,
                                                       VkFormat swapchainFormat,
                                                       const std::filesystem::path& vertSpvPath,
                                                       const std::filesystem::path& fragSpvPath) {
    if (!ctx.isValid() || pipelineLayout == VK_NULL_HANDLE) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    auto vertModule = createShaderModule(ctx.device, vertSpvPath);
    if (!vertModule) {
        return std::unexpected(vertModule.error());
    }
    auto fragModule = createShaderModule(ctx.device, fragSpvPath);
    if (!fragModule) {
        vkDestroyShaderModule(ctx.device, *vertModule, nullptr);
        return std::unexpected(fragModule.error());
    }

    const std::array stages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = *vertModule,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = *fragModule,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };
    const VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = nullptr, // dynamic
        .scissorCount = 1,
        .pScissors = nullptr, // dynamic
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };
    const VkPipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &blendAttachment,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
    };

    constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    // Chain VkPipelineRenderingCreateInfo for dynamic rendering (no render pass object needed).
    const VkPipelineRenderingCreateInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapchainFormat,
        .depthAttachmentFormat = VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };
    const VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .flags = 0,
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pTessellationState = nullptr,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = VK_NULL_HANDLE, // dynamic rendering — no render pass
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };

    ToneMapper mapper;
    mapper.m_device = ctx.device;
    mapper.m_pipelineLayout = pipelineLayout;
    mapper.m_attachmentFormat = swapchainFormat;
    const VkResult result =
        vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mapper.m_pipeline);

    vkDestroyShaderModule(ctx.device, *vertModule, nullptr);
    vkDestroyShaderModule(ctx.device, *fragModule, nullptr);

    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    ctx.setDebugName(
        VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(mapper.m_pipeline), "hyperion.tonemapPipeline");
    return mapper;
}

ToneMapper::ToneMapper(ToneMapper&& other) noexcept
    : m_device(std::exchange(other.m_device, VK_NULL_HANDLE)),
      m_pipeline(std::exchange(other.m_pipeline, VK_NULL_HANDLE)),
      m_pipelineLayout(std::exchange(other.m_pipelineLayout, VK_NULL_HANDLE)),
      m_attachmentFormat(std::exchange(other.m_attachmentFormat, VK_FORMAT_UNDEFINED)) {}

ToneMapper& ToneMapper::operator=(ToneMapper&& other) noexcept {
    if (this != &other) {
        destroy();
        m_device = std::exchange(other.m_device, VK_NULL_HANDLE);
        m_pipeline = std::exchange(other.m_pipeline, VK_NULL_HANDLE);
        m_pipelineLayout = std::exchange(other.m_pipelineLayout, VK_NULL_HANDLE);
        m_attachmentFormat = std::exchange(other.m_attachmentFormat, VK_FORMAT_UNDEFINED);
    }
    return *this;
}

ToneMapper::~ToneMapper() {
    destroy();
}

void ToneMapper::record(VkCommandBuffer cmd,
                        VkImageView hdrView,
                        VkImageView swapchainView,
                        VkExtent2D extent,
                        OutputColorSpace colorSpace) const noexcept {
    if (cmd == VK_NULL_HANDLE || m_pipeline == VK_NULL_HANDLE || hdrView == VK_NULL_HANDLE ||
        swapchainView == VK_NULL_HANDLE) {
        return;
    }

    // Push hdrTarget (binding 1) as storage image in GENERAL layout.
    const VkDescriptorImageInfo hdrInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = hdrView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkWriteDescriptorSet hdrWrite{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &hdrInfo,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &hdrWrite);

    const uint32_t cs = static_cast<uint32_t>(colorSpace);
    vkCmdPushConstants(cmd,
                       m_pipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       offsetof(PushConstants, outputColorSpace),
                       sizeof(uint32_t),
                       &cs);

    const VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = swapchainView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {},
    };
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderArea = {.offset = {0, 0}, .extent = extent},
        .layerCount = 1,
        .viewMask = 0,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = nullptr,
        .pStencilAttachment = nullptr,
    };

    vkCmdBeginRendering(cmd, &renderingInfo);

    const VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    const VkRect2D scissor{.offset = {0, 0}, .extent = extent};

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);
}

void ToneMapper::destroy() noexcept {
    if (m_device != VK_NULL_HANDLE && m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
    }
    m_device = VK_NULL_HANDLE;
    m_pipeline = VK_NULL_HANDLE;
    m_pipelineLayout = VK_NULL_HANDLE;
    m_attachmentFormat = VK_FORMAT_UNDEFINED;
}
