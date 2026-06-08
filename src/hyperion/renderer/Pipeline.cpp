#include "hyperion/renderer/Pipeline.hpp"

#include <volk/volk.h>

#include <array>
#include <cstdint>
#include <expected>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "hyperion/renderer/Descriptors.hpp"

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

[[nodiscard]] std::expected<VkShaderModule, VkResult> createShaderModule(const DeviceContext& ctx,
                                                                         const std::filesystem::path& path) {
    auto code = readSpirv(path);
    if (!code) {
        return std::unexpected(code.error());
    }

    const VkShaderModuleCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = code->size() * sizeof(uint32_t),
        .pCode = code->data(),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    if (const VkResult result = vkCreateShaderModule(ctx.device, &createInfo, nullptr, &module); result != VK_SUCCESS) {
        return std::unexpected(result);
    }
    return module;
}
} // namespace

Pipeline::~Pipeline() noexcept {
    reset();
}

Pipeline::Pipeline(Pipeline&& other) noexcept
    : m_ctx(other.m_ctx), m_rtPipeline(std::exchange(other.m_rtPipeline, VK_NULL_HANDLE)) {
    other.m_ctx = nullptr;
}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
    if (this != &other) {
        reset();
        m_ctx = other.m_ctx;
        m_rtPipeline = std::exchange(other.m_rtPipeline, VK_NULL_HANDLE);
        other.m_ctx = nullptr;
    }
    return *this;
}

std::expected<Pipeline, VkResult> Pipeline::create(const DeviceContext& ctx,
                                                   const Descriptors& descriptors,
                                                   const ShaderPaths& paths,
                                                   uint32_t maxRayRecursion) {
    std::array<VkShaderModule, 6> modules{};
    const std::array shaderPaths{
        paths.raygen,
        paths.closesthitTriangle,
        paths.closesthitSphere,
        paths.intersection,
        paths.miss,
        paths.shadowMiss,
    };

    for (size_t i = 0; i < shaderPaths.size(); ++i) {
        auto module = createShaderModule(ctx, shaderPaths[i]);
        if (!module) {
            for (VkShaderModule created : modules) {
                if (created != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(ctx.device, created, nullptr);
                }
            }
            return std::unexpected(module.error());
        }
        modules[i] = *module;
    }

    const std::array rtEntryPoints{
        "main",
        "triangleClosestHit",
        "sphereClosestHit",
        "main",
        "main",
        "main",
    };
    const std::array rtStages{
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        nullptr,
                                        0,
                                        VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                                        modules[0],
                                        rtEntryPoints[0],
                                        nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        nullptr,
                                        0,
                                        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                        modules[1],
                                        rtEntryPoints[1],
                                        nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        nullptr,
                                        0,
                                        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                        modules[2],
                                        rtEntryPoints[2],
                                        nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        nullptr,
                                        0,
                                        VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                                        modules[3],
                                        rtEntryPoints[3],
                                        nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        nullptr,
                                        0,
                                        VK_SHADER_STAGE_MISS_BIT_KHR,
                                        modules[4],
                                        rtEntryPoints[4],
                                        nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                        nullptr,
                                        0,
                                        VK_SHADER_STAGE_MISS_BIT_KHR,
                                        modules[5],
                                        rtEntryPoints[5],
                                        nullptr},
    };
    const std::array groups{
        VkRayTracingShaderGroupCreateInfoKHR{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             nullptr,
                                             VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                                             0,
                                             VK_SHADER_UNUSED_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             nullptr},
        VkRayTracingShaderGroupCreateInfoKHR{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             nullptr,
                                             VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                                             4,
                                             VK_SHADER_UNUSED_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             nullptr},
        VkRayTracingShaderGroupCreateInfoKHR{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             nullptr,
                                             VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                                             5,
                                             VK_SHADER_UNUSED_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             nullptr},
        VkRayTracingShaderGroupCreateInfoKHR{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             nullptr,
                                             VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             1,
                                             VK_SHADER_UNUSED_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             nullptr},
        VkRayTracingShaderGroupCreateInfoKHR{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             nullptr,
                                             VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
                                             VK_SHADER_UNUSED_KHR,
                                             2,
                                             VK_SHADER_UNUSED_KHR,
                                             3,
                                             nullptr},
    };
    const VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<uint32_t>(rtStages.size()),
        .pStages = rtStages.data(),
        .groupCount = static_cast<uint32_t>(groups.size()),
        .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = maxRayRecursion,
        .pLibraryInfo = nullptr,
        .pLibraryInterface = nullptr,
        .pDynamicState = nullptr,
        .layout = descriptors.pipelineLayout(),
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };

    Pipeline pipeline;
    pipeline.m_ctx = &ctx;

    if (const VkResult result = vkCreateRayTracingPipelinesKHR(
            ctx.device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtPipelineInfo, nullptr, &pipeline.m_rtPipeline);
        result != VK_SUCCESS) {
        for (VkShaderModule module : modules) {
            vkDestroyShaderModule(ctx.device, module, nullptr);
        }
        return std::unexpected(result);
    }

    ctx.setDebugName(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(pipeline.m_rtPipeline), "hyperion.rtPipeline");

    for (VkShaderModule module : modules) {
        vkDestroyShaderModule(ctx.device, module, nullptr);
    }

    return pipeline;
}

void Pipeline::reset() noexcept {
    if (m_ctx != nullptr && m_rtPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_ctx->device, m_rtPipeline, nullptr);
        m_rtPipeline = VK_NULL_HANDLE;
    }
    m_ctx = nullptr;
}
