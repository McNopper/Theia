#ifndef THEIA_TESTS_MODULE_SHADERPROBEFIXTURE_HPP
#define THEIA_TESTS_MODULE_SHADERPROBEFIXTURE_HPP

// Runs a compute test shader of Theia/tests/shaders that reads one float4 array and
// writes another, both through device addresses in its push constants:
//
//     struct Push { float4* inputs; float4* outputs; uint count, pad; };
//     [[vk::push_constant]] Push pc;      // [numthreads(64, 1, 1)]
//
// One dispatch of a single workgroup: 64 threads, `count` outputs. The shader's
// module is the code under test itself (no copy): the tests compare it with the shared
// Harmonia module both renderers consume.

#include <volk/volk.h>

#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <slang-math/slang-math.hpp>
#include <span>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "harmonia/core/VulkanHandle.hpp"

class ShaderProbeFixture : public VulkanFixture {
  public:
    /// One dispatch of @p shaderName (a .spv of THEIA_TEST_SHADER_DIR). The shader reads
    /// @p inputs and writes @p outputCount float4, which @p outputs receives.
    void probe(const char* shaderName,
               std::span<const sm::float4> inputs,
               std::size_t outputCount,
               std::vector<sm::float4>& outputs) {
        const auto& ctx = deviceCtx();
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VkPipelineLayout rawLayout{};
        ASSERT_EQ(vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &rawLayout), VK_SUCCESS);
        const harmonia::core::UniquePipelineLayout layout{ctx.device, rawLayout};
        auto shader =
            harmonia::core::createShaderModule(ctx.device, std::filesystem::path(THEIA_TEST_SHADER_DIR) / shaderName);
        ASSERT_TRUE(shader.has_value());
        const harmonia::core::UniqueShaderModule module{ctx.device, *shader};
        VkComputePipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        VkPipeline rawPipeline{};
        ASSERT_EQ(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &rawPipeline), VK_SUCCESS);
        const harmonia::core::UniquePipeline pipeline{ctx.device, rawPipeline};
        constexpr auto storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        const std::size_t inputBytes =
            std::max<std::size_t>(inputs.size(), 1U) * sizeof(sm::float4); // never a zero-sized buffer
        auto source = harmonia::core::Buffer::create(
            ctx, inputBytes, storage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.shaderProbe.input");
        ASSERT_TRUE(source.has_value());
        if (!inputs.empty()) {
            source->uploadData(inputs.data(), inputs.size_bytes());
        }
        outputs.assign(outputCount, sm::float4{});
        auto target = harmonia::core::Buffer::create(
            ctx, outputs.size() * sizeof(sm::float4), storage, VMA_MEMORY_USAGE_GPU_TO_CPU, "test.shaderProbe.output");
        ASSERT_TRUE(target.has_value());
        auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        vkCmdBindPipeline(*cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        const Push push{source->deviceAddress(), target->deviceAddress(), static_cast<std::uint32_t>(outputCount), 0U};
        vkCmdPushConstants(*cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(*cmd, 1, 1, 1);
        VkMemoryBarrier2 memory{};
        memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(*cmd, &dependency);
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        ASSERT_EQ(target->readData(outputs.data(), outputs.size() * sizeof(sm::float4)), VK_SUCCESS);
    }

  private:
    /// The push constants of the test shaders, as Slang declares them.
    struct Push {
        VkDeviceAddress inputs{};
        VkDeviceAddress outputs{};
        std::uint32_t count{};
        std::uint32_t pad{};
    };
    static_assert(sizeof(Push) == 24);
};

#endif // THEIA_TESTS_MODULE_SHADERPROBEFIXTURE_HPP
