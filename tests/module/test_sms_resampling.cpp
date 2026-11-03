#include <volk/volk.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <slang-math/slang-math.hpp>
#include <span>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "harmonia/core/VulkanHandle.hpp"

namespace {
using Words = std::array<std::uint32_t, 4>;
struct Push {
    VkDeviceAddress inputs, outputs;
    std::uint32_t count, pad;
};
static_assert(sizeof(Push) == 24);
} // namespace

class SmsResamplingFixture : public VulkanFixture {
  protected:
    void probe(std::span<const Words> inputs, std::vector<sm::float4>& outputs) {
        const auto& ctx = deviceCtx();
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VkPipelineLayout rawLayout{};
        ASSERT_EQ(vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &rawLayout), VK_SUCCESS);
        const harmonia::core::UniquePipelineLayout layout{ctx.device, rawLayout};
        auto shader = harmonia::core::createShaderModule(
            ctx.device, std::filesystem::path(THEIA_TEST_SHADER_DIR) / "sms_resampling_test.spv");
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
        auto source = harmonia::core::Buffer::create(
            ctx, inputs.size_bytes(), storage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.sms.resampling.input");
        outputs.resize(inputs.size());
        auto target = harmonia::core::Buffer::create(ctx,
                                                     outputs.size() * sizeof(sm::float4),
                                                     storage,
                                                     VMA_MEMORY_USAGE_GPU_TO_CPU,
                                                     "test.sms.resampling.output");
        ASSERT_TRUE(source.has_value() && target.has_value());
        source->uploadData(inputs.data(), inputs.size_bytes());
        auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        vkCmdBindPipeline(*cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        const Push push{source->deviceAddress(), target->deviceAddress(), static_cast<std::uint32_t>(inputs.size()), 0};
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
};

TEST_F(SmsResamplingFixture, GeneralizedPairwiseMatchesFiniteDomainsAndAlternativeProxyScalings) {
    const std::array inputs{Words{0, 0, 0, 0}, Words{1, 0, 0, 0}};
    std::vector<sm::float4> outputs;
    ASSERT_NO_FATAL_FAILURE(probe(inputs, outputs));
    ASSERT_EQ(outputs.size(), 2U);
    for (const auto& value : outputs) {
        EXPECT_NEAR(value.x, 10.0F, 1e-4F);
        EXPECT_NEAR(value.y, 10.0F, 1e-4F);
        EXPECT_NEAR(value.z, 10.0F, 1e-4F);
        EXPECT_EQ(value.w, 0.0F);
    }
}
TEST_F(SmsResamplingFixture, StableHdrPairwiseAndExactRisSelectionBoundaries) {
    const std::array inputs{
        Words{2, 0, 0, 0}, Words{3, 0, 0, 0}, Words{3, 1, 0, 0}, Words{3, 2, 0, 0}, Words{3, 3, 0, 0}};
    std::vector<sm::float4> outputs;
    ASSERT_NO_FATAL_FAILURE(probe(inputs, outputs));
    ASSERT_EQ(outputs.size(), inputs.size());
    EXPECT_NEAR(outputs[0].x, 1.0F / 12, 1e-7F);
    EXPECT_NEAR(outputs[0].y, .25F, 1e-7F);
    EXPECT_EQ(outputs[0].z, 1.0F);
    EXPECT_EQ(outputs[0].w, 1.0F);
    for (std::size_t i = 1; i < outputs.size(); ++i) {
        const bool neighbor = i <= 2;
        EXPECT_EQ(outputs[i].x, neighbor ? 1.0F : 0.0F);
        EXPECT_EQ(outputs[i].y, 4.0F);
        EXPECT_EQ(outputs[i].z, neighbor ? 0.0F : 4.0F);
        EXPECT_EQ(outputs[i].w, neighbor ? 4.0F : 0.0F);
    }
}
