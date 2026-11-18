// 2d.2 failed-inverse controls for sms_shift.slang: a rejected inverse match
// is what defines zero MIS support for a shifted candidate. The forward half
// (scene-projected shift walks) is covered by the frame diagnostics test
// (DiagnosticsCountActualRootsAndAcceptedShifts); this fixture pins the pure
// inverse-check contract: accept exact/within-tolerance, reject beyond-tolerance
// and every identity-field mismatch.
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
struct ShiftInput {
    sm::float4 receiver;               // xyz + mode in w
    sm::float4 original;               // xyz + etaTOverI
    sm::float4 inverse;                // xyz + etaTOverI
    std::array<std::uint32_t, 4> meta; // eventA, eventB, keyA, keyB
};
static_assert(sizeof(ShiftInput) == 64);
struct Push {
    VkDeviceAddress inputs, outputs;
    std::uint32_t count, pad;
};
static_assert(sizeof(Push) == 24);

// receiver at +Z; original root at (1,0,0). eta/event/key stay identical
// unless the case says otherwise, so the direction term is the discriminator.
ShiftInput baseInput(std::uint32_t mode) {
    ShiftInput input{};
    input.receiver = {0.0F, 0.0F, 10.0F, static_cast<float>(mode)};
    input.original = {1.0F, 0.0F, 0.0F, 1.0F};
    input.inverse = {1.0F, 0.0F, 0.0F, 1.0F};
    input.meta = {0u, 0u, 7u, 7u};
    return input;
}
} // namespace

class SmsShiftFixture : public VulkanFixture {
  protected:
    void probe(std::span<const ShiftInput> inputs, std::vector<sm::float4>& outputs) {
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
            ctx.device, std::filesystem::path(THEIA_TEST_SHADER_DIR) / "sms_shift_test.spv");
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
            ctx, inputs.size_bytes(), storage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.sms.shift.input");
        outputs.resize(inputs.size());
        auto target = harmonia::core::Buffer::create(
            ctx, outputs.size() * sizeof(sm::float4), storage, VMA_MEMORY_USAGE_GPU_TO_CPU, "test.sms.shift.output");
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

TEST_F(SmsShiftFixture, SmsInverseMatchAcceptsExactAndWithinToleranceRejectsBeyond) {
    // mode 0 exact, mode 1 beyond tolerance, mode 2 within tolerance.
    std::array inputs{baseInput(0u), baseInput(1u), baseInput(2u)};
    inputs[1].inverse = {2.0F, 0.0F, 0.0F, 1.0F};           // direction differs far past 1e-4
    inputs[2].inverse = {1.0F + 1.0e-5F, 0.0F, 0.0F, 1.0F}; // direction difference ~1e-6
    std::vector<sm::float4> outputs;
    ASSERT_NO_FATAL_FAILURE(probe(inputs, outputs));
    ASSERT_EQ(outputs.size(), inputs.size());
    EXPECT_EQ(outputs[0].x, 1.0F) << "exact inverse must be accepted";
    EXPECT_EQ(outputs[0].y, 1.0F);
    EXPECT_EQ(outputs[0].z, 0.0F) << "identical roots must measure an exact zero difference";
    EXPECT_EQ(outputs[1].x, 0.0F) << "beyond-tolerance inverse must FAIL (zero MIS support)";
    EXPECT_EQ(outputs[1].y, 1.0F) << "identity fields intact: direction is the discriminator";
    EXPECT_GT(outputs[1].z, 1.0e-4F) << "measured difference must exceed the tolerance";
    EXPECT_EQ(outputs[2].x, 1.0F) << "within-tolerance inverse must be accepted";
    EXPECT_LE(outputs[2].z, 1.0e-4F) << "measured difference must sit within the tolerance";
}

TEST_F(SmsShiftFixture, SmsInverseMatchAndTopologyRejectIdentityFieldMismatches) {
    // mode 3 objectKey, mode 4 event, mode 5 eta, mode 6 invalid count.
    std::array inputs{baseInput(3u), baseInput(4u), baseInput(5u), baseInput(6u)};
    inputs[0].meta[3] = 8u;     // inverse objectKey differs
    inputs[1].meta[1] = 1u;     // inverse event differs
    inputs[2].inverse.w = 1.5F; // inverse eta differs
    std::vector<sm::float4> outputs;
    ASSERT_NO_FATAL_FAILURE(probe(inputs, outputs));
    ASSERT_EQ(outputs.size(), inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(outputs[i].x, 0.0F) << "mismatched/invalid inverse must not match";
        EXPECT_EQ(outputs[i].y, 0.0F) << "mismatched/invalid topology must be rejected";
    }
}
