// Native Theia buffers and real BLAS/TLAS, checked against independent triangle
// interpolation/finite differences and virtual-emitter / scalar Snell oracles.
#include <volk/volk.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/DescriptorBufferWriter.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "theia/scene/Scene.hpp"

namespace {
struct Input {
    sm::float4 origin, direction, light;
};
struct Output {
    sm::float4 position, normal, geometric, dpdu, dpdv, dndu, dndv, keys;
};
struct Push {
    VkDeviceAddress inputs, outputs;
    std::uint32_t count, mode;
};
static_assert(sizeof(Input) == 48 && sizeof(Output) == 128 && sizeof(Push) == 24);
sm::float3 xyz(sm::float4 value) {
    return {value.x, value.y, value.z};
}
void expectVector(sm::float3 actual, sm::float3 expected, float tolerance = 3e-4F) {
    EXPECT_LE(sm::length(actual - expected), tolerance);
}
} // namespace

class SmsSceneFixture : public RtFixture {
  protected:
    void probe(const theia::Scene& scene,
               std::span<const Input> inputs,
               std::vector<Output>& outputs,
               std::uint32_t mode = 0) {
        const auto& ctx = deviceCtx();
        std::array<VkDescriptorSetLayoutBinding, 20> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i)
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        for (auto i : {2U, 3U, 4U, 15U, 18U})
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        VkDescriptorSetLayoutCreateInfo setInfo{};
        setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        setInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        setInfo.pBindings = bindings.data();
        VkDescriptorSetLayout rawSet{};
        ASSERT_EQ(vkCreateDescriptorSetLayout(ctx.device, &setInfo, nullptr, &rawSet), VK_SUCCESS);
        const harmonia::core::UniqueDescriptorSetLayout set{ctx.device, rawSet};
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = set.ptr();
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VkPipelineLayout rawLayout{};
        ASSERT_EQ(vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &rawLayout), VK_SUCCESS);
        const harmonia::core::UniquePipelineLayout layout{ctx.device, rawLayout};
        harmonia::core::DescriptorBufferWriter writer;
        ASSERT_TRUE(writer.init(ctx, set, 20, "test.theia.sms.scene"));
        writer.writeAccelerationStructure(ctx, 0, scene.tlas());
        writer.writeStorageBufferHandle(ctx, 6, scene.materialBuffer());
        writer.writeStorageBufferHandle(ctx, 7, scene.vertexBuffer());
        writer.writeStorageBufferHandle(ctx, 8, scene.instanceBuffer());
        writer.writeStorageBufferHandle(ctx, 9, scene.indexBuffer());
        writer.writeStorageBufferHandle(ctx, 19, scene.instanceTransformBuffer());
        constexpr auto storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto source = harmonia::core::Buffer::create(
            ctx, inputs.size_bytes(), storage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.theia.sms.input");
        outputs.resize(inputs.size());
        auto target = harmonia::core::Buffer::create(
            ctx, outputs.size() * sizeof(Output), storage, VMA_MEMORY_USAGE_GPU_TO_CPU, "test.theia.sms.output");
        ASSERT_TRUE(source.has_value() && target.has_value());
        source->uploadData(inputs.data(), inputs.size_bytes());
        auto shader = harmonia::core::createShaderModule(
            ctx.device, std::filesystem::path(THEIA_TEST_SHADER_DIR) / "sms_scene_test.spv");
        ASSERT_TRUE(shader.has_value());
        const harmonia::core::UniqueShaderModule module{ctx.device, *shader};
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = layout;
        VkPipeline rawPipeline{};
        ASSERT_EQ(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &rawPipeline),
                  VK_SUCCESS);
        const harmonia::core::UniquePipeline pipeline{ctx.device, rawPipeline};
        auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        VkMemoryBarrier2 memory{};
        memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(*cmd, &dependency);
        writer.bind(*cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout);
        vkCmdBindPipeline(*cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        const Push pc{
            source->deviceAddress(), target->deviceAddress(), static_cast<std::uint32_t>(inputs.size()), mode};
        vkCmdPushConstants(*cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(*cmd, 1, 1, 1);
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        vkCmdPipelineBarrier2(*cmd, &dependency);
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        ASSERT_EQ(target->readData(outputs.data(), outputs.size() * sizeof(Output)), VK_SUCCESS);
    }
};

TEST_F(SmsSceneFixture, NativeReaderMatchesTransformedTriangleAndRawOutwardDifferentials) {
    theia::Scene scene;
    auto material = scene.addMaterial(harmonia::scene::Material::mirror());
    harmonia::scene::MeshData data;
    const std::array positions{sm::float3(-1, -1, 0), sm::float3(1, -1, 0), sm::float3(0, 1, 0)};
    const std::array normals{
        sm::float3(0, 0, 1), sm::normalize(sm::float3(.2F, 0, 1)), sm::normalize(sm::float3(0, .3F, 1))};
    for (std::size_t i = 0; i < positions.size(); ++i) {
        harmonia::GpuVertex vertex{};
        vertex.position = positions[i];
        vertex.normal = normals[i];
        data.vertices.push_back(vertex);
    }
    data.indices = {0, 1, 2};
    const auto mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.theia.sms.triangle");
    ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
    ASSERT_NE(scene.addInstance(
                  mesh, harmonia::scene::Transform{.translation = {-3, 0, 0}, .scale = {1.5F, .75F, 1}}, material),
              std::numeric_limits<std::uint32_t>::max());
    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    const std::array inputs{Input{{-3, 0, 3, 0}, {0, 0, -1, 0}, {}},
                            Input{{-3, 0, -3, 0}, {0, 0, 1, 0}, {}},
                            Input{{0, 0, 3, 0}, {0, 0, 1, 0}, {}}};
    std::vector<Output> outputs;
    ASSERT_NO_FATAL_FAILURE(probe(scene, inputs, outputs));
    ASSERT_EQ(outputs.size(), inputs.size());
    for (std::size_t i = 0; i < 2; ++i) {
        ASSERT_EQ(outputs[i].position.w, 1.0F);
        expectVector(xyz(outputs[i].position), {-3, 0, 0});
        expectVector(xyz(outputs[i].geometric), {0, 0, 1}); // back hit must retain the same outward winding
    }
    EXPECT_EQ(outputs[2].position.w, 0.0F);
    const auto normalAt = [&](sm::float3 p) {
        const float b2 = (p.y + .75F) / 1.5F, b1 = (p.x + 4.5F - 1.5F * b2) / 3;
        const auto inverseScale = sm::float3(1 / 1.5F, 1 / .75F, 1);
        return sm::normalize(sm::normalize(normals[0] * inverseScale) * (1 - b1 - b2) +
                             sm::normalize(normals[1] * inverseScale) * b1 +
                             sm::normalize(normals[2] * inverseScale) * b2);
    };
    const auto p = xyz(outputs[0].position), u = xyz(outputs[0].dpdu), v = xyz(outputs[0].dpdv);
    constexpr float epsilon = 1e-3F;
    expectVector(xyz(outputs[0].normal), normalAt(p));
    expectVector(xyz(outputs[0].dndu), (normalAt(p + u * epsilon) - normalAt(p - u * epsilon)) / (2 * epsilon));
    expectVector(xyz(outputs[0].dndv), (normalAt(p + v * epsilon) - normalAt(p - v * epsilon)) / (2 * epsilon));
    EXPECT_EQ(outputs[0].keys.x, 0.0F);
    EXPECT_EQ(outputs[0].keys.y, static_cast<float>(material));
}

TEST_F(SmsSceneFixture, NativeProjectionSolvesMirrorAndClosedSlabAtIndependentSnellRoots) {
    for (bool glass : {false, true}) {
        theia::Scene scene;
        const auto material =
            scene.addMaterial(glass ? harmonia::scene::Material::glass(1.5F, 0) : harmonia::scene::Material::mirror());
        const auto mesh = scene.addMesh(deviceCtx(),
                                        commandPool(),
                                        harmonia::scene::ProceduralGeometry::makeBox({100, 100, glass ? .25F : .01F}),
                                        {},
                                        "test.theia.sms.slab");
        ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
        ASSERT_NE(
            scene.addInstance(mesh, harmonia::scene::Transform{.translation = {0, 0, glass ? .25F : .01F}}, material),
            std::numeric_limits<std::uint32_t>::max());
        ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
        std::vector<Input> inputs;
        if (!glass)
            inputs.push_back({{-1, 0, -1, 0}, {1.1F, 0, 1, 0}, {1, 0, -1, 0}});
        else
            for (double angle : {0.0, 25.0, 55.0}) {
                const double t = std::tan(angle * std::numbers::pi / 180), D = std::sqrt(2.25 + 1.25 * t * t);
                inputs.push_back({{0, 0, -1, 0},
                                  {static_cast<float>(t + .1), 0, 1, 0},
                                  {static_cast<float>(2.5 * t + .5 * t / D), 0, 2, 0}});
            }
        std::vector<Output> outputs;
        ASSERT_NO_FATAL_FAILURE(probe(scene, inputs, outputs, glass ? 2U : 1U));
        ASSERT_EQ(outputs.size(), inputs.size());
        if (!glass) {
            EXPECT_EQ(outputs[0].position.w, 1.0F);
            expectVector(xyz(outputs[0].position), {0, 0, 0});
            EXPECT_NEAR(outputs[0].geometric.w, 1 / std::sqrt(8.0) / 4, 2e-6);
        } else
            for (std::size_t i = 0; i < outputs.size(); ++i) {
                SCOPED_TRACE(i);
                const double angle = std::array{0.0, 25.0, 55.0}[i], t = std::tan(angle * std::numbers::pi / 180);
                const double D = std::sqrt(2.25 + 1.25 * t * t), rho = 2.5 * t + .5 * t / D;
                const double derivative = 2.5 + .5 * 2.25 / (D * D * D);
                const double geometric =
                    i == 0 ? 1 / std::pow(2.5 + .5 / 1.5, 2) : t / (rho * std::pow(1 + t * t, 1.5) * derivative);
                EXPECT_EQ(outputs[i].position.w, 2.0F);
                expectVector(xyz(outputs[i].position), {static_cast<float>(t), 0, 0});
                expectVector(xyz(outputs[i].geometric), {static_cast<float>(t + .5 * t / D), 0, .5F});
                EXPECT_NEAR(outputs[i].geometric.w / geometric, 1.0, 2e-5);
                EXPECT_LE(outputs[i].keys.w, 1e-5F);
            }
    }
}

TEST_F(SmsSceneFixture, NativeReaderPreservesProductionGeometricFallback) {
    // All-zero, exact nonzero cancellation, and a nonzero blend below production's
    // squared-length cutoff. The oracle is the flat triangle's constant +Z normal.
    for (float epsilon : {-1.0F, 0.0F, 1.0e-5F}) {
        SCOPED_TRACE(epsilon);
        theia::Scene scene;
        const auto material = scene.addMaterial(harmonia::scene::Material::mirror());
        harmonia::scene::MeshData data;
        const std::array positions{sm::float3(-1, -1, 0), sm::float3(1, -1, 0), sm::float3(0, 1, 0)};
        for (std::size_t i = 0; i < positions.size(); ++i) {
            harmonia::GpuVertex vertex{};
            vertex.position = positions[i];
            if (epsilon >= 0)
                vertex.normal = i == 2 ? sm::normalize(sm::float3(-1, 0, epsilon)) : sm::float3(1, 0, 0);
            data.vertices.push_back(vertex);
        }
        data.indices = {0, 1, 2};
        const auto mesh =
            scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.theia.sms.normal.fallback");
        ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
        ASSERT_NE(scene.addInstance(mesh, {}, material), std::numeric_limits<std::uint32_t>::max());
        ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
        const std::array inputs{Input{{0, 0, 1, 0}, {0, 0, -1, 0}, {}}};
        std::vector<Output> outputs;
        ASSERT_NO_FATAL_FAILURE(probe(scene, inputs, outputs));
        ASSERT_EQ(outputs.size(), 1U);
        EXPECT_EQ(outputs[0].position.w, 1.0F);
        expectVector(xyz(outputs[0].normal), {0, 0, 1});
        expectVector(xyz(outputs[0].dndu), {0, 0, 0});
        expectVector(xyz(outputs[0].dndv), {0, 0, 0});
    }
}

TEST_F(SmsSceneFixture, NativeVisibilityDoesNotDependOnShadingNormals) {
    theia::Scene scene;
    const auto material = scene.addMaterial(harmonia::scene::Material::mirror());
    harmonia::scene::MeshData data;
    for (auto position : {sm::float3(-1, -1, 0), sm::float3(1, -1, 0), sm::float3(0, 1, 0)}) {
        harmonia::GpuVertex vertex{};
        vertex.position = position; // deliberately zero shading normals
        data.vertices.push_back(vertex);
    }
    data.indices = {0, 1, 2};
    const auto mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.theia.sms.blocker");
    ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
    ASSERT_NE(scene.addInstance(mesh, {}, material), std::numeric_limits<std::uint32_t>::max());
    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    const std::array inputs{Input{{0, 0, 3, 0}, {0, 0, -1, 0}, {0, 0, 0, 5}},
                            Input{{0, 0, 3, 0}, {0, 0, -1, 0}, {0, 0, 0, 2}},
                            Input{{0, 0, 3, 0}, {0, 0, 1, 0}, {0, 0, 0, 5}}};
    std::vector<Output> outputs;
    ASSERT_NO_FATAL_FAILURE(probe(scene, inputs, outputs, 3));
    ASSERT_EQ(outputs.size(), 3U);
    EXPECT_EQ(outputs[0].position.w, 1.0F);
    EXPECT_EQ(outputs[1].position.w, 0.0F);
    EXPECT_EQ(outputs[2].position.w, 0.0F);
}

TEST_F(SmsSceneFixture, NativeManifoldShiftPreservesEmitterAndChecksInverseSupport) {
    for (bool glass : {false, true}) {
        theia::Scene scene;
        const auto material =
            scene.addMaterial(glass ? harmonia::scene::Material::glass(1.5F, 0) : harmonia::scene::Material::mirror());
        const auto mesh = scene.addMesh(deviceCtx(),
                                        commandPool(),
                                        harmonia::scene::ProceduralGeometry::makeBox({100, 100, glass ? .25F : .01F}),
                                        {},
                                        "test.sms.shift.caster");
        ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
        ASSERT_NE(
            scene.addInstance(mesh, harmonia::scene::Transform{.translation = {0, 0, glass ? .25F : .01F}}, material),
            std::numeric_limits<std::uint32_t>::max());
        ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
        Input input{{-1, 0, -1, 0}, {1.1F, 0, 1, 0}, {1, 0, -1, .2F}};
        if (glass) {
            const double t = std::tan(25 * std::numbers::pi / 180), D = std::sqrt(2.25 + 1.25 * t * t);
            input = {{0, 0, -1, 0},
                     {static_cast<float>(t + .1), 0, 1, 0},
                     {static_cast<float>(2.5 * t + .5 * t / D), 0, 2, .2F}};
        }
        std::vector<Output> outputs;
        ASSERT_NO_FATAL_FAILURE(probe(scene, std::span(&input, 1), outputs, glass ? 5U : 4U));
        ASSERT_EQ(outputs.size(), 1U);
        ASSERT_EQ(outputs[0].position.w, 1.0F);
        if (!glass)
            expectVector(xyz(outputs[0].position), {.1F, 0, 0});
        else {
            const double rho = double(input.light.x) - .2;
            double lo = 0, hi = rho / 2.5;
            for (int iteration = 0; iteration < 64; ++iteration) {
                const double t = (lo + hi) * .5, D = std::sqrt(2.25 + 1.25 * t * t);
                if (2.5 * t + .5 * t / D < rho)
                    lo = t;
                else
                    hi = t;
            }
            const double t = (lo + hi) * .5, D = std::sqrt(2.25 + 1.25 * t * t);
            expectVector(xyz(outputs[0].position), {static_cast<float>(.2 + t), 0, 0});
            expectVector(xyz(outputs[0].geometric), {static_cast<float>(.2 + t + .5 * t / D), 0, .5F});
            ASSERT_NO_FATAL_FAILURE(probe(scene, std::span(&input, 1), outputs, 6U));
            EXPECT_EQ(outputs[0].position.w, 0.0F) << "TT shift must respect target receiver depth";
        }
        if (!glass) {
            for (auto mode : {7U, 8U, 9U}) {
                ASSERT_NO_FATAL_FAILURE(probe(scene, std::span(&input, 1), outputs, mode));
                EXPECT_EQ(outputs[0].position.w, mode == 8U ? 1.0F : 0.0F) << "Inverse root identity mode=" << mode;
            }
            input.light.w = 1000;
            ASSERT_NO_FATAL_FAILURE(probe(scene, std::span(&input, 1), outputs, 4U));
            EXPECT_EQ(outputs[0].position.w, 0.0F) << "Shift may not invent a root outside the finite caster";
        }
    }
}
