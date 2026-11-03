// Actual GiPass, native scene/TLAS and common SMS. CPU inputs stand in only for
// deterministic raster camera-ray G-buffers; all transport/roots are production.
#include <volk/volk.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "harmonia/pipeline/AccumulationPass.hpp"
#include "harmonia/pipeline/PassContext.hpp"
#include "harmonia/presentation/ImageCapture.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "theia/gi/GiPass.hpp"
#include "theia/scene/Scene.hpp"

namespace {
constexpr VkExtent2D kExtent{128, 96};
struct Prediction {
    std::size_t pixel;
    double direct, caustic;
    std::array<double, 3> causticRgb{};
};
struct Energy {
    double observed = 0, direct = 0, caustic = 0, mse = 0;
    std::array<double, 3> observedRgb{}, expectedRgb{};
};
double radicalInverse(std::uint32_t value, std::uint32_t base) {
    double result = 0, scale = 1;
    while (value != 0) {
        scale /= base;
        result += scale * (value % base);
        value /= base;
    }
    return result;
}
} // namespace

class SmsGiFrameFixture : public RtFixture {
  protected:
    virtual bool glassSlab() const noexcept { return false; }
    virtual bool absorbing() const noexcept { return false; }
    virtual bool specularPrefix() const noexcept { return false; }
    virtual bool pathReuse() const noexcept { return false; }
    virtual std::uint32_t partitionResolution() const noexcept { return 1; }
    virtual bool pilotPrior() const noexcept { return false; }
    virtual std::uint32_t trialBudget() const noexcept { return 1; }
    virtual bool spatialReuse() const noexcept { return false; }
    virtual std::uint32_t surfaceBudget() const noexcept { return glassSlab() || specularPrefix() ? 4U : 3U; }
    sm::float3 cameraPosition() const { return glassSlab() ? sm::float3(2, .5F, 3) : sm::float3(2, 3, 6); }
    float fov() const { return glassSlab() ? 10.0F : 43.0F; }
    void SetUp() override {
        RtFixture::SetUp();
        if (IsSkipped())
            return;
        auto diffuse = harmonia::scene::Material::diffuse(sm::float3(.5F), 0).gpu();
        diffuse.specularColorWeight.w = 0;
        const auto floor = scene.addMaterial(harmonia::scene::Material::fromGpu(diffuse));
        auto caster =
            glassSlab() ? harmonia::scene::Material::glass(1.5F, 0) : harmonia::scene::Material::mirror(sm::float3(1));
        if (absorbing()) {
            auto gpu = caster.gpu();
            gpu.transmissionParams.x = 1;
            gpu.transmissionColorWeight = {.6F, .8F, .9F, 1};
            gpu.transmissionScatter = sm::float4(0);
            caster = harmonia::scene::Material::fromGpu(gpu);
        }
        const auto mirror = scene.addMaterial(std::move(caster));
        const auto light = scene.addMaterial(harmonia::scene::Material::emissive(sm::float3(1), 10));
        if (specularPrefix())
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(.25F, 0, 4.25F), {4, 0, 4.25F}, {4, 0, 8}, {.25F, 0, 8}}, {0, 1, 0}, floor, true));
        else
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(.25F, 0, -2), {4, 0, -2}, {4, 0, 2}, {.25F, 0, 2}}, {0, 1, 0}, floor, true));
        if (glassSlab()) {
            const auto mesh = scene.addMesh(deviceCtx(),
                                            commandPool(),
                                            harmonia::scene::ProceduralGeometry::makeBox({6, .25F, 5}),
                                            {},
                                            "test.theia.sms.closed.slab");
            ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
            ASSERT_NE(scene.addInstance(mesh, harmonia::scene::Transform{.translation = {2, 1.25F, 0}}, mirror),
                      std::numeric_limits<std::uint32_t>::max());
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(1.75F, 3, -.25F), {2.25F, 3, -.25F}, {2.25F, 3, .25F}, {1.75F, 3, .25F}},
                     {0, -1, 0},
                     light,
                     false));
        } else if (specularPrefix()) {
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(0, 0, 4), {0, 3, 4}, {0, 3, 8}, {0, 0, 8}}, {1, 0, 0}, mirror, false));
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(3, 1.75F, 5.75F), {3, 1.75F, 6.25F}, {3, 2.25F, 6.25F}, {3, 2.25F, 5.75F}},
                     {-1, 0, 0},
                     light,
                     false));
            auto gray = harmonia::scene::Material::mirror(sm::float3(.4F)).gpu();
            gray.specularColorWeight = {0, 0, 0, 1};
            const auto prefix = scene.addMaterial(harmonia::scene::Material::fromGpu(gray));
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(.25F, 0, 4), {4, 0, 4}, {4, 4, 4}, {.25F, 4, 4}}, {0, 0, 1}, prefix, false));
        } else {
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(0, 0, -2), {0, 3, -2}, {0, 3, 2}, {0, 0, 2}}, {1, 0, 0}, mirror, false));
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(3, 1.75F, -.25F), {3, 1.75F, .25F}, {3, 2.25F, .25F}, {3, 2.25F, -.25F}},
                     {-1, 0, 0},
                     light,
                     false));
        }
        ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
        ASSERT_EQ(scene.emissiveTriangleCount(), 2U);
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        VkSampler rawSampler{};
        ASSERT_EQ(vkCreateSampler(deviceCtx().device, &samplerInfo, nullptr, &rawSampler), VK_SUCCESS);
        sampler = harmonia::core::UniqueSampler{deviceCtx().device, rawSampler};
        ASSERT_NO_FATAL_FAILURE(
            image(hdr, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "test.theia.sms.hdr"));
        ASSERT_NO_FATAL_FAILURE(image(giBuffer, VK_IMAGE_USAGE_SAMPLED_BIT, "test.theia.sms.gbuffer"));
        ASSERT_NO_FATAL_FAILURE(image(normal, VK_IMAGE_USAGE_SAMPLED_BIT, "test.theia.sms.normal"));
        auto buffer = harmonia::core::Buffer::create(deviceCtx(),
                                                     std::size_t{kExtent.width} * kExtent.height * sizeof(sm::float4),
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                                     "test.theia.sms.camera.inputs");
        ASSERT_TRUE(buffer.has_value());
        inputs = std::move(*buffer);
        view = sm::lookAt(cameraPosition(), sm::float3(2, 0, specularPrefix() ? 2.0F : 0.0F), sm::float3(0, 1, 0));
        auto projection = sm::perspective(sm::radians(fov()), float(kExtent.width) / kExtent.height, .01F, 100.0F);
        projection[1][1] *= -1;
        viewProjection = projection * view;
        ASSERT_NO_FATAL_FAILURE(createGi());
        auto history = harmonia::pipeline::AccumulationPass::create(
            deviceCtx(), kExtent, std::filesystem::path(HARMONIA_SHADER_DIR) / "accumulation.spv");
        ASSERT_TRUE(history.has_value());
        accumulation = std::move(*history);
        buildPredictions();
    }
    void createGi() {
        const theia::GiPass::Config config{.width = kExtent.width,
                                           .height = kExtent.height,
                                           .hdrImage = hdr.handle(),
                                           .hdrView = hdr.view(),
                                           .giBufferImage = giBuffer.handle(),
                                           .giBufferView = giBuffer.view(),
                                           .gbufferImage = normal.handle(),
                                           .gbufferView = normal.view(),
                                           .smsEnabled = true,
                                           .smsPartitionResolution = partitionResolution(),
                                           .smsPriorEnabled = pilotPrior(),
                                           .smsTrialBudget = trialBudget(),
                                           .smsSpatialReuse = spatialReuse()};
        ASSERT_TRUE(gi.initialize(deviceCtx(), config, commandPool()));
    }
    void quad(std::array<sm::float3, 4> positions, sm::float3 n, std::uint32_t material, bool reverse) {
        harmonia::scene::MeshData data;
        for (auto p : positions) {
            harmonia::GpuVertex vertex{};
            vertex.position = p;
            vertex.normal = n;
            data.vertices.push_back(vertex);
        }
        data.indices =
            reverse ? std::vector<std::uint32_t>{0, 3, 2, 0, 2, 1} : std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3};
        const auto mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.theia.sms.quad");
        ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
        ASSERT_NE(scene.addInstance(mesh, {}, material), std::numeric_limits<std::uint32_t>::max());
    }
    void image(harmonia::core::Image& output, VkImageUsageFlags extra, const char* label) {
        auto value =
            harmonia::core::Image::create(deviceCtx(),
                                          kExtent,
                                          VK_FORMAT_R32G32B32A32_SFLOAT,
                                          extra | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                          VK_IMAGE_ASPECT_COLOR_BIT,
                                          label);
        ASSERT_TRUE(value.has_value());
        output = std::move(*value);
    }
    std::array<double, 3> floorPoint(std::uint32_t x, std::uint32_t y, double jx, double jy) const {
        const double height = glassSlab() ? .5 : 3, originZ = glassSlab() ? 3 : 6;
        const double tangent = std::tan(double(fov()) * std::numbers::pi / 360);
        const double forwardZ = specularPrefix() ? 4 : originZ;
        const double sine = height / std::hypot(height, forwardZ), cosine = forwardZ / std::hypot(height, forwardZ);
        const double u = (2 * (x + jx) / kExtent.width - 1) * tangent * kExtent.width / kExtent.height;
        const double v = (1 - 2 * (y + jy) / kExtent.height) * tangent;
        const double dy = -sine + v * cosine, dz = -cosine - v * sine, distance = -height / dy;
        return {2 + u * distance, specularPrefix() ? 2 - dz * distance : originZ + dz * distance, distance};
    }
    void frame(std::uint32_t index, std::uint32_t seed, std::uint64_t& continuations, std::uint64_t resetToken = 0) {
        std::vector<sm::float4> camera(std::size_t{kExtent.width} * kExtent.height, sm::float4(0));
        const double jx = radicalInverse(index + 1, 2), jy = radicalInverse(index + 1, 3);
        for (std::uint32_t y = 0; y < kExtent.height; ++y)
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                if (specularPrefix()) {
                    const double tangent = std::tan(double(fov()) * std::numbers::pi / 360);
                    const double u = (2 * (x + jx) / kExtent.width - 1) * tangent * kExtent.width / kExtent.height;
                    const double v = (1 - 2 * (y + jy) / kExtent.height) * tangent, dy = -.6 + .8 * v,
                                 dz = -.8 - .6 * v, t = -2 / dz;
                    const double mx = 2 + u * t, my = 3 + dy * t;
                    if (t > 0 && mx > .25 && mx < 4 && my > 0 && my < 4)
                        camera[std::size_t{y} * kExtent.width + x] = {
                            static_cast<float>(mx), static_cast<float>(my), 4, std::bit_cast<float>(4U)};
                    continue;
                }
                const auto p = floorPoint(x, y, jx, jy);
                if (p[2] > 0 && p[0] > .25 && p[0] < 4 && std::abs(p[1]) < 2)
                    camera[std::size_t{y} * kExtent.width + x] = {
                        static_cast<float>(p[0]), 0, static_cast<float>(p[1]), std::bit_cast<float>(1U)};
            }
        inputs.uploadData(camera.data(), camera.size() * sizeof(sm::float4));
        auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (auto* target : {&hdr, &giBuffer, &normal}) {
            const auto old =
                first ? VK_IMAGE_LAYOUT_UNDEFINED
                      : (target == &hdr ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            target->transition(*cmd,
                               old,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT);
        }
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kExtent.width, kExtent.height, 1};
        vkCmdCopyBufferToImage(
            *cmd, inputs.handle(), giBuffer.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        const VkClearColorValue zero{};
        vkCmdClearColorImage(*cmd, hdr.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
        const auto n = view * (specularPrefix() ? sm::float4(0, 0, 1, 0) : sm::float4(0, 1, 0, 0));
        const VkClearColorValue encoded{{n.x * .5F + .5F, n.y * .5F + .5F, n.z * .5F + .5F, 1}};
        vkCmdClearColorImage(*cmd, normal.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &encoded, 1, &range);
        for (auto* target : {&hdr, &giBuffer, &normal})
            target->transition(*cmd,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkMemoryBarrier2 memory{};
        memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                               VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(*cmd, &dependency);
        theia::GiPass::FrameParams params{};
        params.scene = &scene;
        params.envSampler = sampler;
        params.view = view;
        params.prevViewProj = viewProjection;
        params.cameraPos = cameraPosition();
        params.exposure = 1 / 1.2F;
        params.frameSampleIndex = index;
        params.rngBaseSeed = seed;
        params.maxDepth = surfaceBudget();
        params.useA3Regularization = false;
        params.adaptiveMaxSamples = 1;
        params.useRestirDi = false;
        params.useRestirPt = true;
        params.useRestirPtPath = pathReuse();
        params.fireflyClampEnabled = false;
        gi.record(*cmd, params);
        first = false;
        ASSERT_TRUE(gi.completionRequired());
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        auto ready = gi.frameReady();
        ASSERT_TRUE(ready.has_value());
        if (specularPrefix() && surfaceBudget() < 4)
            EXPECT_TRUE(*ready);
        else
            EXPECT_FALSE(*ready);
        for (std::uint32_t safety = 0; !(*ready); ++safety) {
            ASSERT_LT(safety, 16384U) << "Test hang guard fails, never accepts a capped estimate";
            auto next = commandPool().beginOneShot();
            ASSERT_TRUE(next.has_value());
            ASSERT_EQ(gi.recordContinuation(*next), VK_SUCCESS);
            ASSERT_EQ(commandPool().endOneShot(*next), VK_SUCCESS);
            ++continuations;
            ready = gi.frameReady();
            ASSERT_TRUE(ready.has_value());
        }
        auto post = commandPool().beginOneShot();
        ASSERT_TRUE(post.has_value());
        accumulation.record(
            harmonia::pipeline::PassContext{.cmd = *post,
                                            .frameIndex = index,
                                            .extent = kExtent,
                                            .fixedView = true,
                                            .accumulationResetToken = resetToken == 0 ? seed : resetToken,
                                            .hdrBuffer = &hdr});
        ASSERT_EQ(commandPool().endOneShot(*post), VK_SUCCESS);
    }
    void readPixels(std::vector<sm::float4>& pixels) {
        pixels.resize(std::size_t{kExtent.width} * kExtent.height);
        auto buffer = harmonia::core::Buffer::create(deviceCtx(),
                                                     pixels.size() * sizeof(sm::float4),
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                     VMA_MEMORY_USAGE_GPU_TO_CPU,
                                                     "test.theia.sms.readback");
        ASSERT_TRUE(buffer.has_value());
        auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        hdr.transition(*cmd,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kExtent.width, kExtent.height, 1};
        vkCmdCopyImageToBuffer(*cmd, hdr.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer->handle(), 1, &copy);
        hdr.transition(*cmd,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        VkMemoryBarrier2 memory{};
        memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(*cmd, &dependency);
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        ASSERT_EQ(buffer->readData(pixels.data(), pixels.size() * sizeof(sm::float4)), VK_SUCCESS);
    }
    void buildPredictions() {
        if (glassSlab()) {
            buildGlassPredictions();
            return;
        }
        constexpr int subdivisions = 24;
        for (std::uint32_t y = 0; y < kExtent.height; ++y)
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                const auto p = floorPoint(x, y, .5, .5);
                const double zOffset = specularPrefix() ? 6 : 0;
                if (!(p[2] > 0 && p[0] > .6 && p[0] < 2.7 && std::abs(p[1] - zOffset) < 1.5))
                    continue;
                double direct = 0, caustic = 0;
                for (int iy = 0; iy < subdivisions; ++iy)
                    for (int iz = 0; iz < subdivisions; ++iz) {
                        const double ly = 1.75 + (iy + .5) * .5 / subdivisions,
                                     lz = zOffset - .25 + (iz + .5) * .5 / subdivisions;
                        const double dy2 = ly * ly, dz2 = (lz - p[1]) * (lz - p[1]), dx = 3 - p[0], rx = -3 - p[0];
                        const double directR2 = dx * dx + dy2 + dz2, mirrorR2 = rx * rx + dy2 + dz2;
                        const double area = .25 / (subdivisions * subdivisions);
                        direct += area * ly * dx / (directR2 * directR2);
                        caustic += area * ly * (-rx) / (mirrorR2 * mirrorR2);
                        if (specularPrefix()) {
                            const double rz2 = dx * dx + dy2 + (8 - lz - p[1]) * (8 - lz - p[1]);
                            caustic += .4 * area * ly * dx / (rz2 * rz2);
                        }
                    }
                const double scale = (specularPrefix() ? .4 : 1) * .5 / std::numbers::pi * 10 / 1.2;
                predictions.push_back({std::size_t{y} * kExtent.width + x,
                                       direct * scale,
                                       caustic * scale,
                                       {caustic * scale, caustic * scale, caustic * scale}});
            }
        ASSERT_GT(predictions.size(), 500U);
    }
    void buildGlassPredictions() {
        constexpr std::array nodes{-.9602898564975363,
                                   -.7966664774136267,
                                   -.5255324099163290,
                                   -.1834346424956498,
                                   .1834346424956498,
                                   .5255324099163290,
                                   .7966664774136267,
                                   .9602898564975363};
        constexpr std::array weights{.1012285362903763,
                                     .2223810344533745,
                                     .3137066458778873,
                                     .3626837833783620,
                                     .3626837833783620,
                                     .3137066458778873,
                                     .2223810344533745,
                                     .1012285362903763};
        constexpr std::array colors{.6F, .8F, .9F};
        constexpr double airHeight = 2.5, eta = 1.5, thickness = .5, scale = .5 / std::numbers::pi * 10 / 1.2;
        double maximumFresnel = 0, minimumPrediction = 1e30;
        for (std::uint32_t y = 0; y < kExtent.height; ++y)
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                const auto p = floorPoint(x, y, .5, .5);
                if (!(p[2] > 0 && p[0] > .6 && p[0] < 2.7 && std::abs(p[1]) < 1.5))
                    continue;
                std::array<double, 3> value{};
                for (std::size_t ix = 0; ix < nodes.size(); ++ix)
                    for (std::size_t iz = 0; iz < nodes.size(); ++iz) {
                        const double rho = std::hypot(2 + .25 * nodes[ix] - p[0], .25 * nodes[iz] - p[1]);
                        for (int pairs = 0; pairs < (absorbing() ? 4 : 1); ++pairs) {
                            const double depth = (2 * pairs + 1) * thickness;
                            double lo = 0, hi = rho / airHeight;
                            for (int iteration = 0; iteration < 64; ++iteration) {
                                const double t = (lo + hi) * .5, D = std::sqrt(eta * eta + (eta * eta - 1) * t * t);
                                if (airHeight * t + depth * t / D < rho)
                                    lo = t;
                                else
                                    hi = t;
                            }
                            const double t = (lo + hi) * .5, D = std::sqrt(eta * eta + (eta * eta - 1) * t * t);
                            const double derivative = airHeight + depth * eta * eta / (D * D * D);
                            const double geometric = rho == 0 ? 1 / std::pow(airHeight + depth / eta, 2)
                                                              : t / (rho * std::pow(1 + t * t, 1.5) * derivative);
                            const double ca = 1 / std::sqrt(1 + t * t), cg = std::sqrt(1 - (1 - ca * ca) / (eta * eta));
                            const double rs = (ca - eta * cg) / (ca + eta * cg), rp = (eta * ca - cg) / (eta * ca + cg),
                                         F = .5 * (rs * rs + rp * rp);
                            maximumFresnel = std::max(maximumFresnel, F);
                            const double common = weights[ix] * weights[iz] * .0625 * ca * geometric * (1 - F) *
                                                  (1 - F) * std::pow(F, 2 * pairs) * scale;
                            for (std::size_t c = 0; c < 3; ++c)
                                value[c] +=
                                    common * (absorbing() ? std::exp(std::log(double(colors[c])) * depth / cg) : 1);
                        }
                    }
                for (double channel : value)
                    minimumPrediction = std::min(minimumPrediction, channel);
                predictions.push_back(
                    {std::size_t{y} * kExtent.width + x, 0, (value[0] + value[1] + value[2]) / 3, value});
            }
        ASSERT_GT(predictions.size(), 500U);
        EXPECT_LT(maximumFresnel, .1);
        if (absorbing()) {
            const double bound = scale * .25 / (airHeight * airHeight) * std::pow(.1, 8) / (1 - .1 * .1);
            EXPECT_LT(bound / minimumPrediction, 1e-6);
        }
    }
    Energy energy(const std::vector<sm::float4>& pixels) const {
        Energy result;
        for (const auto& reference : predictions) {
            const auto value = pixels[reference.pixel];
            EXPECT_TRUE(std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z));
            const double observed = (double(value.x) + value.y + value.z) / 3;
            result.observed += observed;
            result.direct += reference.direct;
            result.caustic += reference.caustic;
            result.mse += std::pow(observed - reference.direct - reference.caustic, 2);
            const std::array<double, 3> rgb{value.x, value.y, value.z};
            for (std::size_t c = 0; c < 3; ++c) {
                result.observedRgb[c] += rgb[c];
                result.expectedRgb[c] += reference.direct + reference.causticRgb[c];
            }
        }
        return result;
    }
    theia::Scene scene;
    harmonia::core::UniqueSampler sampler;
    harmonia::core::Image hdr, giBuffer, normal;
    harmonia::core::Buffer inputs;
    theia::GiPass gi;
    harmonia::pipeline::AccumulationPass accumulation;
    sm::float4x4 view{1}, viewProjection{1};
    bool first = true;
    std::vector<Prediction> predictions;
};

class SmsGiGlassFrameFixture : public SmsGiFrameFixture {
  protected:
    bool glassSlab() const noexcept override { return true; }
};
class SmsGiAbsorptionFrameFixture : public SmsGiGlassFrameFixture {
  protected:
    bool absorbing() const noexcept override { return true; }
};

class SmsGiSdsFrameFixture : public SmsGiFrameFixture {
  protected:
    bool specularPrefix() const noexcept override { return true; }
};
class SmsGiSdsDepthFrameFixture : public SmsGiSdsFrameFixture {
  protected:
    std::uint32_t surfaceBudget() const noexcept override { return 3; }
};

class SmsGiConfiguredFrameFixture : public SmsGiFrameFixture {
  protected:
    bool pathReuse() const noexcept override { return true; }
    std::uint32_t partitionResolution() const noexcept override { return 16; }
    bool pilotPrior() const noexcept override { return true; }
    std::uint32_t trialBudget() const noexcept override { return 32; }
};

TEST_F(SmsGiConfiguredFrameFixture, ProductionConfigurationPreservesLocalMirrorRadiometry) {
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        if (seed != 17U) {
            gi.shutdown();
            ASSERT_NO_FATAL_FAILURE(createGi());
        }
        std::uint64_t continuations = 0;
        for (std::uint32_t index = 0; index < 512; ++index)
            ASSERT_NO_FATAL_FAILURE(frame(index, seed, continuations));
        std::vector<sm::float4> pixels;
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = energy(pixels);
        std::cout << "Theia configured SMS seed=" << seed
                  << " caustic ratio=" << (measured.observed - measured.direct) / measured.caustic
                  << " continuations=" << continuations << '\n';
        for (std::size_t channel = 0; channel < 3; ++channel)
            EXPECT_NEAR((measured.observedRgb[channel] - measured.direct) / measured.caustic, 1.0, .02)
                << "channel=" << channel;
        EXPECT_GE(continuations, 512U);
    }
}

class SmsGiReuseFrameFixture : public SmsGiConfiguredFrameFixture {
  protected:
    bool spatialReuse() const noexcept override { return reuse; }
    bool reuse = false;
};

TEST_F(SmsGiReuseFrameFixture, SpatialMirrorPreservesEnergyAndImprovesSameBudgetError) {
    double localMse = 0, reuseMse = 0;
    for (auto seed : {17U, 91U, 1234U, 2026U})
        for (bool enabled : {false, true}) {
            gi.shutdown();
            reuse = enabled;
            ASSERT_NO_FATAL_FAILURE(createGi());
            std::uint64_t continuations = 0;
            const auto token = std::uint64_t{seed} + (enabled ? 0x100000000ULL : 0ULL);
            std::vector<sm::float4> pixels;
            for (std::uint32_t index = 0; index < 512; ++index) {
                ASSERT_NO_FATAL_FAILURE(frame(index, seed, continuations, token));
                if (index == 127) {
                    ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
                    if (enabled)
                        reuseMse += energy(pixels).mse;
                    else
                        localMse += energy(pixels).mse;
                }
            }
            ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
            const auto measured = energy(pixels);
            std::cout << "Theia spatial SMS seed=" << seed << " enabled=" << enabled
                      << " caustic ratio=" << (measured.observed - measured.direct) / measured.caustic << '\n';
            for (std::size_t channel = 0; channel < 3; ++channel)
                EXPECT_NEAR((measured.observedRgb[channel] - measured.direct) / measured.caustic, 1.0, .02)
                    << "channel=" << channel;
        }
    std::cout << "Theia spatial SMS mirror MSE reuse/local=" << reuseMse / localMse << '\n';
    EXPECT_LT(reuseMse, localMse);
}

TEST_F(SmsGiSdsFrameFixture, SpecularCameraPrefixMatchesIndependentVirtualEmitterEnergy) {
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        if (seed != 17U) {
            gi.shutdown();
            ASSERT_NO_FATAL_FAILURE(createGi());
        }
        std::uint64_t continuations = 0;
        for (std::uint32_t index = 0; index < 512; ++index)
            ASSERT_NO_FATAL_FAILURE(frame(index, seed, continuations));
        std::vector<sm::float4> pixels;
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = energy(pixels);
        const double ratio = (measured.observed - measured.direct) / measured.caustic;
        std::cout << "Theia local SMS SDS seed=" << seed << " caustic ratio=" << ratio << '\n';
        for (std::size_t channel = 0; channel < 3; ++channel)
            EXPECT_NEAR((measured.observedRgb[channel] - measured.direct) / measured.caustic, 1.0, .02)
                << "SDS channel=" << channel;
        EXPECT_GT(continuations, 512U);
    }
}
TEST_F(SmsGiSdsDepthFrameFixture, InsufficientReceiverBudgetRetainsOnlyDirectPrefixLighting) {
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        if (seed != 17U) {
            gi.shutdown();
            ASSERT_NO_FATAL_FAILURE(createGi());
        }
        std::uint64_t continuations = 0;
        for (std::uint32_t index = 0; index < 64; ++index)
            ASSERT_NO_FATAL_FAILURE(frame(index, seed, continuations));
        std::vector<sm::float4> pixels;
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = energy(pixels);
        for (std::size_t channel = 0; channel < 3; ++channel)
            EXPECT_NEAR((measured.observedRgb[channel] - measured.direct) / measured.caustic, 0.0, .02)
                << "Receiver-relative cutoff must not retain an R+L tail; channel=" << channel;
    }
}

TEST_F(SmsGiGlassFrameFixture, LocalClosedGlassMatchesIndependentSnellIntegral) {
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        if (seed != 17U) {
            gi.shutdown();
            ASSERT_NO_FATAL_FAILURE(createGi());
        }
        std::uint64_t continuations = 0;
        for (std::uint32_t index = 0; index < 512; ++index)
            ASSERT_NO_FATAL_FAILURE(frame(index, seed, continuations));
        std::vector<sm::float4> pixels;
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = energy(pixels);
        const double ratio = measured.observed / measured.caustic;
        std::cout << "Theia local SMS glass seed=" << seed << " ratio=" << ratio << '\n';
        EXPECT_NEAR(ratio, 1.0, .02);
        for (std::size_t channel = 0; channel < 3; ++channel)
            EXPECT_NEAR(measured.observedRgb[channel] / measured.expectedRgb[channel], 1.0, .02)
                << "Clear-glass energy must hold per channel; channel=" << channel;
        EXPECT_GT(continuations, 512U);
    }
}
TEST_F(SmsGiAbsorptionFrameFixture, LocalPureAbsorberMatchesIndependentSnellBeerRgbIntegral) {
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        if (seed != 17U) {
            gi.shutdown();
            ASSERT_NO_FATAL_FAILURE(createGi());
        }
        std::uint64_t continuations = 0;
        for (std::uint32_t index = 0; index < 512; ++index)
            ASSERT_NO_FATAL_FAILURE(frame(index, seed, continuations));
        std::vector<sm::float4> pixels;
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = energy(pixels);
        std::cout << "Theia local SMS absorption seed=" << seed << " RGB ratios=";
        for (std::size_t c = 0; c < 3; ++c) {
            const double ratio = measured.observedRgb[c] / measured.expectedRgb[c];
            std::cout << ratio << ' ';
            EXPECT_NEAR(ratio, 1.0, .02) << "channel=" << c;
        }
        std::cout << '\n';
        EXPECT_GT(continuations, 512U);
    }
}

TEST_F(SmsGiFrameFixture, LocalMirrorMatchesIndependentEnergyAndWaitsBeforeAccumulation) {
    double shortMse = 0, longMse = 0;
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        if (seed != 17U) {
            gi.shutdown();
            ASSERT_NO_FATAL_FAILURE(createGi());
        }
        std::uint64_t continuations = 0;
        std::vector<sm::float4> pixels;
        for (std::uint32_t index = 0; index < 512; ++index) {
            ASSERT_NO_FATAL_FAILURE(frame(index, seed, continuations));
            if (index == 31 || index == 127) {
                ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
                if (index == 31)
                    shortMse += energy(pixels).mse;
                else
                    longMse += energy(pixels).mse;
            }
        }
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = energy(pixels);
        const double ratio = (measured.observed - measured.direct) / measured.caustic;
        std::cout << "Theia local SMS mirror seed=" << seed << " caustic ratio=" << ratio
                  << " continuations=" << continuations << '\n';
        EXPECT_NEAR(ratio, 1.0, .02);
        for (std::size_t channel = 0; channel < 3; ++channel)
            EXPECT_NEAR((measured.observedRgb[channel] - measured.direct) / measured.caustic, 1.0, .02)
                << "Mirror reflected energy must hold per channel; channel=" << channel;
        EXPECT_GT(continuations, 512U);
        const auto output = std::filesystem::path(THEIA_TEST_SHADER_DIR).parent_path() /
                            ("sms_gi_mirror_seed" + std::to_string(seed) + ".exr");
        ASSERT_TRUE(harmonia::presentation::ImageCapture::saveExr(deviceCtx(), commandPool(), hdr, output));
    }
    std::cout << "Theia local SMS mirror MSE 128/32=" << longMse / shortMse << '\n';
    EXPECT_LT(longMse, shortMse * .65);
}

TEST_F(SmsGiFrameFixture, InitializationFailureCannotPublishInitializedOrReadyState) {
    gi.shutdown();
    const theia::GiPass::Config invalid{.width = kExtent.width,
                                        .height = kExtent.height,
                                        .hdrImage = hdr.handle(),
                                        .hdrView = hdr.view(),
                                        .giBufferImage = giBuffer.handle(),
                                        .giBufferView = giBuffer.view(),
                                        .gbufferImage = normal.handle(),
                                        .gbufferView = normal.view(),
                                        .smsEnabled = true,
                                        .smsPartitionResolution = 0};
    EXPECT_FALSE(gi.initialize(deviceCtx(), invalid, commandPool()));
    EXPECT_FALSE(gi.isInitialized());
    EXPECT_TRUE(gi.completionRequired());
    const auto ready = gi.frameReady();
    ASSERT_FALSE(ready.has_value());
    EXPECT_EQ(ready.error(), VK_ERROR_INITIALIZATION_FAILED);
    EXPECT_EQ(gi.recordContinuation(VK_NULL_HANDLE), VK_ERROR_INITIALIZATION_FAILED);
}
