// The GI stage's path reservoir keeps per-pixel state across frames: the emit
// stage of the main kernel (shaders/gi.comp.slang, restirPtEmit) stores a
// pixel's candidates in t_ptParams, and combineMain merges them and adds the
// result to the pixel. A pixel that is background never reaches the emit
// stage. It must then not keep the candidates of a frame in which it showed
// geometry: combineMain would add the light of a surface that is no longer in
// front of the pixel to whatever the pixel shows now. On a dense animated scene
// this left the sun glints of moving structures in the sky after they had passed,
// at the screen positions where those structures had been.

#include <volk/volk.h>

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <slang-math/slang-math.hpp>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "theia/gi/GiPass.hpp"
#include "theia/scene/Scene.hpp"

namespace {

constexpr VkExtent2D kExtent{128, 96};
// The block of pixels that is floor in the first frame and background in the
// frames after it.
constexpr std::uint32_t kBlockX0 = 16U;
constexpr std::uint32_t kBlockX1 = 112U;
constexpr std::uint32_t kBlockY0 = 24U;
constexpr std::uint32_t kBlockY1 = 88U;
// What the HDR image holds before the GI stage runs: a stand-in for what the
// forward pass wrote, different from anything the stage would add by accident.
constexpr sm::float4 kFill{0.125F, 0.25F, 0.5F, 1.0F};

[[nodiscard]] bool inBlock(std::uint32_t x, std::uint32_t y) noexcept {
    return x >= kBlockX0 && x < kBlockX1 && y >= kBlockY0 && y < kBlockY1;
}

/// A blue floor under an emissive quad. The G-buffer stands in for the
/// rasterizer: a pixel records the floor hit on its camera ray, or nothing.
class GiStalePathStateFixture : public RtFixture {
  protected:
    void SetUp() override {
        RtFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        const std::uint32_t floorMaterial =
            scene.addMaterial(harmonia::scene::Material::diffuse(sm::float3(0.05F, 0.15F, 0.8F), 1.0F));
        const std::uint32_t light = scene.addMaterial(harmonia::scene::Material::emissive(sm::float3(1), 10));

        // Wound to face up, like the floor of the GI frame fixtures.
        ASSERT_NO_FATAL_FAILURE(quad({sm::float3(-20, 0, -20), {20, 0, -20}, {20, 0, 20}, {-20, 0, 20}},
                                     {0, 1, 0},
                                     floorMaterial,
                                     true,
                                     floorInstance));
        // An emissive quad above and behind the view, wound to face down.
        std::uint32_t lightInstance = std::numeric_limits<std::uint32_t>::max();
        ASSERT_NO_FATAL_FAILURE(quad({sm::float3(-0.5F, 5, 2.5F), {0.5F, 5, 2.5F}, {0.5F, 5, 3.5F}, {-0.5F, 5, 3.5F}},
                                     {0, -1, 0},
                                     light,
                                     false,
                                     lightInstance));
        EXPECT_NE(lightInstance, std::numeric_limits<std::uint32_t>::max());
        ASSERT_EQ(scene.instanceCount(), 2U);
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
        ASSERT_NO_FATAL_FAILURE(image(hdr, "test.theia.giStale.hdr"));
        ASSERT_NO_FATAL_FAILURE(image(giBuffer, "test.theia.giStale.gbuffer"));
        ASSERT_NO_FATAL_FAILURE(image(normal, "test.theia.giStale.normal"));
        auto buffer = harmonia::core::Buffer::create(deviceCtx(),
                                                     std::size_t{kExtent.width} * kExtent.height * sizeof(sm::float4),
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                                     "test.theia.giStale.inputs");
        ASSERT_TRUE(buffer.has_value());
        inputs = std::move(*buffer);
        camera = sm::float3(0, 2.5F, 7);
        view = sm::lookAt(camera, sm::float3(0, 0.5F, 0), sm::float3(0, 1, 0));
        ASSERT_TRUE(gi.initialize(deviceCtx(),
                                  theia::GiPass::Config{.width = kExtent.width,
                                                        .height = kExtent.height,
                                                        .hdrImage = hdr.handle(),
                                                        .hdrView = hdr.view(),
                                                        .giBufferImage = giBuffer.handle(),
                                                        .giBufferView = giBuffer.view(),
                                                        .gbufferImage = normal.handle(),
                                                        .gbufferView = normal.view(),
                                                        .smsEnabled = false},
                                  commandPool()));
    }

    void TearDown() override {
        gi.shutdown();
        RtFixture::TearDown();
    }

    theia::Scene scene;
    std::uint32_t floorInstance = std::numeric_limits<std::uint32_t>::max();
    sm::float3 camera{};
    sm::float4x4 view{1.0F};

  private:
    void quad(std::array<sm::float3, 4> positions,
              sm::float3 n,
              std::uint32_t material,
              bool reverse,
              std::uint32_t& instance) {
        harmonia::scene::MeshData data;
        for (auto p : positions) {
            harmonia::GpuVertex vertex{};
            vertex.position = p;
            vertex.normal = n;
            data.vertices.push_back(vertex);
        }
        data.indices =
            reverse ? std::vector<std::uint32_t>{0, 3, 2, 0, 2, 1} : std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3};
        const auto mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.theia.giStale.quad");
        ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
        instance = scene.addInstance(mesh, {}, material);
        ASSERT_NE(instance, std::numeric_limits<std::uint32_t>::max());
    }

    void image(harmonia::core::Image& output, const char* label) {
        auto value = harmonia::core::Image::create(
            deviceCtx(),
            kExtent,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            label);
        ASSERT_TRUE(value.has_value());
        output = std::move(*value);
    }

  public:
    /// One GI frame. Every pixel outside `background` records a floor hit on
    /// its camera ray; a background pixel records nothing. The HDR image is
    /// filled with kFill before the stage runs.
    /// Returns the HDR image after the stage.
    std::vector<sm::float4> render(std::uint32_t index, bool blockIsBackground) {
        const float tangent = std::tan(43.0F * std::numbers::pi / 360.0F);
        const float aspect = float(kExtent.width) / float(kExtent.height);
        const sm::float3 forward = sm::normalize(sm::float3(0, 0.5F, 0) - camera);
        const sm::float3 right = sm::normalize(sm::cross(forward, sm::float3(0, 1, 0)));
        const sm::float3 up = sm::cross(right, forward);
        std::vector<sm::float4> gbuffer(std::size_t{kExtent.width} * kExtent.height);
        for (std::uint32_t y = 0; y < kExtent.height; ++y) {
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                sm::float4& record = gbuffer[std::size_t{y} * kExtent.width + x];
                if (blockIsBackground && inBlock(x, y)) {
                    record = sm::float4(0.0F, 0.0F, 0.0F, 0.0F); // the background sentinel
                    continue;
                }
                const float u = (2.0F * (float(x) + 0.5F) / float(kExtent.width) - 1.0F) * tangent * aspect;
                const float v = (1.0F - 2.0F * (float(y) + 0.5F) / float(kExtent.height)) * tangent;
                const sm::float3 direction = sm::normalize(forward + right * u + up * v);
                const float distance = direction.y < 0.0F ? -camera.y / direction.y : 50.0F;
                const sm::float3 world = camera + direction * distance;
                record = sm::float4(world.x, world.y, world.z, std::bit_cast<float>(floorInstance + 1U));
            }
        }
        inputs.uploadData(gbuffer.data(), gbuffer.size() * sizeof(sm::float4));
        auto cmd = commandPool().beginOneShot();
        EXPECT_TRUE(cmd.has_value());
        if (!cmd) {
            return {};
        }
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        // The layout protocol of the GI frame fixtures: the targets end in
        // ATTACHMENT_OPTIMAL, where the rasterizer leaves them; the GI stage
        // moves them to GENERAL, and the read-back leaves the HDR image there.
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
        const VkClearColorValue fill{{kFill.x, kFill.y, kFill.z, kFill.w}};
        vkCmdClearColorImage(*cmd, hdr.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &fill, 1, &range);
        const auto n = view * sm::float4(0, 1, 0, 0);
        const VkClearColorValue encoded{{n.x * 0.5F + 0.5F, n.y * 0.5F + 0.5F, n.z * 0.5F + 0.5F, 1}};
        vkCmdClearColorImage(*cmd, normal.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &encoded, 1, &range);
        for (auto* target : {&hdr, &giBuffer, &normal}) {
            target->transition(*cmd,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        }
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
        params.envSampler = sampler.get();
        params.view = view;
        params.cameraPos = camera;
        params.exposure = 1 / 1.2F;
        params.frameSampleIndex = index;
        params.rngBaseSeed = 7U;
        params.maxDepth = 3;
        // The estimator only, the interactive tiers off exactly as in a
        // capture; the path reservoir on, as the application runs it.
        params.useA3Regularization = false;
        params.adaptiveMaxSamples = 1;
        params.useRestirDi = false;
        params.useRestirPt = true;
        params.useRestirPtPath = true;
        params.fireflyClampEnabled = false;
        params.materialLodEnabled = false;
        params.radianceCacheEnabled = false;
        gi.record(*cmd, params);
        first = false;
        EXPECT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        // The frame is complete, or needs continuations until it is.
        if (gi.completionRequired()) {
            auto ready = gi.frameReady();
            EXPECT_TRUE(ready.has_value());
            if (!ready) {
                return {};
            }
            for (std::uint32_t safety = 0; !(*ready); ++safety) {
                EXPECT_LT(safety, 16384U) << "Test hang guard fails, never accepts a capped estimate";
                if (safety >= 16384U) {
                    return {};
                }
                auto next = commandPool().beginOneShot();
                EXPECT_TRUE(next.has_value());
                if (!next) {
                    return {};
                }
                EXPECT_EQ(gi.recordContinuation(*next), VK_SUCCESS);
                EXPECT_EQ(commandPool().endOneShot(*next), VK_SUCCESS);
                ready = gi.frameReady();
                EXPECT_TRUE(ready.has_value());
                if (!ready) {
                    return {};
                }
            }
        }

        std::vector<sm::float4> pixels(std::size_t{kExtent.width} * kExtent.height);
        auto buffer = harmonia::core::Buffer::create(deviceCtx(),
                                                     pixels.size() * sizeof(sm::float4),
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                     VMA_MEMORY_USAGE_GPU_TO_CPU,
                                                     "test.theia.giStale.readback");
        EXPECT_TRUE(buffer.has_value());
        if (!buffer) {
            return {};
        }
        auto read = commandPool().beginOneShot();
        EXPECT_TRUE(read.has_value());
        if (!read) {
            return {};
        }
        hdr.transition(*read,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT);
        copy = {};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kExtent.width, kExtent.height, 1};
        vkCmdCopyImageToBuffer(*read, hdr.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer->handle(), 1, &copy);
        hdr.transition(*read,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        VkMemoryBarrier2 host{};
        host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        host.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo hostDependency{};
        hostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        hostDependency.memoryBarrierCount = 1;
        hostDependency.pMemoryBarriers = &host;
        vkCmdPipelineBarrier2(*read, &hostDependency);
        EXPECT_EQ(commandPool().endOneShot(*read), VK_SUCCESS);
        EXPECT_EQ(buffer->readData(pixels.data(), pixels.size() * sizeof(sm::float4)), VK_SUCCESS);
        return pixels;
    }

    theia::GiPass gi;
    harmonia::core::Image hdr, giBuffer, normal;
    harmonia::core::Buffer inputs;
    harmonia::core::UniqueSampler sampler;
    bool first = true;
};

} // namespace

// The defect: a pixel that was floor in one frame and is background in the
// next must not receive the stored candidates of its last floor frame.
TEST_F(GiStalePathStateFixture, GiPass_LeavesAPixelThatBecameBackgroundUntouched) {
    // Frame 0: the whole image is floor. The stage stores each pixel's
    // candidates and adds its reservoir sample count to the pixel's alpha.
    const std::vector<sm::float4> floor = render(0U, false);
    ASSERT_EQ(floor.size(), std::size_t{kExtent.width} * kExtent.height);
    // The precondition of the test: every pixel of the block ran the emit
    // stage and carries state that could go stale.
    const std::size_t blockSize = std::size_t{kBlockX1 - kBlockX0} * (kBlockY1 - kBlockY0);
    std::size_t carrying = 0U;
    for (std::uint32_t y = kBlockY0; y < kBlockY1; ++y) {
        for (std::uint32_t x = kBlockX0; x < kBlockX1; ++x) {
            if (floor[std::size_t{y} * kExtent.width + x].w > kFill.w) {
                ++carrying;
            }
        }
    }
    EXPECT_EQ(carrying, blockSize) << "every pixel of the block carries a reservoir sample count";

    // Frames 1 and 2: the block is background, the rest is floor.
    for (std::uint32_t frame = 1U; frame <= 2U; ++frame) {
        const std::vector<sm::float4> pixels = render(frame, true);
        ASSERT_EQ(pixels.size(), floor.size()) << "frame " << frame;
        std::size_t changed = 0U;
        sm::float4 largest{};
        for (std::uint32_t y = kBlockY0; y < kBlockY1; ++y) {
            for (std::uint32_t x = kBlockX0; x < kBlockX1; ++x) {
                const sm::float4& p = pixels[std::size_t{y} * kExtent.width + x];
                if (std::bit_cast<std::array<std::uint32_t, 4>>(p) !=
                    std::bit_cast<std::array<std::uint32_t, 4>>(kFill)) {
                    ++changed;
                    largest = sm::float4(std::max(largest.x, std::abs(p.x - kFill.x)),
                                         std::max(largest.y, std::abs(p.y - kFill.y)),
                                         std::max(largest.z, std::abs(p.z - kFill.z)),
                                         std::max(largest.w, std::abs(p.w - kFill.w)));
                }
            }
        }
        EXPECT_EQ(changed, 0U) << "frame " << frame << ": a background pixel keeps the fill exactly; " << largest.x
                               << ", " << largest.y << ", " << largest.z << ", " << largest.w
                               << " was the largest change in the block";
        // The floor around the block still receives its light.
        std::size_t outsideLit = 0U;
        for (std::uint32_t y = 0U; y < kExtent.height; ++y) {
            for (std::uint32_t x = 0U; x < kExtent.width; ++x) {
                if (inBlock(x, y)) {
                    continue;
                }
                const sm::float4& p = pixels[std::size_t{y} * kExtent.width + x];
                if (p.x > kFill.x || p.y > kFill.y || p.z > kFill.z) {
                    ++outsideLit;
                }
            }
        }
        EXPECT_GT(outsideLit, 0U) << "frame " << frame << ": the floor outside the block is lit";
    }
}
