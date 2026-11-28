// The GI stage re-traces each pixel's primary surface through the scene TLAS
// (shaders/gi.comp.slang, buildPrimarySurface) and trusts that hit over the one
// the rasterizer recorded. The shared update policy (harmonia::scene_rt::updateTlas)
// replaces the TLAS with a fresh acceleration structure on every full rebuild: on
// every update of a static scene and on every Scene::kTlasRebuildInterval-th update of an animated one. A
// animated one. A frame slot whose descriptor still holds the replaced structure traces a frozen
// frozen pose. The real GiPass must follow the scene's current TLAS in both slots.

#include <volk/volk.h>

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <slang-math/slang-math.hpp>
#include <string_view>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "harmonia/scene/SceneLoader.hpp"
#include "theia/gi/GiPass.hpp"
#include "theia/scene/Scene.hpp"

namespace {

constexpr VkExtent2D kExtent{128, 96};
// Column ranges of the three poses the mover takes: x = -2, 0 and +2 metres.
constexpr std::uint32_t kCenterBegin = 48U;
constexpr std::uint32_t kRightBegin = 80U;
// A mover fills about 600 pixels at this distance; a region holds it or nothing.
constexpr std::size_t kPresent = 300U;

/// The scene under test, shared by both rebuild paths: a blue floor, a red
/// mover and an emissive quad above and behind the view.
class GiTlasRebuildFixture : public RtFixture {
  protected:
    virtual bool animated() const noexcept { return false; }

    void SetUp() override {
        RtFixture::SetUp();
        if (IsSkipped())
            return;
        const std::uint32_t floorMaterial =
            scene.addMaterial(harmonia::scene::Material::diffuse(sm::float3(0.05F, 0.15F, 0.8F), 1.0F));
        const std::uint32_t light = scene.addMaterial(harmonia::scene::Material::emissive(sm::float3(1), 10));

        if (animated()) {
            // The loader attaches the translation track, which makes the TLAS
            // update-capable: the first Scene::kTlasRebuildInterval updates refit, the one after rebuilds.
            ASSERT_TRUE(loadAnimatedMover());
        } else {
            mover = scene.addMesh(deviceCtx(),
                                  commandPool(),
                                  harmonia::scene::ProceduralGeometry::makeBox(sm::float3(0.75F)),
                                  {},
                                  "test.theia.giTlas.mover");
            ASSERT_NE(mover, std::numeric_limits<std::uint32_t>::max());
            const std::uint32_t red =
                scene.addMaterial(harmonia::scene::Material::diffuse(sm::float3(0.63F, 0.065F, 0.05F), 1.0F));
            ASSERT_NE(scene.addInstance(mover, harmonia::scene::Transform{.translation = {-2.0F, 0.75F, 0.0F}}, red),
                      std::numeric_limits<std::uint32_t>::max());
        }
        // Wound to face up, like the floor of the SMS GI-frame fixture.
        ASSERT_NO_FATAL_FAILURE(quad({sm::float3(-20, 0, -20), {20, 0, -20}, {20, 0, 20}, {-20, 0, 20}},
                                     {0, 1, 0},
                                     floorMaterial,
                                     true,
                                     floorInstance));
        // Wound to face down, toward the floor and the mover.
        std::uint32_t lightInstance = std::numeric_limits<std::uint32_t>::max();
        ASSERT_NO_FATAL_FAILURE(quad({sm::float3(-0.5F, 5, 2.5F), {0.5F, 5, 2.5F}, {0.5F, 5, 3.5F}, {-0.5F, 5, 3.5F}},
                                     {0, -1, 0},
                                     light,
                                     false,
                                     lightInstance));
        EXPECT_NE(lightInstance, std::numeric_limits<std::uint32_t>::max());
        ASSERT_EQ(scene.instanceCount(), 3U);
        ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
        EXPECT_EQ(scene.hasAnimation(), animated());
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
        ASSERT_NO_FATAL_FAILURE(image(hdr, "test.theia.giTlas.hdr"));
        ASSERT_NO_FATAL_FAILURE(image(giBuffer, "test.theia.giTlas.gbuffer"));
        ASSERT_NO_FATAL_FAILURE(image(normal, "test.theia.giTlas.normal"));
        auto buffer = harmonia::core::Buffer::create(deviceCtx(),
                                                     std::size_t{kExtent.width} * kExtent.height * sizeof(sm::float4),
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                                     "test.theia.giTlas.inputs");
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
        if (!dir.empty()) {
            std::filesystem::remove_all(dir);
        }
        RtFixture::TearDown();
    }

    /// Counts the mover's pixels per pose region in the HDR image.
    struct Poses {
        std::size_t left = 0, center = 0, right = 0;
    };
    Poses render() {
        recordFrame(frameIndex++);
        std::vector<sm::float4> pixels;
        readPixels(pixels);
        Poses poses{};
        for (std::uint32_t y = 0; y < kExtent.height; ++y) {
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                const sm::float4& p = pixels[std::size_t{y} * kExtent.width + x];
                // The mover's red albedo keeps green at a small fraction of
                // red. The floor is blue: neither where the light reaches it
                // nor in the mover's shadow, which the mover's red bounce
                // lights, does it reflect more red than green.
                if (!(p.x > 1.0e-4F && p.y < 0.35F * p.x)) {
                    continue;
                }
                if (x < kCenterBegin) {
                    ++poses.left;
                } else if (x < kRightBegin) {
                    ++poses.center;
                } else {
                    ++poses.right;
                }
            }
        }
        return poses;
    }

    theia::Scene scene;
    std::uint32_t mover = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t floorInstance = std::numeric_limits<std::uint32_t>::max();
    std::filesystem::path dir;
    sm::float3 camera{};
    sm::float4x4 view{1.0F};

  private:
    /// A sphere on a linear translation track from x = -2 to x = +2 over one
    /// second (ANI1/ANI2 TOML schema). Its material comes from a library
    /// written next to the scene, with the same red as the static mover.
    bool loadAnimatedMover() {
        namespace fs = std::filesystem;
        dir = fs::temp_directory_path() / "theia_gi_tlas_rebuild_test";
        fs::create_directories(dir);
        constexpr std::string_view kLibrary = R"toml(model = "openpbr"
colorspace = "lin_rec709_scene"

[MoverRed]
base_color = [0.630, 0.065, 0.050]
specular_roughness = 1.0
)toml";
        constexpr std::string_view kScene = R"toml(material_libraries = ["mover.materials.toml"]

[animation_clock]
fps = 30

[camera]
translate = [0.0, 1.0, 4.0]
rotate_x = -10.0
rotate_y = 0.0

[[mesh]]
name = "ball"
type = "sphere"
radius = 0.75

[[node]]
id = "mover"
mesh = "ball"
material = "MoverRed"

[[animation]]
target = "nodes/mover/translation"
interpolation = "linear"
times = [0.0, 1.0]
values = [-2.0, 0.75, 0.0, 2.0, 0.75, 0.0]
)toml";
        {
            std::ofstream lib(dir / "mover.materials.toml");
            lib << kLibrary;
        }
        {
            std::ofstream scn(dir / "s.scene.toml");
            scn << kScene;
        }
        const auto cfg =
            harmonia::scene::SceneLoader::load(dir / "s.scene.toml", dir, scene, deviceCtx(), commandPool());
        if (!cfg.has_value()) {
            return false;
        }
        mover = 0U;
        return scene.instanceCount() == 1U;
    }

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
        const auto mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.theia.giTlas.quad");
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

    /// One GI frame. The G-buffer stands in for the rasterizer: every pixel
    /// records a floor hit on its camera ray (or a point far along the ray for
    /// pixels that point above the floor), so the stage's own re-trace decides
    /// which surface is in front.
    void recordFrame(std::uint32_t index) {
        const float tangent = std::tan(43.0F * std::numbers::pi / 360.0F);
        const float aspect = float(kExtent.width) / float(kExtent.height);
        const sm::float3 forward = sm::normalize(sm::float3(0, 0.5F, 0) - camera);
        const sm::float3 right = sm::normalize(sm::cross(forward, sm::float3(0, 1, 0)));
        const sm::float3 up = sm::cross(right, forward);
        std::vector<sm::float4> gbuffer(std::size_t{kExtent.width} * kExtent.height);
        for (std::uint32_t y = 0; y < kExtent.height; ++y) {
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                const float u = (2.0F * (float(x) + 0.5F) / float(kExtent.width) - 1.0F) * tangent * aspect;
                const float v = (1.0F - 2.0F * (float(y) + 0.5F) / float(kExtent.height)) * tangent;
                const sm::float3 direction = sm::normalize(forward + right * u + up * v);
                const float distance = direction.y < 0.0F ? -camera.y / direction.y : 50.0F;
                const sm::float3 world = camera + direction * distance;
                gbuffer[std::size_t{y} * kExtent.width + x] = {
                    world.x, world.y, world.z, std::bit_cast<float>(floorInstance + 1U)};
            }
        }
        inputs.uploadData(gbuffer.data(), gbuffer.size() * sizeof(sm::float4));
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
        const auto n = view * sm::float4(0, 1, 0, 0);
        const VkClearColorValue encoded{{n.x * 0.5F + 0.5F, n.y * 0.5F + 0.5F, n.z * 0.5F + 0.5F, 1}};
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
        params.envSampler = sampler.get();
        params.view = view;
        params.cameraPos = camera;
        params.exposure = 1 / 1.2F;
        params.frameSampleIndex = index;
        params.rngBaseSeed = 7U;
        params.maxDepth = 2;
        // The estimator only: the interactive tiers are off exactly as in a capture.
        params.useA3Regularization = false;
        params.adaptiveMaxSamples = 1;
        params.useRestirDi = false;
        params.useRestirPt = true;
        params.useRestirPtPath = false;
        params.fireflyClampEnabled = false;
        params.materialLodEnabled = false;
        params.radianceCacheEnabled = false;
        gi.record(*cmd, params);
        first = false;
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        if (!gi.completionRequired()) {
            return;
        }
        auto ready = gi.frameReady();
        ASSERT_TRUE(ready.has_value());
        for (std::uint32_t safety = 0; !(*ready); ++safety) {
            ASSERT_LT(safety, 16384U) << "Test hang guard fails, never accepts a capped estimate";
            auto next = commandPool().beginOneShot();
            ASSERT_TRUE(next.has_value());
            ASSERT_EQ(gi.recordContinuation(*next), VK_SUCCESS);
            ASSERT_EQ(commandPool().endOneShot(*next), VK_SUCCESS);
            ready = gi.frameReady();
            ASSERT_TRUE(ready.has_value());
        }
    }

    void readPixels(std::vector<sm::float4>& pixels) {
        pixels.resize(std::size_t{kExtent.width} * kExtent.height);
        auto buffer = harmonia::core::Buffer::create(deviceCtx(),
                                                     pixels.size() * sizeof(sm::float4),
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                     VMA_MEMORY_USAGE_GPU_TO_CPU,
                                                     "test.theia.giTlas.readback");
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

    theia::GiPass gi;
    harmonia::core::Image hdr, giBuffer, normal;
    harmonia::core::Buffer inputs;
    harmonia::core::UniqueSampler sampler;
    bool first = true;
    std::uint32_t frameIndex = 0;
};

class GiTlasStaticRebuildFixture : public GiTlasRebuildFixture {};
class GiTlasPeriodicRebuildFixture : public GiTlasRebuildFixture {
  protected:
    [[nodiscard]] bool animated() const noexcept override { return true; }
};

} // namespace

// Static scene: every TLAS update is a full rebuild (fresh structure). The GI
// stage must trace the new one in the slot of the next frame and in the other
// slot of the frame after it.
TEST_F(GiTlasStaticRebuildFixture, GiPass_TracesTheTlasAfterARebuild) {
    const Poses built = render();
    EXPECT_GE(built.left, kPresent) << "the mover is built at x = -2";
    EXPECT_EQ(built.center, 0U);
    EXPECT_EQ(built.right, 0U);

    const VkAccelerationStructureKHR before = scene.tlas();
    scene.instanceTransform(mover).translation.x = 2.0F;
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
    ASSERT_NE(scene.tlas(), before) << "a static scene rebuilds the TLAS";

    for (std::uint32_t frame = 0; frame < 2U; ++frame) {
        const Poses moved = render();
        EXPECT_EQ(moved.left, 0U) << "frame " << frame << ": the slot must not trace the replaced TLAS";
        EXPECT_EQ(moved.center, 0U) << "frame " << frame;
        EXPECT_GE(moved.right, kPresent) << "frame " << frame << ": the mover is at x = +2";
    }

    scene.instanceTransform(mover).translation.x = 0.0F;
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
    const Poses returned = render();
    EXPECT_EQ(returned.left, 0U);
    EXPECT_GE(returned.center, kPresent) << "a second rebuild is traced too";
    EXPECT_EQ(returned.right, 0U);
}

// Animated scene: Scene::kTlasRebuildInterval updates refit the TLAS in place, the one after rebuilds it.
// A sequence capture reaches it in the animated frame after them.
TEST_F(GiTlasPeriodicRebuildFixture, GiPass_TracesTheTlasAfterThePeriodicRebuild) {
    const VkAccelerationStructureKHR before = scene.tlas();
    // The track stamps the start pose on the first update (x = -2).
    scene.setAnimationTime(0.0F);
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
    ASSERT_EQ(scene.tlas(), before) << "update 1 refits in place";
    const Poses built = render();
    EXPECT_GE(built.left, kPresent) << "the mover starts at x = -2";
    EXPECT_EQ(built.center, 0U);
    EXPECT_EQ(built.right, 0U);

    for (std::uint32_t update = 2; update <= theia::Scene::kTlasRebuildInterval; ++update) {
        scene.setAnimationTime(0.0F);
        ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
        ASSERT_EQ(scene.tlas(), before) << "update " << update << " refits in place";
    }
    // The update after the interval crosses it and also moves the mover: the replaced
    // structure holds the pose of the last refit, x = -2.
    scene.setAnimationTime(1.0F);
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
    ASSERT_NE(scene.tlas(), before) << "update " << theia::Scene::kTlasRebuildInterval + 1U << " rebuilds the TLAS";

    for (std::uint32_t frame = 0; frame < 2U; ++frame) {
        const Poses moved = render();
        EXPECT_EQ(moved.left, 0U) << "frame " << frame << ": the slot must not trace the replaced TLAS";
        EXPECT_EQ(moved.center, 0U) << "frame " << frame;
        EXPECT_GE(moved.right, kPresent) << "frame " << frame << ": the mover is at x = +2";
    }

    scene.setAnimationTime(0.5F);
    ASSERT_EQ(scene.updateTlas(deviceCtx(), commandPool()), VK_SUCCESS);
    const Poses middle = render();
    EXPECT_EQ(middle.left, 0U);
    EXPECT_GE(middle.center, kPresent) << "refits of the rebuilt TLAS are traced too";
    EXPECT_EQ(middle.right, 0U);
}
