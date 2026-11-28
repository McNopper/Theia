// The distance haze of a scene in the GI stage (shaders/gi.comp.slang, hazePixel and
// the sites that call it): every pixel that shows a surface is seen through the
// medium, whatever path the kernel took to the pixel.
//
// The oracle does not use the shader's closed form. The scene is rendered twice per
// frame with the same seed, once without haze and once with; the haze never feeds back
// into the estimator, so the terms of the two renders are the same, and a hazed pixel
// must be
//
//     hazed = T * clear                      T = exp(-tau),  tau = integral sigma(h) ds
//
// with tau taken here by a numerical integral of the medium. There is no environment
// map, so the light the haze scatters into the pixel is zero, and the sky keeps its
// radiance: a background pixel is left exactly as the forward pass wrote it.
//
// The G-buffer stands in for the rasterizer, so one frame holds every family of pixel:
//   floor        the camera ray hits the floor: main composites the pixel (direct light
//                of the reservoir; the indirect light through the path reservoir in
//                combineMain, or through main when the path reservoir is off)
//   emitter      the camera ray hits the emissive panel: the forward pass owns its EDF,
//                while GI adds the coexisting BSDF's environment/indirect transport
//   visibility   the rasterizer recorded a surface, the re-traced camera ray finds none:
//                without manifold sampling the view-space normal is the fallback
//                surface; with manifold sampling the kernel returns before its terms,
//                and the pixel keeps the forward pass's radiance, through the haze
//                (it did not: the pixel stayed clear while its neighbours were hazed)
//   background   the rasterizer recorded nothing: the sky, left as drawn

#include <volk/volk.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <slang-math/slang-math.hpp>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "harmonia/scene/Material.hpp"
#include "theia/gi/GiPass.hpp"
#include "theia/scene/Scene.hpp"

namespace {

constexpr VkExtent2D kExtent{128, 96};
// The block of background pixels, on the floor and away from the panel.
constexpr std::uint32_t kBlockX1 = 24U;
constexpr std::uint32_t kBlockY0 = 64U;
// What the HDR image holds before the GI stage runs: a stand-in for what the forward
// pass wrote, different from anything the stage would add by accident.
constexpr sm::float4 kFill{0.125F, 0.25F, 0.5F, 1.0F};
// The emissive panel the camera looks at, 4 m in front of it.
const std::array<sm::float3, 4> kPanel{{
    sm::float3{-1.0F, 1.6F, 3.0F},
    sm::float3{1.0F, 1.6F, 3.0F},
    sm::float3{1.0F, 2.75F, 3.0F},
    sm::float3{-1.0F, 2.75F, 3.0F},
}};
// A pixel is the panel's only when its ray meets the panel well inside its outline:
// around the outline the rasterizer and the re-traced ray may disagree.
constexpr float kPanelMargin = 5.0e-3F;
// Extinction of 0.06 per metre at height zero, scale height 40 m: the floor, 8 to 12 m
// away, keeps about half its radiance, a point 50 m away about five percent.
constexpr theia::SceneHaze kHaze{.density = 0.06F, .height = 40.0F};
// The terms of a pixel are this far apart in the two renders only through the haze:
// the transmittance on the GPU is a float product away from the integral's.
constexpr double kTransmittance = 2.0e-5;

enum class Pixel { Floor, Emitter, Edge, Visibility, Background };

struct Ray {
    sm::float3 direction{};
    Pixel pixel{};
    std::uint32_t instance{}; ///< the instance the rasterizer records (UINT32_MAX: none)
    sm::float3 world{};       ///< and where
};

class GiHazeFixture : public RtFixture {
  protected:
    void SetUp() override {
        RtFixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        const std::uint32_t floorMaterial =
            scene.addMaterial(harmonia::scene::Material::diffuse(sm::float3(0.05F, 0.15F, 0.8F), 1.0F));
        const std::uint32_t light = scene.addMaterial(harmonia::scene::Material::emissive(sm::float3(1), 10));
        const std::uint32_t panel = scene.addMaterial(harmonia::scene::Material::emissive(sm::float3(1), 4));

        // Wound to face up, like the floor of the GI frame fixtures.
        ASSERT_NO_FATAL_FAILURE(quad({sm::float3(-20, 0, -20), {20, 0, -20}, {20, 0, 20}, {-20, 0, 20}},
                                     {0, 1, 0},
                                     floorMaterial,
                                     true,
                                     floorInstance));
        // An emissive quad above and behind the view, wound to face down: it lights the
        // floor.
        std::uint32_t lightInstance = std::numeric_limits<std::uint32_t>::max();
        ASSERT_NO_FATAL_FAILURE(quad({sm::float3(-0.5F, 5, 2.5F), {0.5F, 5, 2.5F}, {0.5F, 5, 3.5F}, {-0.5F, 5, 3.5F}},
                                     {0, -1, 0},
                                     light,
                                     false,
                                     lightInstance));
        // The panel, wound to face the camera.
        ASSERT_NO_FATAL_FAILURE(quad(kPanel, {0, 0, 1}, panel, false, panelInstance));
        ASSERT_EQ(scene.instanceCount(), 3U);
        ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
        // Manifold sampling runs on a scene with emitters.
        ASSERT_EQ(scene.emissiveTriangleCount(), 4U);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        VkSampler rawSampler{};
        ASSERT_EQ(vkCreateSampler(deviceCtx().device, &samplerInfo, nullptr, &rawSampler), VK_SUCCESS);
        sampler = harmonia::core::UniqueSampler{deviceCtx().device, rawSampler};
        ASSERT_NO_FATAL_FAILURE(image(hdr, "test.theia.giHaze.hdr"));
        ASSERT_NO_FATAL_FAILURE(image(giBuffer, "test.theia.giHaze.gbuffer"));
        ASSERT_NO_FATAL_FAILURE(image(normal, "test.theia.giHaze.normal"));
        auto buffer = harmonia::core::Buffer::create(deviceCtx(),
                                                     std::size_t{kExtent.width} * kExtent.height * sizeof(sm::float4),
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                     VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                                     "test.theia.giHaze.inputs");
        ASSERT_TRUE(buffer.has_value());
        inputs = std::move(*buffer);
        camera = sm::float3(0, 2.5F, 7);
        view = sm::lookAt(camera, sm::float3(0, 0.5F, 0), sm::float3(0, 1, 0));
    }

    void TearDown() override {
        for (auto& gi : passes) {
            gi->shutdown();
        }
        passes.clear();
        RtFixture::TearDown();
    }

    /// A GI stage with or without manifold sampling. The stages of a test own their
    /// reservoirs: the haze never enters them, so a clear and a hazed render of the same
    /// frame from two stages hold the same terms.
    theia::GiPass& makePass(bool sms) {
        auto gi = std::make_unique<theia::GiPass>();
        EXPECT_TRUE(gi->initialize(deviceCtx(),
                                   theia::GiPass::Config{.width = kExtent.width,
                                                         .height = kExtent.height,
                                                         .hdrImage = hdr.handle(),
                                                         .hdrView = hdr.view(),
                                                         .giBufferImage = giBuffer.handle(),
                                                         .giBufferView = giBuffer.view(),
                                                         .gbufferImage = normal.handle(),
                                                         .gbufferView = normal.view(),
                                                         .smsEnabled = sms},
                                   commandPool()));
        passes.push_back(std::move(gi));
        return *passes.back();
    }

    /// What the rasterizer records for a pixel, and which family the pixel is.
    Ray rayOf(std::uint32_t x, std::uint32_t y) const {
        const float tangent = std::tan(43.0F * std::numbers::pi / 360.0F);
        const float aspect = float(kExtent.width) / float(kExtent.height);
        const sm::float3 forward = sm::normalize(sm::float3(0, 0.5F, 0) - camera);
        const sm::float3 right = sm::normalize(sm::cross(forward, sm::float3(0, 1, 0)));
        const sm::float3 up = sm::cross(right, forward);
        const float u = (2.0F * (float(x) + 0.5F) / float(kExtent.width) - 1.0F) * tangent * aspect;
        const float v = (1.0F - 2.0F * (float(y) + 0.5F) / float(kExtent.height)) * tangent;
        Ray ray;
        ray.direction = sm::normalize(forward + right * u + up * v);
        ray.instance = floorInstance;
        if (x < kBlockX1 && y >= kBlockY0) {
            ray.pixel = Pixel::Background;
            ray.instance = std::numeric_limits<std::uint32_t>::max();
            return ray;
        }
        if (ray.direction.z < 0.0F) {
            const double distance = (double(kPanel[0].z) - camera.z) / ray.direction.z;
            const sm::float3 p(float(camera.x + ray.direction.x * distance),
                               float(camera.y + ray.direction.y * distance),
                               float(camera.z + ray.direction.z * distance));
            const bool inside = p.x > kPanel[0].x + kPanelMargin && p.x < kPanel[1].x - kPanelMargin &&
                                p.y > kPanel[0].y + kPanelMargin && p.y < kPanel[2].y - kPanelMargin;
            const bool around = p.x > kPanel[0].x - kPanelMargin && p.x < kPanel[1].x + kPanelMargin &&
                                p.y > kPanel[0].y - kPanelMargin && p.y < kPanel[2].y + kPanelMargin;
            if (inside) {
                ray.pixel = Pixel::Emitter;
                ray.instance = panelInstance;
                ray.world = p;
                return ray;
            }
            if (around) {
                ray.pixel = Pixel::Edge;
            }
        }
        // The floor along the ray, or a point far along a ray that points at the sky: a
        // surface the rasterizer recorded and the re-traced camera ray does not find.
        const double distance = ray.direction.y < 0.0F ? -double(camera.y) / ray.direction.y : 50.0;
        ray.world = sm::float3(float(camera.x + ray.direction.x * distance),
                               float(camera.y + ray.direction.y * distance),
                               float(camera.z + ray.direction.z * distance));
        if (ray.pixel != Pixel::Edge) {
            ray.pixel = ray.direction.y < 0.0F ? Pixel::Floor : Pixel::Visibility;
        }
        return ray;
    }

    /// One GI frame: every pixel records what the rasterizer would, the HDR image is
    /// filled with kFill, and the stage runs. Returns the HDR image after the stage.
    std::vector<sm::float4> render(theia::GiPass& gi, std::uint32_t index, theia::SceneHaze haze, bool pathReservoir) {
        std::vector<sm::float4> gbuffer(std::size_t{kExtent.width} * kExtent.height);
        for (std::uint32_t y = 0; y < kExtent.height; ++y) {
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                const Ray ray = rayOf(x, y);
                gbuffer[std::size_t{y} * kExtent.width + x] =
                    ray.instance == std::numeric_limits<std::uint32_t>::max()
                        ? sm::float4(0.0F, 0.0F, 0.0F, 0.0F)
                        : sm::float4(ray.world.x, ray.world.y, ray.world.z, std::bit_cast<float>(ray.instance + 1U));
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
        // ATTACHMENT_OPTIMAL, where the rasterizer leaves them; the GI stage moves
        // them, and the read-back leaves the HDR image in GENERAL.
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
        // The view-space normal of the fallback surface: up.
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
        params.haze = haze;
        params.frameSampleIndex = index;
        params.rngBaseSeed = 7U;
        params.maxDepth = 3;
        // The estimator only, the interactive tiers off exactly as in a capture.
        params.useA3Regularization = false;
        params.adaptiveMaxSamples = 1;
        params.useRestirDi = false;
        params.useRestirPt = true;
        params.useRestirPtPath = pathReservoir;
        params.fireflyClampEnabled = false;
        params.materialLodEnabled = false;
        params.radianceCacheEnabled = false;
        gi.record(*cmd, params);
        first = false;
        EXPECT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        // The frame is complete, or needs continuations (manifold sampling) until it is.
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
                                                     "test.theia.giHaze.readback");
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

    theia::Scene scene;
    std::uint32_t floorInstance = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t panelInstance = std::numeric_limits<std::uint32_t>::max();
    sm::float3 camera{};

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
        const auto mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.theia.giHaze.quad");
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

    sm::float4x4 view{1.0F};
    harmonia::core::Image hdr, giBuffer, normal;
    harmonia::core::Buffer inputs;
    harmonia::core::UniqueSampler sampler;
    std::vector<std::unique_ptr<theia::GiPass>> passes;
    bool first = true;
};

/// The optical depth from the camera to @p b: the integral of the medium over the
/// segment, by the midpoint rule, in double. Written from the definition of the medium
/// (sigma(h) = density * exp(-h / height)), not from the shader's closed form. With 1024
/// steps over at most 50 m and a scale height of 40 m the rule is within 1e-7 of the
/// integral, relative.
double opticalDepth(sm::float3 b) {
    const sm::float3 a{0, 2.5F, 7};
    const double length =
        std::sqrt(double(b.x - a.x) * (b.x - a.x) + double(b.y - a.y) * (b.y - a.y) + double(b.z - a.z) * (b.z - a.z));
    constexpr int steps = 1024;
    const double ds = length / steps;
    double sum = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double s = (double(i) + 0.5) * ds;
        const double h = a.y + (double(b.y) - a.y) * (s / length);
        sum += double(kHaze.density) * std::exp(-h / double(kHaze.height)) * ds;
    }
    return sum;
}

[[nodiscard]] bool sameBits(sm::float4 a, sm::float4 b) {
    return std::bit_cast<std::array<std::uint32_t, 4>>(a) == std::bit_cast<std::array<std::uint32_t, 4>>(b);
}

/// What the checks of one frame found.
struct Tally {
    std::size_t failures = 0U; ///< pixels not matching the oracle
    std::size_t kept = 0U;     ///< pixels that had to keep the fill exactly, and did not
    double worst = 0.0;        ///< the largest relative difference from the oracle
};

void report(const Tally& tally, const char* what) {
    EXPECT_EQ(tally.kept, 0U) << what << ": pixels that should have kept the forward pass's radiance exactly";
    EXPECT_EQ(tally.failures, 0U) << what << ": pixels off the oracle (hazed = T * clear), worst relative difference "
                                  << tally.worst;
}

/// `hazed` against T * `clear`, component by component.
void expectThroughHaze(Tally& tally, sm::float4 clear, sm::float4 hazed, double transmittance) {
    for (const auto& [c, h] : {std::pair{clear.x, hazed.x}, {clear.y, hazed.y}, {clear.z, hazed.z}}) {
        const double expected = transmittance * c;
        const double difference = std::abs(double(h) - expected);
        tally.worst = std::max(tally.worst, difference / std::max(std::abs(expected), 1.0e-12));
        if (difference > kTransmittance * std::abs(expected) + 1.0e-9) {
            ++tally.failures;
        }
    }
}

} // namespace

// Every family of pixel, with the path reservoir on (the indirect light arrives in
// combineMain) and off (it arrives through main), without and with manifold sampling.
TEST_F(GiHazeFixture, GiPass_EveryPixelIsSeenThroughTheHaze) {
    struct Variation {
        const char* name;
        bool pathReservoir;
        bool sms;
    };
    constexpr std::array<Variation, 3U> kVariations{{
        {"path reservoir", true, false},
        {"per-sample walk", false, false},
        {"manifold sampling", true, true},
    }};
    for (const Variation& variation : kVariations) {
        SCOPED_TRACE(variation.name);
        theia::GiPass& clearPass = makePass(variation.sms);
        theia::GiPass& hazedPass = makePass(variation.sms);
        for (std::uint32_t frame = 0; frame < 2U; ++frame) {
            SCOPED_TRACE("frame " + std::to_string(frame));
            const std::vector<sm::float4> clear = render(clearPass, frame, theia::SceneHaze{}, variation.pathReservoir);
            const std::vector<sm::float4> hazed = render(hazedPass, frame, kHaze, variation.pathReservoir);
            ASSERT_EQ(clear.size(), std::size_t{kExtent.width} * kExtent.height);
            ASSERT_EQ(hazed.size(), clear.size());

            Tally floor, emitter, visibility, other;
            std::size_t background = 0U, emitters = 0U, emitterTerms = 0U, visibleMisses = 0U, litFloor = 0U,
                        floorPixels = 0U;
            for (std::uint32_t y = 0; y < kExtent.height; ++y) {
                for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                    const std::size_t i = std::size_t{y} * kExtent.width + x;
                    const Ray ray = rayOf(x, y);
                    switch (ray.pixel) {
                    case Pixel::Background: {
                        ++background;
                        // The sky is the radiance at infinity and stays as drawn.
                        if (!sameBits(clear[i], kFill)) {
                            ++other.kept;
                        }
                        if (!sameBits(hazed[i], kFill)) {
                            ++other.failures;
                        }
                        break;
                    }
                    case Pixel::Emitter: {
                        ++emitters;
                        // The EDF stays in the forward pass; the coexisting BSDF now contributes
                        // here. Both the fill and those added terms obey the same haze oracle.
                        const float added = std::max({std::abs(clear[i].x - kFill.x),
                                                      std::abs(clear[i].y - kFill.y),
                                                      std::abs(clear[i].z - kFill.z)});
                        if (added > 1.0e-6F) {
                            ++emitterTerms;
                        }
                        expectThroughHaze(emitter, clear[i], hazed[i], std::exp(-opticalDepth(ray.world)));
                        break;
                    }
                    case Pixel::Visibility: {
                        ++visibleMisses;
                        // A visibility miss returns before its terms only under manifold sampling;
                        // without it the view-space normal is the fallback surface and the pixel
                        // takes the floor's road. The exact fill is retained only on the early return.
                        Tally& tally = variation.sms ? visibility : floor;
                        if (variation.sms && !sameBits(clear[i], kFill)) {
                            ++tally.kept;
                        }
                        expectThroughHaze(tally, clear[i], hazed[i], std::exp(-opticalDepth(ray.world)));
                        break;
                    }
                    case Pixel::Edge:
                        break;
                    case Pixel::Floor: {
                        ++floorPixels;
                        expectThroughHaze(floor, clear[i], hazed[i], std::exp(-opticalDepth(ray.world)));
                        const float added =
                            std::max({clear[i].x - kFill.x, clear[i].y - kFill.y, clear[i].z - kFill.z});
                        if (added > 0.01F) {
                            ++litFloor;
                        }
                        break;
                    }
                    }
                }
            }
            report(floor, "floor");
            report(emitter, "emitter");
            report(visibility, "visibility miss");
            report(other, "background");

            // The pixels that make the comparisons mean something were all there, and
            // the floor received light the haze has to dim: were the floor's terms to
            // skip it, a quarter of the pixels would be off the oracle.
            EXPECT_GT(background, 0U);
            EXPECT_GT(emitters, 100U) << "pixels on the emissive panel";
            EXPECT_GT(emitterTerms, 0U) << "the emissive panel's coexisting BSDF must contribute to GI";
            EXPECT_GT(visibleMisses, 100U) << "pixels whose re-traced camera ray misses";
            EXPECT_GT(floorPixels, 1000U);
            EXPECT_GE(litFloor * 4U, floorPixels) << "floor pixels with terms above one percent of the fill";
        }
    }
}
