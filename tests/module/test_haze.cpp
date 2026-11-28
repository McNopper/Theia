// Module tests: the distance haze of a scene, from its `[haze]` section to the radiance
// the GI stage composites.
//
//  - HazeMath_*: Harmonia/shaders/haze.slang, the module both renderers use, against a
//    numerical integral of the medium (a CPU double-precision midpoint integral).
//  - TheiaHazeConfig_*: the two keys of the section reach the frame parameters of the GI
//    stage, which record() pushes to the shader, and nothing does when the scene has no
//    section. The loaded scenes go through the shared SceneLoader without a device.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <slang-math/slang-math.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "ShaderProbeFixture.hpp"
#include "harmonia/scene/SceneLoader.hpp"
#include "theia/gi/GiPass.hpp"
#include "theia/scene/Scene.hpp"

namespace {

constexpr std::size_t kInputsPerCase = 3U;
constexpr std::size_t kValuesPerCase = 2U;

/// A camera ray of the haze probe scene (Aether's `haze`): the camera 2 m up, the red
/// cube 20 m away on the ground.
constexpr sm::float3 kProbeCamera{0.0F, 2.0F, 6.0F};
constexpr sm::float3 kProbeCube{0.0F, 1.0F, -14.0F};
constexpr float kProbeDensity = 0.02F;
constexpr float kProbeHeight = 40.0F;

/// A segment and the medium it lies in.
struct Segment {
    sm::float3 a{};
    sm::float3 b{};
    float density = kProbeDensity;
    float height = kProbeHeight;
    bool active = true; ///< the medium of the segment is a haze (else its optical depth is 0)
};

[[nodiscard]] Segment segment(sm::float3 a, sm::float3 b) {
    return Segment{.a = a, .b = b};
}

/// A segment of the probe scene: from the camera to @p b.
[[nodiscard]] Segment probe(sm::float3 b) {
    return segment(kProbeCamera, b);
}

/// A horizontal segment at height zero.
[[nodiscard]] Segment ground(float length) {
    return segment(sm::float3{0.0F, 0.0F, 0.0F}, sm::float3{length, 0.0F, 0.0F});
}

/// A vertical segment from @p from to @p to metres.
[[nodiscard]] Segment vertical(float from, float to) {
    return segment(sm::float3{0.0F, from, 0.0F}, sm::float3{0.0F, to, 0.0F});
}

/// The medium is no haze: no density, or no scale height.
[[nodiscard]] Segment inactive(Segment s, float density, float height) {
    s.active = false;
    s.density = density;
    s.height = height;
    return s;
}

/// Every family of segment the closed form of the module has a branch for.
const std::array<Segment, 14U> kSegments{{
    ground(10.0F), // horizontal at height zero
    segment(sm::float3{0.0F, 5.0e-4F, 0.0F},
            sm::float3{10.0F, 0.0F, 0.0F}), // x = -1.25e-5: a slight descent, inside the series branch
    segment(sm::float3{0.0F, 8.0e-1F, 0.0F}, sm::float3{10.0F, 0.0F, 0.0F}), // x = -0.02: direct branch
    segment(sm::float3{0.0F, 2.0F, 0.0F}, sm::float3{3.0F, 2.0004F, 4.0F}),  // x = +1e-5: series branch
    segment(sm::float3{0.0F, 0.0F, 0.0F}, sm::float3{3.0F, 0.8F, 4.0F}),     // x = +0.02: direct branch
    probe(kProbeCube),                                                       // the probe's ray to the 20 m cube
    vertical(0.0F, 200.0F),                                                  // a steep ascent
    vertical(300.0F, 0.0F),                                                  // a steep descent
    vertical(1000.0F, 0.0F),                                                 // a descent through most of the medium
    vertical(-30.0F, 10.0F),                                                 // from below height zero
    segment(sm::float3{1.0F, 2.0F, 3.0F}, sm::float3{1.0F, 2.0F, 3.0F}),     // no length
    inactive(ground(10.0F), 0.0F, kProbeHeight),                             // no density
    inactive(ground(10.0F), kProbeDensity, 0.0F),                            // no scale height
    Segment{.a = sm::float3{0.0F, 0.0F, 0.0F},
            .b = sm::float3{100.0F, 0.0F, 0.0F},
            .density = kProbeDensity,
            .height = 0.5F}, // a scale height far below the length
}};

std::vector<sm::float4> segmentInputs() {
    std::vector<sm::float4> inputs;
    inputs.reserve(kSegments.size() * kInputsPerCase);
    for (const Segment& segment : kSegments) {
        inputs.emplace_back(segment.density, segment.height, 0.0F, 0.0F);
        inputs.emplace_back(segment.a.x, segment.a.y, segment.a.z, 0.0F);
        inputs.emplace_back(segment.b.x, segment.b.y, segment.b.z, 0.0F);
    }
    return inputs;
}

/// Independent midpoint integral in double precision. A fp32 GPU sum accumulates several
/// 1e-5 of error over 16K steps even with compensation under fast-math; the oracle must not
/// confuse its own rounding with the closed form's error.
double opticalDepthIntegral(const Segment& s) {
    if (!s.active || s.a == s.b) {
        return 0.0;
    }
    const double dx = double(s.b.x) - s.a.x;
    const double dy = double(s.b.y) - s.a.y;
    const double dz = double(s.b.z) - s.a.z;
    const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
    constexpr int kSteps = 16384;
    const double ds = length / kSteps;
    double sum = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        const double t = (i + 0.5) / kSteps;
        const double height = s.a.y + dy * t;
        sum += s.density * std::exp(-height / s.height) * ds;
    }
    return sum;
}

/// A scene of the loader tests: a camera placement and, with @p section, a `[haze]`.
std::filesystem::path writeScene(const char* name, std::string_view section) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "theia_haze_config_test";
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / name;
    std::ofstream out(path);
    out << "[camera]\ntranslate = [0.0, 2.0, 6.0]\nrotate_x = -8.0\nrotate_y = 0.0\n\n" << section;
    return path;
}

/// The renderer configuration of a scene, as the shared loader hands it out. The scene
/// has no mesh, so the load never reaches the (absent) device.
[[nodiscard]] std::expected<harmonia::scene::SceneLoader::SceneConfig, VkResult>
loadScene(const std::filesystem::path& path) {
    theia::Scene scene;
    const harmonia::DeviceContext dummyCtx{};
    const harmonia::core::CommandPool dummyPool{};
    return harmonia::scene::SceneLoader::load(path, path.parent_path(), scene, dummyCtx, dummyPool);
}

} // namespace

// The optical depth of the module against the integral of the medium, and the composite
// and the transmittance against the integral's transmittance.
TEST_F(ShaderProbeFixture, HazeMath_OpticalDepthAndCompositeMatchTheIntegralOfTheMedium) {
    const std::vector<sm::float4> inputs = segmentInputs();
    std::vector<sm::float4> outputs;
    ASSERT_NO_FATAL_FAILURE(probe("haze_test.spv", inputs, kSegments.size() * kValuesPerCase, outputs));
    ASSERT_EQ(outputs.size(), kSegments.size() * kValuesPerCase);

    for (std::size_t k = 0U; k < kSegments.size(); ++k) {
        SCOPED_TRACE("segment " + std::to_string(k));
        const sm::float4& depth = outputs[k * kValuesPerCase];
        const sm::float4& composite = outputs[k * kValuesPerCase + 1U];
        const double expectedDepth = opticalDepthIntegral(kSegments[k]);
        if (expectedDepth == 0.0) {
            EXPECT_FLOAT_EQ(depth.x, 0.0F) << "the optical depth of a segment without haze";
            EXPECT_FLOAT_EQ(composite.w, 1.0F) << "the transmittance of such a segment";
            continue;
        }
        const float expectedTransmittance = static_cast<float>(std::exp(-expectedDepth));
        EXPECT_NEAR(depth.x, expectedDepth, 1.0e-5 * expectedDepth) << "optical depth against the CPU double integral";
        EXPECT_NEAR(depth.z, expectedTransmittance, 1.0e-5F * expectedTransmittance)
            << "transmittance against the integral's";
        EXPECT_NEAR(composite.w, expectedTransmittance, 1.0e-5F * expectedTransmittance)
            << "hazeTransmittance against the integral's";
        const sm::float3 radiance{0.5F, 2.0F, 7.0F};
        const sm::float3 sky{0.25F, 0.5F, 1.0F};
        const float transmittance = expectedTransmittance;
        const sm::float4 expected{transmittance * radiance.x + (1.0F - transmittance) * sky.x,
                                  transmittance * radiance.y + (1.0F - transmittance) * sky.y,
                                  transmittance * radiance.z + (1.0F - transmittance) * sky.z,
                                  0.0F};
        EXPECT_NEAR(composite.x, expected.x, 1.0e-5F * expected.x + 1.0e-6F) << "hazeComposite . red";
        EXPECT_NEAR(composite.y, expected.y, 1.0e-5F * expected.y + 1.0e-6F) << "hazeComposite . green";
        EXPECT_NEAR(composite.z, expected.z, 1.0e-5F * expected.z + 1.0e-6F) << "hazeComposite . blue";
    }
    // Two closed forms of the medium, by hand: a horizontal segment at height zero is
    // the extinction times its length, and a scale height far below a horizontal segment
    // makes it depth * length * exp(0).
    EXPECT_NEAR(outputs[0].x, 0.2F, 1.0e-6F) << "10 m at 0.02 per metre";
    EXPECT_NEAR(outputs[13 * kValuesPerCase].x, 2.0F, 1.0e-5F) << "100 m at 0.02 per metre";
}

// The blur of the in-scattered sky is an angle, so the mip level of the environment map
// follows the map's width. It used to be the fixed level 6, which is six degrees only
// on a map 3840 texels wide: 22.5 degrees on a 1K map, 3 degrees on an 8K one.
TEST_F(ShaderProbeFixture, HazeMath_SkyBlurIsTheSameAngleOnEveryEnvironmentMap) {
    // 1K, the width the fixed level was tuned on, 4K and 8K.
    constexpr std::array<float, 4U> kWidths{1024.0F, 3840.0F, 4096.0F, 8192.0F};
    const std::size_t first = kSegments.size() * kValuesPerCase;
    const std::vector<sm::float4> inputs = segmentInputs();
    std::vector<sm::float4> outputs;
    ASSERT_NO_FATAL_FAILURE(
        probe("haze_test.spv", inputs, (kSegments.size() + kWidths.size()) * kValuesPerCase, outputs));
    ASSERT_EQ(outputs.size(), (kSegments.size() + kWidths.size()) * kValuesPerCase);

    for (std::size_t j = 0U; j < kWidths.size(); ++j) {
        SCOPED_TRACE("environment map " + std::to_string(static_cast<int>(kWidths[j])) + " wide");
        const sm::float4& lod = outputs[first + j * kValuesPerCase];
        EXPECT_FLOAT_EQ(lod.x, kWidths[j]);
        EXPECT_FLOAT_EQ(lod.w, 6.0F) << "the blur the module is written for, in degrees";
        EXPECT_NEAR(lod.z, lod.w, 1.0e-3F) << "degrees one texel spans at the level the module picks";
    }
    const auto level = [&outputs](std::size_t j) { return outputs[first + j * kValuesPerCase].y; };
    EXPECT_NEAR(level(1U), 6.0F, 1.0e-4F) << "the map the fixed level was tuned on keeps it";
    EXPECT_LT(level(0U), 4.2F) << "a 1K map needs a lower level than the fixed 6";
    EXPECT_GT(level(3U), 7.0F) << "an 8K map needs a higher one";
}

TEST(TheiaHazeConfig, NoSectionIsNoHaze) {
    const theia::GiPass::FrameParams defaults{};
    EXPECT_FLOAT_EQ(defaults.haze.density, 0.0F);
    EXPECT_FLOAT_EQ(defaults.haze.height, 0.0F);

    const theia::SceneHaze haze = theia::sceneHaze(harmonia::scene::SceneLoader::SceneConfig{});
    EXPECT_FLOAT_EQ(haze.density, 0.0F);
    EXPECT_FLOAT_EQ(haze.height, 0.0F);
}

TEST(TheiaHazeConfig, BothKeysReachTheFrameParameters) {
    harmonia::scene::SceneLoader::SceneConfig config{};
    config.hazeDensity = 0.02F;
    config.hazeHeight = 40.0F;
    const theia::SceneHaze haze = theia::sceneHaze(config);
    EXPECT_FLOAT_EQ(haze.density, 0.02F);
    EXPECT_FLOAT_EQ(haze.height, 40.0F);

    theia::GiPass::FrameParams params{};
    params.haze = haze;
    EXPECT_FLOAT_EQ(params.haze.density, 0.02F);
    EXPECT_FLOAT_EQ(params.haze.height, 40.0F);
}

TEST(TheiaHazeConfig, ASectionWithoutItsKeysIsNoHaze) {
    // The format rejects such a section at load; should one arrive, the missing key is
    // 0, which the shader treats like the default.
    harmonia::scene::SceneLoader::SceneConfig densityOnly{};
    densityOnly.hazeDensity = 0.02F;
    const theia::SceneHaze haze = theia::sceneHaze(densityOnly);
    EXPECT_FLOAT_EQ(haze.density, 0.02F);
    EXPECT_FLOAT_EQ(haze.height, 0.0F) << "no scale height: hazeActive is false";
}

// The section of a scene file, through the shared loader, to the frame parameters the
// GI stage records with every frame.
TEST(TheiaHazeConfig, LoadedSectionReachesTheFrameParameters) {
    const auto path = writeScene("hazed.scene.toml", "[haze]\ndensity = 0.02\nheight = 40.0\n");
    const auto config = loadScene(path);
    ASSERT_TRUE(config.has_value());
    theia::GiPass::FrameParams params{};
    params.haze = theia::sceneHaze(*config);
    EXPECT_FLOAT_EQ(params.haze.density, 0.02F);
    EXPECT_FLOAT_EQ(params.haze.height, 40.0F);
    std::filesystem::remove_all(path.parent_path());
}

TEST(TheiaHazeConfig, LoadedSceneWithoutSectionKeepsNoHaze) {
    const auto path = writeScene("clear.scene.toml", "");
    const auto config = loadScene(path);
    ASSERT_TRUE(config.has_value());
    theia::GiPass::FrameParams params{};
    params.haze = theia::sceneHaze(*config);
    EXPECT_FLOAT_EQ(params.haze.density, 0.0F);
    EXPECT_FLOAT_EQ(params.haze.height, 0.0F);
    std::filesystem::remove_all(path.parent_path());
}

// The format validates the two keys together; a scene that breaks the rule loads nowhere.
TEST(TheiaHazeConfig, InvalidSectionsDoNotLoad) {
    constexpr std::array<std::string_view, 4U> kRejected{{
        "[haze]\ndensity = 0.02\n",
        "[haze]\nheight = 40.0\n",
        "[haze]\ndensity = -0.02\nheight = 40.0\n",
        "[haze]\ndensity = 0.02\nheight = 0.0\n",
    }};
    for (std::size_t k = 0U; k < kRejected.size(); ++k) {
        SCOPED_TRACE(kRejected[k]);
        const auto path = writeScene("invalid.scene.toml", kRejected[k]);
        EXPECT_FALSE(loadScene(path).has_value()) << "the loader must refuse this section";
        std::filesystem::remove_all(path.parent_path());
    }
}
