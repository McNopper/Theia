// Module tests: the frame of the anisotropy in the forward fragment shader. It is the
// tangent of the texture coordinates, taken from the screen-space derivatives of the
// fragment (shaders/forward_tangent.slang) and turned into a frame by the shared
// buildAnisotropyFrame, exactly as the GI pass and Hyperion do for the same surface.
//
// The shader built it before from the vertex tangent, which OBJ and box meshes do not
// carry, and then from the normal alone. That frame fails this test: its direction is
// not the texture tangent and its anisotropic response differs from the shared one
// (the test shader also evaluates that old frame, as the control of this test).

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <slang-math/slang-math.hpp>
#include <string>
#include <vector>

#include "ShaderProbeFixture.hpp"

namespace {

constexpr std::size_t kValuesPerCase = 5U;
constexpr std::size_t kCases = 10U;
constexpr std::size_t kDegenerateCase = kCases - 1U;
constexpr std::size_t kOutputs = kCases * kValuesPerCase;

/// A direction is this far from exact, and a response difference (relative to the
/// response) this far from equal.
constexpr float kDirection = 1.0e-4F;
constexpr float kResponse = 1.0e-4F;

[[nodiscard]] sm::float3 xyz(const sm::float4& v) {
    return sm::float3{v.x, v.y, v.z};
}

[[nodiscard]] float dot3(sm::float3 a, sm::float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

} // namespace

TEST_F(ShaderProbeFixture, ForwardTangent_MatchesTheSharedFrameOfTheSameSurface) {
    std::vector<sm::float4> outputs;
    ASSERT_NO_FATAL_FAILURE(probe("forward_tangent_test.spv", {}, kOutputs, outputs));
    ASSERT_EQ(outputs.size(), kOutputs);

    std::size_t oldFrameDiffers = 0U;
    for (std::size_t c = 0U; c < kCases; ++c) {
        SCOPED_TRACE("case " + std::to_string(c));
        const std::size_t base = c * kValuesPerCase;
        const sm::float4& response = outputs[base + 3U]; // x: new against shared, y: old against shared, z:
                                                         // isotropic, w: the shared response
        const sm::float4& frame = outputs[base + 4U];    // x: B . B', y: T . N, z: |T|, w: |B|
        const sm::float3 theia = xyz(outputs[base + 0U]);
        const sm::float3 shared = xyz(outputs[base + 1U]);

        // The precondition of the comparisons: the surface responds, and Theia's frame
        // is a frame (unit vectors, the tangent in the plane of the normal).
        ASSERT_GT(response.w, 1.0e-4F) << "the shared frame gives a specular response";
        EXPECT_NEAR(frame.z, 1.0F, kDirection) << "|T|";
        EXPECT_NEAR(frame.w, 1.0F, kDirection) << "|B|";
        EXPECT_NEAR(frame.y, 0.0F, kDirection) << "T . N";

        // The direction of increasing u is the shared one. The sign of the tangent is
        // free (the anisotropy is even in it), so is the bitangent's.
        EXPECT_NEAR(std::abs(dot3(theia, shared)), 1.0F, kDirection) << "Theia's tangent against the shared tangent";
        EXPECT_NEAR(std::abs(frame.x), 1.0F, kDirection) << "Theia's bitangent against the shared bitangent";

        // The anisotropic response of the analytic-light evaluation (evalBSDF, what
        // shadeOneLight evaluates) with Theia's frame is the one with the shared frame.
        EXPECT_LT(response.x, kResponse) << "anisotropic response, relative difference against the shared frame";
        // Anisotropy 0: the frame is irrelevant, the response of the new frame is the
        // one of the old frame (the output of a scene without anisotropy is unchanged).
        EXPECT_LT(response.z, kResponse) << "anisotropy 0: relative difference against the old frame";

        if (c != kDegenerateCase && response.y > 1.0e-2F) {
            ++oldFrameDiffers;
        }
    }
    // The control: the old frame gives another anisotropic response on most of the
    // surfaces. Were it equal to the shared frame everywhere, nothing here would
    // distinguish the fix from the defect.
    EXPECT_GE(oldFrameDiffers, kCases / 2U) << "cases on which the old frame's anisotropic response differs";
}
