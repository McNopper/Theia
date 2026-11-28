// The frame of the anisotropy of the forward fragment shader
// (shaders/forward_tangent.slang) against the frame the shared surface module builds for
// the same surface, which the GI pass and Hyperion use. The fragment has no triangle: it
// has the screen-space derivatives of its world position and UV. Over a triangle those
// are linear combinations of the two edge deltas, with the same coefficients for the
// positions and the UVs, so each case here feeds the shared `reconstructTriangleGeometry`
// with the edges of a triangle and the Theia `uvTangent` with such a combination of them.
//
// Ten cases (thread t: case = t / 5, value = t % 5), one triangle each: a normal and a
// texture coordinate that vary with the case; case 9 has degenerate texture coordinates.
// Per case five float4:
//   0  the tangent of Theia's frame       1  the tangent of the shared frame
//   2  the tangent of the frame the forward shader built before (from the normal alone)
//   3  (response of Theia's frame against the shared one, response of the old frame
//      against the shared one, response of Theia's frame against the old one at
//      anisotropy 0, the shared response itself) of the analytic-light BSDF evaluation
//   4  (Theia's bitangent . the shared bitangent, tangent . normal, |tangent|, |bitangent|)

import common;
import math;
import bsdf;
import surface;
import forward_tangent;

struct Push {
    float4* inputs;
    float4* outputs;
    uint count, pad;
};
[[vk::push_constant]] Push pc;

static const uint kValuesPerCase = 5u;
static const uint kCases = 10u;

/// The frame the forward shader built from the vertex tangent when a mesh carries none:
/// from the world up, in the plane of the normal.
void fallbackFrame(float3 N, out float3 T, out float3 B) {
    const float3 up = abs(N.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    T = normalize(cross(up, N));
    B = normalize(cross(N, T));
}

/// The material of the anisotropy cases of Harmonia's bsdf_measure_test.slang: the base
/// specular lobe alone.
GpuMaterial anisotropicSpecular(float anisotropy) {
    GpuMaterial mat = {};
    mat.baseColorWeight = float4(0.0, 0.0, 0.0, 1.0);
    mat.specularColorWeight = float4(1.0, 1.0, 1.0, 1.0);
    mat.specularRoughAnisoIor = float4(0.3, anisotropy, 1.5, 0.0);
    mat.opacityFlagsPad = float4(1.0, 0.0, 0.0, 0.0);
    return mat;
}

struct Case {
    float3 N;          ///< shading normal, and the plane of the triangle
    float3 e1, e2;     ///< orthonormal basis of that plane
    float3 p1, p2;     ///< the triangle's edges (p0 is the origin)
    float2 uv1, uv2;   ///< the texture coordinates of their ends (uv0 is the origin)
};

Case makeCase(uint index) {
    Case c;
    const float a = 0.7 * float(index);
    const float b = 0.35 + 0.31 * float(index);
    c.N = normalize(float3(cos(a) * sin(b), cos(b), sin(a) * sin(b)));
    c.e1 = normalize(cross(c.N, float3(0.31, 0.17, 0.93)));
    c.e2 = cross(c.N, c.e1);
    c.p1 = c.e1 * 1.3 + c.e2 * 0.4;
    c.p2 = c.e1 * -0.6 + c.e2 * 2.1;
    // The texture coordinates of the second edge are turned by an angle that varies
    // with the case, so the direction of increasing u turns with it.
    const float theta = 0.6 * float(index);
    const float2 loose = float2(-0.3, 1.7);
    c.uv1 = float2(1.0, 0.25);
    c.uv2 = float2(cos(theta) * loose.x - sin(theta) * loose.y, sin(theta) * loose.x + cos(theta) * loose.y);
    if (index == kCases - 1u) {
        c.uv1 = float2(0.0, 0.0); // texture coordinates without a u direction
        c.uv2 = float2(0.0, 0.0);
    }
    return c;
}

/// What the fragment stage sees: two "derivatives", each the same mix of the two edges
/// in position and in UV.
void derivativePairs(Case c, out float3 dp1, out float3 dp2, out float2 duv1, out float2 duv2) {
    dp1 = c.p1 * 0.7 + c.p2 * 0.2;
    dp2 = c.p1 * -0.1 + c.p2 * 0.9;
    duv1 = c.uv1 * 0.7 + c.uv2 * 0.2;
    duv2 = c.uv1 * -0.1 + c.uv2 * 0.9;
}

/// Sum of |f_A - f_B| over the light directions, over the sum of |f_ref|.
float responseDifference(GpuMaterial mat, Case c, float3 nA, float3 tA, float3 bA, float3 nB, float3 tB, float3 bB,
                         float3 refT, float3 refB, out float response) {
    const float3 wo = normalize(c.N * 0.8 + c.e1 * 0.5 + c.e2 * 0.3);
    float3 difference = 0.0.xxx;
    float3 total = 0.0.xxx;
    [unroll]
    for (uint i = 0u; i < 8u; ++i) {
        const float phi = 2.4 * float(i);
        const float s = sqrt(max(0.0, 1.0 - (0.3 + 0.08 * float(i)) * (0.3 + 0.08 * float(i))));
        const float3 wi = c.N * (0.3 + 0.08 * float(i)) + c.e1 * (s * cos(phi)) + c.e2 * (s * sin(phi));
        const float3 fA = evalBSDF(mat, wo, wi, nA, tA, bA, nA, tA, bA);
        const float3 fB = evalBSDF(mat, wo, wi, nB, tB, bB, nB, tB, bB);
        const float3 fRef = evalBSDF(mat, wo, wi, nA, refT, refB, nA, refT, refB);
        difference += abs(fA - fB);
        total += fRef;
    }
    response = total.x + total.y + total.z;
    return (difference.x + difference.y + difference.z) / max(response, 1.0e-20);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint t = id.x;
    if (t >= pc.count) return;
    const uint index = t / kValuesPerCase;
    const uint value = t % kValuesPerCase;
    if (index >= kCases) return;

    const Case c = makeCase(index);
    float3 dp1, dp2;
    float2 duv1, duv2;
    derivativePairs(c, dp1, dp2, duv1, duv2);

    // Theia's forward frame: the fragment's derivatives through its own module.
    float3 Tn, Bn;
    buildAnisotropyFrame(c.N, uvTangent(dp1, dp2, duv1, duv2), Tn, Bn);
    // The frame the GI pass and Hyperion build: the shared module on the triangle.
    const TriangleGeometry g = reconstructTriangleGeometry(
        float3(0.0, 0.0, 0.0), c.p1, c.p2, c.N, c.N, c.N, float2(0.0, 0.0), c.uv1, c.uv2, float3(0.2, 0.3, 0.5));
    float3 Ts, Bs;
    buildAnisotropyFrame(c.N, g.tangent, Ts, Bs);
    // The frame the forward shader had before, with a mesh that carries no vertex tangent.
    float3 To, Bo;
    fallbackFrame(c.N, To, Bo);

    if (value == 0u) pc.outputs[t] = float4(Tn, 0.0);
    if (value == 1u) pc.outputs[t] = float4(Ts, 0.0);
    if (value == 2u) pc.outputs[t] = float4(To, 0.0);
    if (value == 3u) {
        float response;
        const float newAgainstShared = responseDifference(anisotropicSpecular(0.9), c, c.N, Tn, Bn, c.N, Ts, Bs, Ts, Bs, response);
        const float oldAgainstShared = responseDifference(anisotropicSpecular(0.9), c, c.N, To, Bo, c.N, Ts, Bs, Ts, Bs, response);
        float unusedResponse;
        const float isotropic = responseDifference(anisotropicSpecular(0.0), c, c.N, Tn, Bn, c.N, To, Bo, Ts, Bs, unusedResponse);
        pc.outputs[t] = float4(newAgainstShared, oldAgainstShared, isotropic, response);
    }
    if (value == 4u) {
        pc.outputs[t] = float4(dot(Bn, Bs), dot(Tn, c.N), length(Tn), length(Bn));
    }
}
