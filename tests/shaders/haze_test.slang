// The distance haze of Harmonia/shaders/haze.slang, the module both renderers composite
// their camera rays through, against a numerical integral of the medium it models:
//
//     sigma(h) = density * exp(-h / height)          tau(a -> b) = integral sigma(h(s)) ds
//
// The independent integral is taken in double precision by tests/module/test_haze.cpp;
// this shader evaluates only the closed-form module and its composite/transmittance.
// One thread per value of a segment; the segments come from the push constants,
// three float4 per segment:
//
//     inputs[3k + 0] = (density, height, -, -)     the scene's [haze] section
//     inputs[3k + 1] = (a.xyz, -)                  where the camera ray starts
//     inputs[3k + 2] = (b.xyz, -)                  where it ends
//
// Per segment two float4:
//   0  (tau of the module, 0, transmittance of tau, 0)
//   1  (hazeComposite of a radiance through tau, with a sky, .xyz; hazeTransmittance of tau)
//
// The segments are followed by four widths of environment map (tests/module/test_haze.cpp
// names them), one thread each (value 0):
//   0  (width, hazeSkyLod of the width, the degrees one texel spans at that level,
//      the blur the module is written for)

import haze;

struct Push {
    float4* inputs;
    float4* outputs;
    uint count, pad;
};
[[vk::push_constant]] Push pc;

static const uint kInputsPerCase = 3u;
static const uint kValuesPerCase = 2u;
static const uint kSegments = 14u;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint t = id.x;
    if (t >= pc.count) return;
    const uint k = t / kValuesPerCase;   // the segment
    const uint value = t % kValuesPerCase;

    if (k >= kSegments) {
        // The blur of the in-scattered sky on maps of four widths: 1K, the 3840 the
        // fixed level was tuned on, 4K and 8K.
        const float widths[4] = {1024.0, 3840.0, 4096.0, 8192.0};
        const float width = widths[k - kSegments];
        const float lod = hazeSkyLod(width);
        pc.outputs[t] = float4(width, lod, 360.0 / (width / exp2(lod)), kHazeSkyBlurDegrees);
        return;
    }

    const float density = pc.inputs[k * kInputsPerCase + 0u].x;
    const float height = pc.inputs[k * kInputsPerCase + 0u].y;
    const float3 a = pc.inputs[k * kInputsPerCase + 1u].xyz;
    const float3 b = pc.inputs[k * kInputsPerCase + 2u].xyz;

    const float tau = hazeOpticalDepth(Haze(density, height), a, b);

    if (value == 0u) {
        pc.outputs[t] = float4(tau, 0.0, hazeTransmittance(tau), 0.0);
    } else {
        // A radiance and the sky behind it, unlike each other in every channel.
        const float3 radiance = float3(0.5, 2.0, 7.0);
        const float3 sky = float3(0.25, 0.5, 1.0);
        pc.outputs[t] = float4(hazeComposite(radiance, tau, sky), hazeTransmittance(tau));
    }
}
