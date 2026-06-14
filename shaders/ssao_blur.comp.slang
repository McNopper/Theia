// SSAO bilateral blur + apply.
//
// The raw SSAO pass (ssao.comp) produces a noisy per-pixel occlusion factor because it
// rotates a small hemisphere kernel by a per-pixel random angle. This pass removes that
// grain with a depth-aware (bilateral) box blur and attenuates only the indirect/ambient
// fraction of the composited HDR colour, yielding clean soft contact shadows instead of
// dimming direct lighting.
//
// Edge-aware weighting uses view-space Z (reconstructed from depth via invProj) so the
// blur does not bleed occlusion across depth discontinuities (e.g. a piece against the
// board, or a box edge against a far wall).
//
// Ref: NVIDIA HBAO+ denoise; "A Gentle Introduction to Bilateral Filtering"

[[vk::push_constant]]
cbuffer BlurPC {
    float4x4 invProj;   // inverse(proj) — to linearise depth
    float2   texelSize; // 1/width, 1/height
    float2   _pad;
};

[[vk::binding(0, 0)]] Texture2D<float>    t_ao;
[[vk::binding(1, 0)]] Texture2D<float>    t_depth;
[[vk::binding(2, 0)]] RWTexture2D<float4> rw_hdr;
[[vk::binding(3, 0)]] SamplerState        s_nearest;
[[vk::binding(4, 0)]] SamplerState        s_linear;

static const int kRadius = 3; // 7x7 kernel

float viewZ(float2 uv, float depth) {
    float2 ndc  = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 view = mul(invProj, float4(ndc, depth, 1.0));
    return view.z / view.w;
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 dims;
    t_depth.GetDimensions(dims.x, dims.y);
    if (any(id.xy >= dims)) {
        return;
    }

    const float2 uv    = (float2(id.xy) + 0.5) / float2(dims);
    const float  depth = t_depth.SampleLevel(s_nearest, uv, 0.0);

    // Sky: no occlusion, leave HDR untouched.
    if (depth >= 1.0) {
        return;
    }

    const float centerZ = viewZ(uv, depth);
    // Depth tolerance scales with distance so the filter behaves across scene scales.
    const float zTol = max(0.05 * abs(centerZ), 1e-3);

    float aoSum = 0.0;
    float wSum  = 0.0;
    [unroll]
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
        [unroll]
        for (int dx = -kRadius; dx <= kRadius; ++dx) {
            float2 suv = uv + float2(dx, dy) * texelSize;
            float  sd  = t_depth.SampleLevel(s_nearest, suv, 0.0);
            if (sd >= 1.0) {
                continue;
            }
            float sz = viewZ(suv, sd);
            // Bilateral weight: drop samples on the far side of a depth edge.
            float w = exp(-(sz - centerZ) * (sz - centerZ) / (2.0 * zTol * zTol));
            aoSum += t_ao.SampleLevel(s_nearest, suv, 0.0) * w;
            wSum  += w;
        }
    }

    float ao = (wSum > 0.0) ? (aoSum / wSum) : t_ao.SampleLevel(s_nearest, uv, 0.0);

    float4 hdr = rw_hdr[id.xy];
    const float indirectWeight = saturate(hdr.a);
    const float aoMix = lerp(1.0, ao, indirectWeight);
    rw_hdr[id.xy] = float4(hdr.rgb * aoMix, hdr.a);
}
