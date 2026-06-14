// Screen-space global illumination complement.
//
// This is a presentation-only fallback for scenes that benefit from a little
// low-frequency color bleed without paying for true multi-bounce GI. It samples
// the already-lit HDR buffer in a small bilateral neighborhood and adds a soft,
// screen-space bounce term. Keep it disabled for parity renders.

[[vk::binding(0, 0)]] Texture2D<float>    t_depth;
[[vk::binding(1, 0)]] Texture2D<float4>   t_gbuffer;
[[vk::binding(2, 0)]] Texture2D<float4>   t_hdr;
[[vk::binding(3, 0)]] RWTexture2D<float4>  rw_hdr;
[[vk::binding(4, 0)]] SamplerState        s_nearest;
[[vk::binding(5, 0)]] SamplerState        s_linear;

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
    if (depth >= 1.0) {
        return;
    }

    const float4 gb = t_gbuffer.SampleLevel(s_linear, uv, 0.0);
    const float3 N = normalize(gb.xyz * 2.0 - 1.0);
    const float roughness = gb.w;
    const float diffuseWeight = saturate(1.0 - roughness);

    const float  depthRadius = 0.05;

    float3 indirect = 0.0.xxx;
    float  weightSum = 0.0;

    [unroll]
    for (int dy = -2; dy <= 2; ++dy) {
        [unroll]
        for (int dx = -2; dx <= 2; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const float2 suv = uv + float2(dx, dy) * (1.0.xx / float2(dims));
            if (any(suv < 0.0.xx) || any(suv > 1.0.xx)) {
                continue;
            }

            const float sd = t_depth.SampleLevel(s_nearest, suv, 0.0);
            if (sd >= 1.0) {
                continue;
            }

            const float4 sg = t_gbuffer.SampleLevel(s_linear, suv, 0.0);
            const float3 SN = normalize(sg.xyz * 2.0 - 1.0);
            const float  depthWeight = saturate(1.0 - abs(sd - depth) / depthRadius);
            const float  normalWeight = pow(saturate(dot(N, SN)), 4.0);
            const float  w = depthWeight * normalWeight;
            if (w <= 0.0) {
                continue;
            }

            indirect += t_hdr.SampleLevel(s_linear, suv, 0.0).rgb * w;
            weightSum += w;
        }
    }

    if (weightSum <= 0.0) {
        return;
    }

    indirect /= weightSum;
    const float4 hdr = t_hdr.SampleLevel(s_linear, uv, 0.0);
    rw_hdr[id.xy] = float4(hdr.rgb + indirect * (0.08 * diffuseWeight), hdr.a);
}
