// SSR composite: blends the SSR hit buffer into the HDR render target.
//
// The forward pass already contains IBL specular as a soft fallback.
// SSR adds precise screen-space specular on top for low-roughness surfaces
// where IBL's environment prefilter is too blurry to capture screen content.
//
// Blend strategy (replace, not add):
//   hdr.rgb = lerp(hdr.rgb, ssrColor, blendWeight)
//   blendWeight = ssrConfidence * ssrStrength * schlickFresnel * (1 - roughness/roughnessMax)
//
// The forward pass already wrote a COMPLETE mirror reflection of the distant
// environment (IBL specular). For near-mirror surfaces, screen-space geometry
// (other objects, the floor) OCCLUDES that distant environment — so a confident
// SSR hit must REPLACE the IBL env reflection, never add on top of it. An additive
// blend double-counts and produces blown-out blocky artifacts on smooth metals.
//
// Ref: AMD FidelityFX SSSR composite pass (MIT) — github.com/GPUOpen-Effects/FidelityFX-SSSR
// Ref: Unity HDRP SSSR composite — github.com/Unity-Technologies/Graphics

[[vk::push_constant]]
cbuffer CompositePC {
    float ssrStrength;   // global blend scale [0,1]
    float roughnessMax;  // must match ssr.comp kRoughnessMax (0.45)
    float _pad0;
    float _pad1;
};

[[vk::binding(0, 0)]] Texture2D<float4>   t_ssrResult;  // rgb=reflected color, a=confidence
[[vk::binding(1, 0)]] Texture2D<float4>   t_gbuffer;    // xyz=viewN, w=roughness
[[vk::binding(2, 0)]] RWTexture2D<float4> rw_hdr;       // HDR buffer (in-place composite)
[[vk::binding(3, 0)]] SamplerState        s_linear;

// Schlick Fresnel at normal incidence approximation for dielectric coat/specular.
// Uses roughness as a proxy for F0 when per-material F0 is not available.
float schlickFresnel0(float roughness) {
    // Typical dielectric F0 ≈ 0.04; metals ≈ 1. Use a blend.
    return lerp(0.04, 1.0, 1.0 - roughness);
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 dims;
    rw_hdr.GetDimensions(dims.x, dims.y);
    if (any(id.xy >= dims)) {
        return;
    }

    const float4 ssr = t_ssrResult[id.xy];
    if (ssr.a <= 0.001) {
        return; // no SSR hit — skip this pixel
    }

    const float4 gb        = t_gbuffer[id.xy];
    const float  roughness = gb.w;

    // SSR weight: high for low-roughness, zero for rough surfaces
    const float roughWeight = saturate(1.0 - roughness / max(roughnessMax, 0.001));
    // Simple Fresnel proxy at normal incidence
    const float fresnel     = schlickFresnel0(roughness);

    const float blendWeight = saturate(ssr.a * ssrStrength * roughWeight * fresnel);
    if (blendWeight <= 0.0) {
        return;
    }

    const float4 hdr      = rw_hdr[id.xy];
    // Replace (lerp) rather than add: a confident screen-space hit means near
    // geometry occludes the distant environment already present in hdr.rgb.
    rw_hdr[id.xy] = float4(lerp(hdr.rgb, ssr.rgb, blendWeight), hdr.a);
}
