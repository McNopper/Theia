// Screen-Space Ambient Occlusion / contact shadows.
//
// IBL-only scenes (e.g. ABeautifulGame) have no analytic shadow casters, so geometry
// that sits in creases or under other geometry looks unnaturally bright. This pass
// approximates the missing visibility term by sampling a hemisphere of view-space
// points around each surface and testing them against the depth buffer, then
// multiplies the composited HDR colour by the resulting occlusion factor.
//
// Inputs:
//   t_depth   : scene depth (D32 sampled as R32)
//   t_gbuffer : thin G-buffer (xyz = view-space normal * 0.5 + 0.5, w = roughness)
//   rw_ao     : single-channel AO factor output (denoised + applied by ssao_blur.comp)
//
// Ref: Crytek SSAO (Mittring, "Finding Next Gen", SIGGRAPH 2007)
// Ref: John Chapman — "SSAO Tutorial" (john-chapman-graphics.blogspot.com)
// Ref: HBAO/GTAO-style horizon search (deterministic directional sectors)

[[vk::push_constant]]
cbuffer SSAOPC {
    float4x4 proj;      // GLM perspective matrix (NOT transposed); Y-flipped for Vulkan
    float4x4 invProj;   // inverse(proj)
    float    radius;    // view-space sampling radius (scene units)
    float    intensity; // occlusion strength
    float    bias;      // depth bias to avoid self-occlusion
    float    power;     // AO falloff exponent
};

[[vk::binding(0, 0)]] Texture2D<float>    t_depth;
[[vk::binding(1, 0)]] Texture2D<float4>   t_gbuffer;
[[vk::binding(2, 0)]] RWTexture2D<float>  rw_ao;
[[vk::binding(3, 0)]] SamplerState        s_nearest;
[[vk::binding(4, 0)]] SamplerState        s_linear;

static const int kDirections = 8;
static const int kStepsPerDirection = 4;

// Minimum cosine between (occluder - receiver) and the surface normal for the
// occluder to count. Rejects near-coplanar samples (flat ground), which is the
// scale-invariant cure for false self-occlusion / banding on open floors.
static const float kPlaneBias = 0.1;

float3 viewPosFromUVDepth(float2 uv, float depth) {
    float2 ndc  = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 clip = float4(ndc, depth, 1.0);
    float4 view = mul(invProj, clip);
    return view.xyz / view.w;
}

float2 viewPosToUV(float3 viewPos) {
    float4 clip = mul(proj, float4(viewPos, 1.0));
    float2 ndc  = clip.xy / clip.w;
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

float hash12(float2 p) {
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
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

    // Background (sky): no occlusion, write fully-lit factor.
    if (depth >= 1.0) {
        rw_ao[id.xy] = 1.0;
        return;
    }

    const float4 gb    = t_gbuffer.SampleLevel(s_linear, uv, 0.0);
    const float3 viewN = normalize(gb.xyz * 2.0 - 1.0);
    const float3 viewP = viewPosFromUVDepth(uv, depth);

    // Scale the sampling radius by view-space distance so the screen-space footprint
    // stays roughly constant and the effect auto-adapts across scene scales
    // (sub-unit ABeautifulGame ↔ ~560-unit Cornell).
    const float r = radius * abs(viewP.z);

    // Build a TBN frame around the view-space normal with a per-pixel random rotation
    // so the directional sectors do not line up with screen-space axes.
    const float  rnd    = hash12(float2(id.xy)) * 6.2831853;
    const float3 randVec = normalize(float3(cos(rnd), sin(rnd), 0.0));
    const float3 T = normalize(randVec - viewN * dot(randVec, viewN));
    const float3 B = cross(viewN, T);
    const float3x3 TBN = float3x3(T, B, viewN);

    float occlusion = 0.0;
    [unroll]
    for (int dirIdx = 0; dirIdx < kDirections; ++dirIdx) {
        const float angle = rnd + (6.2831853f * float(dirIdx)) / float(kDirections);
        const float2 dir2 = float2(cos(angle), sin(angle));
        const float3 tangentDir = normalize(mul(float3(dir2.x, dir2.y, 0.0), TBN));

        float dirOcclusion = 0.0;
        [unroll]
        for (int stepIdx = 1; stepIdx <= kStepsPerDirection; ++stepIdx) {
            const float stepFrac = (float(stepIdx) + 0.5f) / float(kStepsPerDirection);
            const float3 samplePos = viewP + tangentDir * (r * stepFrac);
            const float2 sampleUV = viewPosToUV(samplePos);
            if (any(sampleUV < 0.0.xx) || any(sampleUV > 1.0.xx)) {
                continue;
            }

            const float sceneDepth = t_depth.SampleLevel(s_nearest, sampleUV, 0.0);
            if (sceneDepth >= 1.0) {
                continue; // sky behind the sample — not an occluder
            }

            const float3 occluderPos = viewPosFromUVDepth(sampleUV, sceneDepth);
            const float3 toOcc = occluderPos - viewP;
            const float occDist = length(toOcc);
            if (occDist < 1e-5) {
                continue;
            }

            const float planeCos = dot(toOcc / occDist, viewN);
            if (planeCos <= kPlaneBias) {
                continue; // coplanar or below the tangent plane — not an occluder
            }

            const float rangeWeight = 1.0 - saturate(occDist / r);
            dirOcclusion = max(dirOcclusion, rangeWeight * saturate(planeCos));
        }

        occlusion += dirOcclusion;
    }

    occlusion = occlusion / float(kDirections);
    float ao  = pow(saturate(1.0 - occlusion * intensity), power);

    rw_ao[id.xy] = ao;
}
