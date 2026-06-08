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

static const int kSamples = 16;

// Minimum cosine between (occluder - receiver) and the surface normal for the
// occluder to count. Rejects near-coplanar samples (flat ground), which is the
// scale-invariant cure for false self-occlusion / banding on open floors.
static const float kPlaneBias = 0.1;

// 16 hemisphere kernel samples (z >= 0), weighted toward the origin.
static const float3 kKernel[16] = {
    float3( 0.0492f,  0.0288f,  0.0410f), float3(-0.0712f,  0.1063f,  0.0532f),
    float3( 0.1284f, -0.0552f,  0.1426f), float3(-0.1632f, -0.1418f,  0.0894f),
    float3( 0.0231f,  0.2104f,  0.1772f), float3( 0.2693f,  0.0584f,  0.0726f),
    float3(-0.1547f,  0.1992f,  0.2531f), float3(-0.2865f, -0.1746f,  0.1183f),
    float3( 0.1138f, -0.3214f,  0.2042f), float3( 0.3521f,  0.1428f,  0.2675f),
    float3(-0.0826f,  0.0517f,  0.4893f), float3(-0.4012f,  0.2876f,  0.1764f),
    float3( 0.2417f, -0.4153f,  0.3318f), float3( 0.5126f, -0.0892f,  0.1925f),
    float3(-0.3284f, -0.4621f,  0.4057f), float3( 0.1043f,  0.6182f,  0.5231f),
};

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

    // Build a TBN frame around the view-space normal with a per-pixel random tangent
    // (rotates the kernel to hide banding).
    const float  rnd    = hash12(float2(id.xy)) * 6.2831853;
    const float3 randVec = normalize(float3(cos(rnd), sin(rnd), 0.0));
    const float3 T = normalize(randVec - viewN * dot(randVec, viewN));
    const float3 B = cross(viewN, T);
    const float3x3 TBN = float3x3(T, B, viewN);

    float occlusion = 0.0;
    [unroll]
    for (int i = 0; i < kSamples; ++i) {
        // Sample position in view space, oriented to the surface hemisphere.
        float3 samplePos = viewP + mul(kKernel[i], TBN) * r;

        float2 sampleUV = viewPosToUV(samplePos);
        if (any(sampleUV < 0.0.xx) || any(sampleUV > 1.0.xx)) {
            continue;
        }

        float sceneDepth = t_depth.SampleLevel(s_nearest, sampleUV, 0.0);
        if (sceneDepth >= 1.0) {
            continue; // sky behind the sample — not an occluder
        }
        // Reconstruct the actual scene surface point seen at this screen location.
        float3 occluderPos = viewPosFromUVDepth(sampleUV, sceneDepth);
        float  sceneZ      = occluderPos.z;

        // Normal-aware rejection: a real occluder must lie ABOVE the receiver's
        // tangent plane (positive component along the surface normal). On a large
        // flat plane (e.g. the studio floor) neighbouring samples are coplanar, so
        // (occluderPos - viewP) is perpendicular to viewN → rejected. This kills the
        // false self-occlusion / banding that otherwise darkens open flat ground.
        float3 toOcc    = occluderPos - viewP;
        float  occDist  = length(toOcc);
        if (occDist < 1e-5) {
            continue;
        }
        float planeCos = dot(toOcc / occDist, viewN); // >0 → occluder rises off the surface
        if (planeCos <= kPlaneBias) {
            continue; // coplanar (flat ground) or below the tangent plane — not an occluder
        }

        // Occluded when the scene surface is closer to the camera (larger view-space Z)
        // than the sample point, within the sampling radius.
        float rangeCheck = smoothstep(0.0, 1.0, r / max(abs(viewP.z - sceneZ), 1e-4));
        if (sceneZ >= samplePos.z + bias * abs(viewP.z)) {
            occlusion += rangeCheck * saturate(planeCos);
        }
    }

    occlusion = occlusion / float(kSamples);
    float ao  = pow(saturate(1.0 - occlusion * intensity), power);

    rw_ao[id.xy] = ao;
}
