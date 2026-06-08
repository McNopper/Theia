// Screen-Space Reflections (SSR) — linear ray march in view space.
//
// Inputs:
//   t_depth   : scene depth buffer (D32_SFLOAT sampled)
//   t_gbuffer : thin G-buffer (xyz = view-space normal * 0.5 + 0.5, w = roughness)
//   t_hdr     : HDR color buffer from the forward pass (reflected scene color source)
// Output:
//   rw_ssrOut : RGBA16F (rgb = reflected color at hit, a = hit confidence [0,1])
//
// Algorithm:
//   1. Reconstruct view-space position from depth + UV
//   2. Reflect view direction about view-space normal → reflected ray
//   3. March reflected ray in view space (kMaxSteps steps, adaptive step size)
//   4. At each step, project to screen, compare against scene depth
//   5. On hit: binary-search refine, compute edge/roughness confidence fade
//
// Ref: Morgan McGuire & Michael Mara — "Efficient GPU Screen-Space Ray Tracing"
//      (JCGT 2014) — https://jcgt.org/published/0003/04/04/
// Ref: AMD FidelityFX SSSR (MIT) — github.com/GPUOpen-Effects/FidelityFX-SSSR
// Ref: Yasin Uludag — "Hi-Z Screen-Space Cone-Traced Reflections"
//      (GPU Pro 5, CRC Press 2014)

[[vk::push_constant]]
cbuffer SSRPC {
    float4x4 proj;    // GLM perspective matrix (NOT transposed); Y-flipped for Vulkan
    float4x4 invProj; // inverse(proj)
};

[[vk::binding(0, 0)]] Texture2D<float>  t_depth;    // D32 sampled as R32
[[vk::binding(1, 0)]] Texture2D<float4> t_gbuffer;  // view-space normal + roughness
[[vk::binding(2, 0)]] Texture2D<float4> t_hdr;      // HDR scene color
[[vk::binding(3, 0)]] RWTexture2D<float4> rw_ssrOut; // SSR result
[[vk::binding(4, 0)]] SamplerState s_nearest;       // for depth (no interpolation)
[[vk::binding(5, 0)]] SamplerState s_linear;        // for GBuffer + HDR

static const int   kMaxSteps      = 64;
static const int   kBinarySteps   = 8;
static const float kThickness     = 0.5;   // view-space surface thickness (units)
static const float kRoughnessMax  = 0.45;  // skip SSR for roughness above this
static const float kRoughnessMin  = 0.06;  // skip near-mirrors: IBL is already pixel-sharp
static const float kStepFraction  = 0.025; // step = abs(viewZ) * fraction

// Reconstruct view-space position from texture UV and sampled depth.
// UV: [0,1]² with (0,0) at top-left; depth: [0,1] Vulkan depth range.
// proj is the GLM (Y-flipped) perspective matrix.
float3 viewPosFromUVDepth(float2 uv, float depth) {
    // UV (0,0) is top-left = NDC y = +1 (Vulkan Y up in NDC but Y down in framebuffer)
    float2 ndc = float2(uv.x * 2.0 - 1.0,
                        1.0 - uv.y * 2.0); // flip Y: UV y=0 → NDC y=+1
    float4 clip = float4(ndc, depth, 1.0);
    float4 view = mul(invProj, clip); // standard mat-vec (invProj not transposed)
    return view.xyz / view.w;
}

// Project view-space position to texture UV.
float2 viewPosToUV(float3 viewPos) {
    float4 clip = mul(proj, float4(viewPos, 1.0));
    float2 ndc = clip.xy / clip.w;
    // ndc.y: +1 at top → uv.y = 0; invert
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
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

    // Background (sky): no reflection
    if (depth >= 1.0) {
        rw_ssrOut[id.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    const float4 gb        = t_gbuffer.SampleLevel(s_linear, uv, 0.0);
    const float3 viewN     = normalize(gb.xyz * 2.0 - 1.0); // decode from [0,1] to [-1,1]
    const float  roughness = gb.w;

    // Skip rough surfaces — IBL already covers them well.
    // Skip near-mirrors — IBL's sharpest mip is already pixel-sharp there, and SSR's
    // coarse march only introduces false-hit blocks (no screen detail to gain).
    if (roughness > kRoughnessMax || roughness < kRoughnessMin) {
        rw_ssrOut[id.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    const float3 viewPos = viewPosFromUVDepth(uv, depth);
    const float3 viewDir = normalize(viewPos); // from camera (origin) toward surface

    const float3 reflDir = reflect(viewDir, viewN);

    // If reflected ray points toward the camera (z > 0), it won't hit anything visible
    if (reflDir.z >= 0.0) {
        rw_ssrOut[id.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Adaptive step: proportional to distance from camera
    const float stepLen  = max(abs(viewPos.z) * kStepFraction, 0.05);
    const float3 rayStep = reflDir * stepLen;

    float3 rayPos = viewPos + rayStep * 1.5; // start slightly off the surface
    float2 hitUV          = 0.0.xx;
    float  hitConfidence  = 0.0;
    bool   hit            = false;

    [loop]
    for (int i = 0; i < kMaxSteps; ++i) {
        const float2 sampleUV = viewPosToUV(rayPos);
        if (any(sampleUV < 0.0.xx) || any(sampleUV > 1.0.xx)) {
            break;
        }

        const float sceneDepth = t_depth.SampleLevel(s_nearest, sampleUV, 0.0);
        if (sceneDepth >= 1.0) {
            // Background hit — no solid surface
            rayPos += rayStep;
            continue;
        }

        const float3 sceneViewPos = viewPosFromUVDepth(sampleUV, sceneDepth);
        const float  rayZ         = rayPos.z;       // view-space Z (negative = in front)
        const float  sceneZ       = sceneViewPos.z; // view-space Z at this UV

        // Ray passed behind geometry (view-space Z more negative = further from camera)
        if (rayZ < sceneZ) {
            // Thickness test: reject if ray went too far behind (thin-surface heuristic)
            const float gap = sceneZ - rayZ; // how far behind (positive)
            if (gap < kThickness * max(1.0, abs(sceneZ) * 0.02)) {
                // Binary-search refinement for precise hit UV
                float3 lo = rayPos - rayStep;
                float3 hi = rayPos;
                [loop]
                for (int j = 0; j < kBinarySteps; ++j) {
                    float3       mid     = (lo + hi) * 0.5;
                    float2       midUV   = viewPosToUV(mid);
                    float        midD    = t_depth.SampleLevel(s_nearest, midUV, 0.0);
                    float3       midSPos = viewPosFromUVDepth(midUV, midD);
                    if (mid.z < midSPos.z) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                hitUV        = viewPosToUV((lo + hi) * 0.5);
                hitConfidence = 1.0;
                hit          = true;
            }
            break;
        }

        rayPos += rayStep;
    }

    if (!hit) {
        rw_ssrOut[id.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Fade confidence toward screen edges (avoids hard cutoff)
    const float2 edgeFade = smoothstep(0.0.xx, 0.08.xx, hitUV)
                          * smoothstep(1.0.xx, 0.92.xx, hitUV);
    hitConfidence *= edgeFade.x * edgeFade.y;

    // Roughness fade: mirror-smooth = full, kRoughnessMax = zero
    hitConfidence *= 1.0 - smoothstep(0.0, kRoughnessMax, roughness);

    // Grazing fade: reflected rays that skim nearly parallel to the screen plane
    // (|reflDir.z| small) march almost in-screen and produce unreliable silhouette
    // false-hits (hard bright blocks at object rims). Fade them out smoothly.
    hitConfidence *= smoothstep(0.05, 0.30, -reflDir.z);

    float3 reflColor = t_hdr.SampleLevel(s_linear, hitUV, 0.0).rgb;
    // Firefly suppression: at SSR's marching resolution a blown-out near-geometry
    // highlight (e.g. another sphere's mirror sun-spot) quantizes into a hard bright
    // block. Soft-clamp the reflected luminance so such hits stay plausible instead
    // of forming a white square. Ref: Karis "tonemap-weighted" SSR reduction.
    const float luma    = dot(reflColor, float3(0.2126, 0.7152, 0.0722));
    const float maxLuma = 8.0;
    if (luma > maxLuma) {
        reflColor *= maxLuma / luma;
    }
    rw_ssrOut[id.xy] = float4(reflColor, saturate(hitConfidence));
}
