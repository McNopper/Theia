// gi.comp.slang — ray-query global-illumination compute stage.
//
// Replaces Theia's flat-IBL indirect approximation with a correct unidirectional
// Monte-Carlo estimator that converges to the Hyperion ground truth under frame
// accumulation. Runs over the forward pass G-buffer: for every primary hit it spawns
// a BSDF continuation ray and walks 1..maxDepth path segments, evaluating the shared
// Harmonia `integrateSurface` estimator at each secondary vertex (NEE for emissive
// triangles + environment, emission with MIS guard, BSDF continuation, Russian
// roulette). Shadow / continuation rays use inline VK_KHR_ray_query (no shader tables).
//
// Division of labour with the forward raster pass (giEnabled != 0):
//   • forward pass  → direct analytic lighting + emission (no IBL/ambient)
//   • this GI pass  → indirect (bounce) light + primary-vertex ENV NEE only
// Primary analytic/emissive direct remains owned by forward; the GI pass contributes
// primary environment direct via shared env-NEE MIS plus bounce transport.

import common;          // GpuMaterial, GpuVertex, GpuInstance, GpuEmissiveTriangle, kRayEpsilon, kInstanceMaskAll, kInvPi
import math;            // buildTBN, randFloat
import rng;             // composeRngSeed
import bsdf;            // evalBSDF / pdfBSDF / sampleBSDF + BSDFSample (resolves to Theia's bsdf.slang)
import env_sample;      // evalEnvImportancePdfDir
import path_integrator; // ITracer, SurfaceHit, PathState, EnvNeeSample, evalEnvNEESample, integrateSurface, sampleEnvMapLatLong

[[vk::push_constant]]
cbuffer GiPC {
    float4x4 view;                  // glm::transpose(view) — same upload convention as ForwardPC
    float4   cameraPos;             // xyz = world-space camera position
    float    exposure;              // 1 / (1.2 * 2^EV100) — matches forward pass
    uint     frameSampleIndex;      // per-frame sample counter (drives RNG decorrelation across accumulated frames)
    uint     rngBaseSeed;           // base seed for composeRngSeed()
    uint     emissiveTriangleCount; // number of emissive triangles for NEE
    uint     envImportanceWidth;    // env CDF width; 0 disables env importance sampling
    uint     envImportanceHeight;   // env CDF height
    uint     hasEnvMap;             // 1 when an environment map is bound
    float    envLuminanceScale;     // env_unit_nits — physical scale applied to raw env samples
    uint     maxDepth;              // maximum number of traced bounce segments
    uint     screenWidth;
    uint     screenHeight;
    uint     _pad0;
};

[[vk::binding(0, 0)]]  RaytracingAccelerationStructure       t_tlas;
[[vk::binding(1, 0)]]  RWTexture2D<float4>                   hdrImage;
[[vk::binding(2, 0)]]  Texture2D<float4>                     t_giBuffer;   // xyz = worldPos, w = asfloat(matIdx+1)
[[vk::binding(3, 0)]]  Texture2D<float4>                     t_gbuffer;    // xyz = viewN*0.5+0.5, w = roughness
[[vk::binding(4, 0)]]  Texture2D<float4>                     t_envMap;
[[vk::binding(5, 0)]]  SamplerState                          s_env;
[[vk::binding(6, 0)]]  StructuredBuffer<GpuMaterial>         materials;
[[vk::binding(7, 0)]]  StructuredBuffer<GpuVertex>           vertices;
[[vk::binding(8, 0)]]  StructuredBuffer<GpuInstance>         instances;
[[vk::binding(9, 0)]]  StructuredBuffer<uint>                indices;
[[vk::binding(10, 0)]] StructuredBuffer<GpuEmissiveTriangle> emissiveTriangles;
[[vk::binding(11, 0)]] StructuredBuffer<float>               envMarginalCdf;
[[vk::binding(12, 0)]] StructuredBuffer<float>               envConditionalCdf;

// Per-sample firefly clamp on the indirect contribution (display-referred, post-exposure),
// matching the forward pass kFireflyClampDisplay threshold.
static const float kFireflyClampDisplay = 100.0;

/// Inline-ray-query shadow tracer satisfying the shared estimator's ITracer interface.
struct TheiaTracer : ITracer {
    bool traceOccluded(float3 origin, float3 direction, float tMax) {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = 0.0;
        ray.TMax = tMax;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(t_tlas, RAY_FLAG_NONE, kInstanceMaskAll, ray);
        q.Proceed();
        // Returns true when the segment is UNOCCLUDED (no committed triangle hit).
        return q.CommittedStatus() != COMMITTED_TRIANGLE_HIT;
    }
};

struct GiHit {
    bool   hit;
    uint   materialIdx;
    float3 worldPos;
    float3 geoNormal; // geometric normal, oriented to face the incoming ray's origin
};

float maxComp(float3 v) { return max(v.x, max(v.y, v.z)); }

/// Closest-hit query against the scene TLAS, reconstructing position/material/normal.
GiHit traceClosestHit(float3 origin, float3 dir) {
    GiHit hit;
    hit.hit = false;
    hit.materialIdx = 0u;
    hit.worldPos = 0.0.xxx;
    hit.geoNormal = float3(0.0, 1.0, 0.0);

    const float3 rayDir = normalize(dir);
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = rayDir;
    ray.TMin = kRayEpsilon;
    ray.TMax = kInf;

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(t_tlas, RAY_FLAG_NONE, kInstanceMaskAll, ray);
    while (q.Proceed()) {}
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return hit;
    }

    const uint instIdx = q.CommittedInstanceID();
    const bool frontFace = q.CommittedTriangleFrontFace();
    const GpuInstance inst = instances[instIdx];
    const uint tri = q.CommittedPrimitiveIndex();
    const uint base = inst.indexOffset + tri * 3u;
    if (base + 2u >= inst.indexOffset + inst.indexCount) {
        return hit;
    }

    const uint i0 = indices[base + 0u];
    const uint i1 = indices[base + 1u];
    const uint i2 = indices[base + 2u];
    const float3 p0 = vertices[i0].position;
    const float3 p1 = vertices[i1].position;
    const float3 p2 = vertices[i2].position;

    const float2 b = q.CommittedTriangleBarycentrics();
    const float3 bary = float3(1.0 - b.x - b.y, b.x, b.y);
    float3 Ng = normalize(cross(p1 - p0, p2 - p0));
    if (!frontFace) {
        Ng = -Ng;
    }

    hit.worldPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    hit.geoNormal = Ng;
    hit.materialIdx = inst.materialIndex;
    hit.hit = true;
    return hit;
}

/// Populate a SurfaceHit from a traced hit, applying the shared geometric-normal
/// frame (no texture maps at secondary vertices — indirect light is low-frequency).
SurfaceHit makeSurfaceHit(GiHit h, float3 wo) {
    SurfaceHit surf;
    float3 N = h.geoNormal;
    if (dot(N, wo) < 0.0) {
        N = -N;
    }
    const float3x3 tbn = buildTBN(N);
    surf.position = h.worldPos;
    surf.normal = N;
    surf.geoNormal = N;
    surf.T = tbn[0];
    surf.B = tbn[1];
    surf.coatN = N;
    surf.coatT = tbn[0];
    surf.coatB = tbn[1];
    surf.mat = materials[h.materialIdx];
    surf.wo = wo;
    return surf;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= screenWidth || tid.y >= screenHeight) {
        return;
    }
    const uint2 px = tid.xy;
    const int3 loadCoord = int3(int(px.x), int(px.y), 0);

    // --- Primary surface from the forward G-buffer ---
    const float4 giSample = t_giBuffer.Load(loadCoord);
    const uint matIdxPlus1 = asuint(giSample.w);
    if (matIdxPlus1 == 0u) {
        return; // background / non-geometry pixel (and transparent pixels keep the sentinel)
    }
    const uint materialIdx = matIdxPlus1 - 1u;
    const float3 worldPos = giSample.xyz;

    GpuMaterial mat = materials[materialIdx];
    // Guard against a materialIdx that indexes uninitialized / garbage material data in
    // the GI compute's descriptor set: a non-finite material would feed NaN/Inf straight
    // into the BSDF eval, ahead of the downstream isfinite() guards.
    if (!all(isfinite(mat.baseColorWeight.xyz)) || !all(isfinite(mat.emissionColorLum.xyz))) {
        return;
    }

    // Skip self-emissive primary surfaces: their radiance is the emitter itself, shown
    // directly by the forward pass; bounced light off an emitter is negligible.
    const float3 emission = max(mat.emissionColorLum.xyz, 0.0.xxx) * max(mat.emissionColorLum.w, 0.0);
    if (maxComp(emission) > 0.0) {
        return;
    }

    // Reconstruct the world-space shading normal from the view-space G-buffer normal.
    // viewN = mul(N, viewRot) in the forward pass (row-vector); the inverse for an
    // orthonormal rotation is mul(viewRot, viewN) (column-vector). See forward_render.frag.
    const float3 viewN = t_gbuffer.Load(loadCoord).xyz * 2.0 - 1.0;
    const float3x3 viewRot = float3x3(view[0].xyz, view[1].xyz, view[2].xyz);
    float3 worldN = mul(viewRot, viewN);
    if (dot(worldN, worldN) < 1.0e-8) {
        return;
    }
    worldN = normalize(worldN);

    const float3 wo = normalize(cameraPos.xyz - worldPos);
    if (dot(worldN, wo) < 0.0) {
        worldN = -worldN;
    }
    const float3x3 tbn = buildTBN(worldN);

    SurfaceHit primary;
    primary.position = worldPos;
    primary.normal = worldN;
    primary.geoNormal = worldN;
    primary.T = tbn[0];
    primary.B = tbn[1];
    primary.coatN = worldN;
    primary.coatT = tbn[0];
    primary.coatB = tbn[1];
    primary.mat = mat;
    primary.wo = wo;

    uint seed = composeRngSeed(px, frameSampleIndex, 0u, rngBaseSeed);

    PathState state;
    state.radiance = 0.0.xxx;
    state.throughput = 1.0.xxx;
    state.depth = 0u;
    state.seed = seed;
    state.terminated = false;
    state.specularBounce = false;
    state.lastBsdfPdf = 0.0;

    const bool envEnabled = (hasEnvMap != 0u);
    const bool envImportance = envEnabled && envImportanceWidth > 0u && envImportanceHeight > 0u;
    TheiaTracer tracer;

    // Primary-vertex environment NEE only (shared estimator conventions). Keep this in
    // a separate accumulator so firefly suppression does not bias the sun term.
    float3 primaryEnvNee = 0.0.xxx;
    if (envImportance) {
        EnvNeeSample envNee = evalEnvNEESample(
            tracer, primary,
            envMarginalCdf, envConditionalCdf,
            envImportanceWidth, envImportanceHeight, envEnabled, envLuminanceScale,
            t_envMap, s_env, state.seed);
        primaryEnvNee = envNee.contribution;
        if (!all(isfinite(primaryEnvNee)) || any(isnan(primaryEnvNee))) {
            primaryEnvNee = 0.0.xxx;
        }
    }

    // --- Primary BSDF continuation (primary env NEE above; analytic/emissive direct handled by forward) ---
    BSDFSample bs = sampleBSDF(primary.mat, primary.wo,
        primary.normal, primary.T, primary.B, primary.coatN, primary.coatT, primary.coatB, state.seed);
    if (bs.pdf <= 0.0 || maxComp(bs.weight) <= 0.0 || !all(isfinite(bs.weight))) {
        return;
    }
    state.throughput = bs.weight;
    state.specularBounce = bs.isDelta;
    state.lastBsdfPdf = bs.isDelta ? 1.0e30 : bs.pdf;

    float3 origin = worldPos + primary.geoNormal * (dot(bs.direction, primary.geoNormal) >= 0.0 ? kRayEpsilon : -kRayEpsilon);
    float3 dir = normalize(bs.direction);

    const uint bounces = max(maxDepth, 1u);
    for (uint bounce = 0u; bounce < bounces; ++bounce) {
        GiHit h = traceClosestHit(origin, dir);
        if (!h.hit) {
            if (envEnabled) {
                const float3 Le = sampleEnvMapLatLong(t_envMap, s_env, dir) * envLuminanceScale;
                float misW = 1.0;
                // Weight BSDF-to-env against the matching environment-NEE strategy.
                if (envImportance) {
                    const float envPdf = evalEnvImportancePdfDir(
                        envMarginalCdf, envConditionalCdf, envImportanceWidth, envImportanceHeight, dir);
                    misW = state.lastBsdfPdf / max(state.lastBsdfPdf + envPdf, 1.0e-6);
                }
                state.radiance += state.throughput * Le * misW;
            }
            break;
        }

        SurfaceHit surf = makeSurfaceHit(h, -dir);
        // Reject a non-finite secondary material rather than propagating NaN through
        // integrateSurface — terminate the path and keep the radiance accumulated so far.
        if (!all(isfinite(surf.mat.baseColorWeight.xyz)) || !all(isfinite(surf.mat.emissionColorLum.xyz))) {
            break;
        }
        // Prevent primary direct-light double count: camera->primary->emissive (bounce 0)
        // is forward-owned direct (RectLight/emissive proxy), so GI must not add it.
        const float3 surfEmission = max(surf.mat.emissionColorLum.xyz, 0.0.xxx)
                                  * max(surf.mat.emissionColorLum.w, 0.0);
        if (bounce == 0u && maxComp(surfEmission) > 0.0 && dot(surf.wo, surf.geoNormal) > 0.0) {
            break;
        }
        state.depth = bounce + 1u;
        integrateSurface(tracer, surf,
            emissiveTriangleCount, emissiveTriangles,
            envMarginalCdf, envConditionalCdf,
            envImportanceWidth, envImportanceHeight, envEnabled, envLuminanceScale,
            t_envMap, s_env, state);

        if (state.terminated) {
            break;
        }
        origin = state.nextOrigin;
        dir = state.nextDirection;
    }

    float3 indirectPrimaryEnvNee = primaryEnvNee * exposure;
    if (!all(isfinite(indirectPrimaryEnvNee)) || any(isnan(indirectPrimaryEnvNee))) {
        indirectPrimaryEnvNee = 0.0.xxx;
    }
    float3 indirect = state.radiance * exposure;
    // Global NaN/Inf guard: discard the whole sample so the pixel keeps its currently
    // accumulated value instead of being poisoned by a single bad bounce.
    if (!all(isfinite(indirect)) || any(isnan(indirect))) {
        return;
    }
    indirect = max(indirect, 0.0.xxx);
    // Hard ceiling on top of the per-sample firefly suppression below.
    indirect = clamp(indirect, 0.0.xxx, (kFireflyClampDisplay * 3.0).xxx);
    // Per-sample firefly suppression (averaged across accumulated frames downstream).
    const float avg = (indirect.x + indirect.y + indirect.z) * (1.0 / 3.0);
    if (avg > kFireflyClampDisplay) {
        indirect *= kFireflyClampDisplay / avg;
    }

    hdrImage[px] += float4(indirect + indirectPrimaryEnvNee, 0.0);
}
