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
import bsdf_shared;     // shadowTerminatorFactor / terminatorSmoothPosition
import env_sample;      // evalEnvImportancePdfDir
import path_integrator; // ITracer, SurfaceHit, PathState, EnvNeeSample, evalEnvNEESample, integrateSurface, sampleEnvMapLatLong
import medium_walk;     // shared volumetric random-walk math (runMediumWalk)
import surface;         // shared OpenPBR surface-frame (texture-map) populator

[[vk::push_constant]]
cbuffer GiPC {
    float4x4 view;                  // view matrix (row-major; mul(M, v) convention — matches -matrix-layout-row-major)
    float4x4 prevViewProj;          // GI2 Phase 3: previous-frame view-projection for inline temporal reprojection
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
    uint     a3RegularizationEnabled;      // A3(a): secondary-bounce roughness regularization (Theia-only bias)
    uint     hasGradientVariance;          // A3(b): 1 when t_gradientVariance holds a real A-SVGF guide
    uint     giAdaptiveMaxSamples;         // c1: max GI samples/pixel for variance-guided adaptive sampling (1 = off)
    uint     restirDiEnabled;              // A4: 1 = spatiotemporal reservoir DI; 0 = no ReSTIR direct (bit-identical fallback)
    uint     restirDiSpatial;              // A4: 1 = enable spatial reuse pass (default 0 = initial+temporal only)
    uint     restirHasMotion;              // A4: 1 = t_motionVectors is a real motion image; 0 = dummy (skip reprojection)
    uint     restirPtEnabled;              // GI2: 1 = ReSTIR PT — path integrator owns primary emissive NEE; 0 = legacy DI path
    uint     restirPtPathEnabled;          // GI2 full PT: 1 = multi-bounce path reservoir owns the indirect walk
};

// ── A4: ReSTIR DI reservoir type ──────────────────────────────────────────────
// Spatiotemporal Reservoir Resampling for direct illumination (Bitterli et al.
// 2020). One reservoir per pixel holds the single light sample surviving weighted
// reservoir sampling (WRS) over this frame's initial candidates + reused history.
//
// GPU layout note (the "40-byte pitfall"): the float3 members may be 16-byte
// aligned by the Slang structured-buffer layout, so the true stride is up to 64
// bytes. The C++ side allocates a generous per-element stride (see GiPass) that
// covers whichever layout Slang picks; the CPU never indexes the contents, so
// only the shader's own read/write consistency and a large-enough allocation
// matter.
struct Reservoir {
    uint   lightIdx;       // emissive-triangle index; 0xFFFFFFFF = empty, 0xFFFFFFFE = env (GI2.6)
    float3 samplePoint;    // world-space point on the chosen light
    float3 sampleNormal;   // light surface normal at samplePoint
    float  W;              // unbiased contribution weight = w_sum / (M * p_hat(y))
    float  w_sum;          // running sum of resampling weights
    uint   M;              // number of candidates this reservoir has seen
    uint   candidateHash;  // GI2.5: hash of the initial candidate (duplication map)
};

static const uint kReservoirEmpty = 0xFFFFFFFFu;
static const uint kReservoirEnv   = 0xFFFFFFFEu;  // GI2.6: env-NEE sample sentinel

// GI2.5: Candidate hash for the duplication map. Two reservoirs that sampled the
// same point on the same light produce the same hash — detected during spatial
// reuse to prevent the same initial candidate from being merged twice (which
// inflates nominal M without proportional independent information → small bias).
// Quantized to ~0.5mm so floating-point jitter doesn't create false negatives.
uint restirCandidateHash(uint lightIdx, float3 samplePoint) {
    int3 q = int3(floor(samplePoint * 2048.0));
    uint h = lightIdx;
    h ^= uint(q.x) + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= uint(q.y) + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= uint(q.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
}

Reservoir makeEmptyReservoir() {
    Reservoir r;
    r.lightIdx       = kReservoirEmpty;
    r.samplePoint    = float3(0.0);
    r.sampleNormal   = float3(0.0, 1.0, 0.0);
    r.W              = 0.0;
    r.w_sum          = 0.0;
    r.M              = 0u;
    r.candidateHash  = 0u;
    return r;
}

// Weighted reservoir sampling update. Returns true when the new candidate is kept.
bool reservoirUpdate(inout Reservoir r, uint lightIdx, float3 samplePt, float3 sampleN,
                     float weight, inout RngState rng) {
    r.w_sum += weight;
    r.M += 1u;
    if (weight > 0.0 && randFloat(rng) < (weight / max(r.w_sum, 1.0e-10))) {
        r.lightIdx      = lightIdx;
        r.samplePoint   = samplePt;
        r.sampleNormal  = sampleN;
        r.candidateHash = restirCandidateHash(lightIdx, samplePt); // GI2.5: track chosen candidate
        return true;
    }
    return false;
}

// ── GI2 full PT: multi-bounce path reservoir (GRIS random-replay shift) ──────
// One indirect PATH candidate per pixel, stored as its RNG replay token. A
// candidate's value g is the full multi-bounce indirect estimate of the walk
// (estimateGiSampleCore with primary env NEE excluded): the walk's throughput
// already carries the 1/pdf factors, so g IS the per-seed path-integral
// estimator value and the RIS target is simply p̂ = luminance(g) — no path-pdf
// instrumentation, no shift Jacobian. This is the seed-space formulation of
// Talbot RIS: every candidate (local, temporal, spatial) is uniform in the
// shared Owen-scrambled Sobol seed space, so replaying a stored seed at the
// CURRENT primary surface draws a sample from the same canonical distribution
// as a local candidate (Lin 2022 GRIS §5 random-replay). Unbiased by
// construction; medium walks, Russian roulette and opacity gates ride along
// inside g for free.
struct PathReservoir {
    float3 g;              // selected candidate's indirect estimate, evaluated at THIS pixel
    uint   rngKey;         // replay token: RngState.key of the selected candidate
    uint   rngIndex;       // replay token: RngState.index (dim always restarts at 0)
    float  W;              // unbiased contribution weight = w_sum / (M * p̂(y))
    float  w_sum;          // running sum of resampling weights
    uint   M;              // number of candidates this reservoir has seen
    uint   candidateHash;  // hash of the replay token (duplication map)
};

PathReservoir makeEmptyPathReservoir() {
    PathReservoir r;
    r.g             = 0.0.xxx;
    r.rngKey        = 0u;
    r.rngIndex      = 0u;
    r.W             = 0.0;
    r.w_sum         = 0.0;
    r.M             = 0u;
    r.candidateHash = 0u;
    return r;
}

uint pathCandidateHash(uint key, uint index) {
    uint h = hashCombine(key, index);
    return (h == 0u) ? 1u : h; // 0 is the empty marker
}

// RngState for a local path candidate. Independent walks are decorrelated via the
// Sobol sample INDEX (not the dimension offset): the main sample loop reserves
// indices [0, 8) per frame (maxSamples ≤ 4 for env NEE + margin), so path
// candidates take index base 8 with a stride of 16 (≥ base + max candidates).
RngState pathCandidateSeed(uint2 px, uint candidateIdx) {
    return composeRngSeed(px, frameSampleIndex * 16u + 8u + candidateIdx, 0u, rngBaseSeed);
}

RngState pathReplaySeed(uint key, uint index) {
    RngState s;
    s.key   = key;
    s.index = index;
    s.dim   = 0u;
    return s;
}

// WRS update with a replayable candidate (weight = p̂ = luminance(g); the
// seed-space proposal density q ≡ 1 cancels). Attempts count toward M even when
// the walk degenerated, keeping M consistent with the candidate budget.
void pathReservoirUpdate(inout PathReservoir r, RngState candSeed, float3 g, bool valid,
                         inout RngState rng) {
    const float w = valid ? restirTargetLuma(g) : 0.0;
    r.w_sum += w;
    r.M += 1u;
    if (w > 0.0 && randFloat(rng) < (w / max(r.w_sum, 1.0e-10))) {
        r.rngKey        = candSeed.key;
        r.rngIndex      = candSeed.index;
        r.g             = g;
        r.candidateHash = pathCandidateHash(candSeed.key, candSeed.index);
    }
}

// Merge a source reservoir (temporal/spatial reuse), Bitterli 2020 Alg. 4 in seed
// space. gReplayed is the source's selected candidate REPLAYED at the current
// primary surface; its luminance is p̂(y_src) evaluated at the current pixel.
// On selection the reservoir stores the LOCAL replay (never the source's g,
// which was evaluated at a different surface).
void pathReservoirCombine(inout PathReservoir r_dst, PathReservoir r_src,
                          float3 gReplayed, bool replayValid, inout RngState rng) {
    const float pHat = replayValid ? restirTargetLuma(gReplayed) : 0.0;
    const float w = pHat * r_src.W * float(r_src.M);
    r_dst.M += r_src.M;
    r_dst.w_sum += w;
    if (w > 0.0 && randFloat(rng) < (w / max(r_dst.w_sum, 1.0e-10))) {
        r_dst.rngKey        = r_src.rngKey;
        r_dst.rngIndex      = r_src.rngIndex;
        r_dst.g             = gReplayed;
        r_dst.candidateHash = r_src.candidateHash;
    }
}

// Rec.709 luminance for the ReSTIR target function (Bitterli 2020 §5.2). Any
// positive scalar target keeps the estimator unbiased; the choice only affects
// variance, so a fixed luminance is used regardless of working colour space.
float restirTargetLuma(float3 c) {
    return max(dot(c, float3(0.2126, 0.7152, 0.0722)), 0.0);
}

// Unshadowed target PDF p_hat(y) = |L_e * bsdf * cos| / dist^2 (scalar, unnormalised).
float targetPdf(float3 Le, float3 bsdfTimesCos, float dist2) {
    return restirTargetLuma(Le * bsdfTimesCos) / max(dist2, 1.0e-6);
}

// GI2.6: Target PDF for env samples — no dist^2 (env is at infinity).
float targetPdfEnv(float3 Le, float3 bsdfTimesCos) {
    return restirTargetLuma(Le * bsdfTimesCos);
}

[[vk::binding(0, 0)]]  RaytracingAccelerationStructure       t_tlas;
[[vk::binding(1, 0)]]  RWTexture2D<float4>                   hdrImage;
[[vk::binding(2, 0)]]  Texture2D<float4>                     t_giBuffer;   // xyz = worldPos, w = asfloat(instanceIdx+1)
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
[[vk::binding(13, 0)]] Sampler2D                             t_bindless[];
[[vk::binding(14, 0)]] StructuredBuffer<float>              emissiveCdf;        // power-proportional NEE selection CDF
[[vk::binding(15, 0)]] Texture2D<float2>                    t_gradientVariance; // R=gradient, G=variance (from A-SVGF); 1×1 dummy when unavailable

// A4: ReSTIR DI reservoir ping-pong buffers + motion vectors.
// Layout: one Reservoir per pixel, indexed [y * screenWidth + x].
[[vk::binding(16, 0)]] RWStructuredBuffer<Reservoir>       t_reservoirsCur;    // current-frame write (and spatial read)
[[vk::binding(17, 0)]] StructuredBuffer<Reservoir>         t_reservoirsPrev;   // previous-frame read (temporal reuse)
[[vk::binding(18, 0)]] Texture2D<float2>                   t_motionVectors;    // screen-space (dx,dy) in pixels; 1×1 dummy → zero motion
// Per-instance object→world transforms (one float4x4 per instance, row-major).
[[vk::binding(19, 0)]] StructuredBuffer<float4x4>          instanceTransforms;

// GI2 full PT: path reservoir ping-pong buffers (multi-bounce indirect reuse).
// Layout: one PathReservoir per pixel, indexed [y * screenWidth + x].
[[vk::binding(20, 0)]] RWStructuredBuffer<PathReservoir>   t_pathReservoirsCur;   // current-frame write
[[vk::binding(21, 0)]] StructuredBuffer<PathReservoir>     t_pathReservoirsPrev;  // previous-frame read

/// Inverse of a row-major 3×3 lives in common.slang (shared with forward_render.mesh).

// Per-sample firefly clamp on the indirect contribution (display-referred, post-exposure),
// matching the forward pass kFireflyClampDisplay threshold.
static const float kFireflyClampDisplay = 100.0;

// c1: variance→extra-samples mapping. Local A-SVGF variance is scaled by this factor and
// saturated to [0,1]; the result linearly interpolates the per-pixel GI sample count between
// 1 and giAdaptiveMaxSamples. Matches the magnitude the A3(b) clamp uses (variance * 20).
static const float kAdaptiveVarianceScale = 20.0;

struct GiHit {
    bool   hit;
    uint   materialIdx;
    float3 worldPos;
    float3 shadingNormal;
    float3 geoNormal; // RAW outward-winding geometric normal (NOT ray-facing): consumers flip by
                      // wo (makeSurfaceHit, mirroring Hyperion's shadeSurface) or use it as the
                      // true interface outward normal (runMediumWalk, backface/side detection).
    float2 uv;
    float3 tangent;
    float  dist;      // committed ray T (used for volumetric medium boundary transmittance)
};

// maxComp lives in math.slang (imported).

/// Closest-hit query against the scene TLAS, reconstructing position/material/normal.
/// `maxDist` bounds the ray (kInf for ordinary bounces; a sampled free-flight distance for
/// volumetric medium segments — a miss within maxDist is an in-medium scattering event).
/// `tMin` is the ray self-intersection guard (kRayEpsilon for surface bounces; kMediumEpsilon
/// inside a medium, matching Hyperion — dense media have free-flight distances below kRayEpsilon).
GiHit traceClosestHit(float3 origin, float3 dir, float maxDist, float tMin) {
    GiHit hit;
    hit.hit = false;
    hit.materialIdx = 0u;
    hit.worldPos = 0.0.xxx;
    hit.shadingNormal = float3(0.0, 1.0, 0.0);
    hit.geoNormal = float3(0.0, 1.0, 0.0);

    const float3 rayDir = normalize(dir);
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = rayDir;
    ray.TMin = tMin;
    ray.TMax = maxDist;

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(t_tlas, RAY_FLAG_NONE, kInstanceMaskAll, ray);
    // Cut-out geometry is non-opaque, so its hits arrive as candidates. Commit them
    // all: for camera/indirect rays the OpenPBR opacity mix is resolved by the shared
    // estimator's stochastic pass-through gate at the surface, exactly as it is for
    // opaque geometry. (An opacity micromap still culls provably-transparent
    // microtriangles before they ever become candidates — the same answer, less work.)
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return hit;
    }

    const uint instIdx = q.CommittedInstanceID();
    const GpuInstance inst = instances[instIdx];
    const uint tri = q.CommittedPrimitiveIndex();
    const uint base = inst.indexOffset + tri * 3u;
    if (base + 2u >= inst.indexOffset + inst.indexCount) {
        return hit;
    }

    const uint i0 = indices[base + 0u];
    const uint i1 = indices[base + 1u];
    const uint i2 = indices[base + 2u];
    // Vertices live in object space; the instance transform places them.
    const float4x4 o2w = instanceTransforms[instIdx];
    const float3x3 normalMat = transpose(inverse3x3((float3x3)o2w));
    const float3 p0 = mul(o2w, float4(vertices[i0].position, 1.0)).xyz;
    const float3 p1 = mul(o2w, float4(vertices[i1].position, 1.0)).xyz;
    const float3 p2 = mul(o2w, float4(vertices[i2].position, 1.0)).xyz;
    const float3 n0 = normalize(mul(normalMat, vertices[i0].normal));
    const float3 n1 = normalize(mul(normalMat, vertices[i1].normal));
    const float3 n2 = normalize(mul(normalMat, vertices[i2].normal));
    const float2 uv0 = vertices[i0].uv;
    const float2 uv1 = vertices[i1].uv;
    const float2 uv2 = vertices[i2].uv;

    const float2 b = q.CommittedTriangleBarycentrics();
    const float3 bary = float3(1.0 - b.x - b.y, b.x, b.y);
    // RAW outward-winding geometric normal (Hyperion's HitInfo contract): the side
    // information is carried by the sign of dot(geoNormal, wo) at the consumers —
    // makeSurfaceHit flips a local copy to face wo and derives SurfaceHit.backface
    // (C7 dielectric exit branch), runMediumWalk needs the TRUE outward normal for the
    // exit refraction/TIR and the inside/outside ray offsets. Pre-flipping it here
    // (ray-facing) would erase the side bit: backface would always read false and the
    // medium walk would refract against a wrong-signed normal (broken exits, spurious
    // re-entry, repeated absorption).
    hit.worldPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    TriangleGeometry g = reconstructTriangleGeometry(p0, p1, p2, n0, n1, n2, uv0, uv1, uv2, bary);
    hit.uv = g.uv;
    hit.tangent = g.tangent;
    hit.shadingNormal = g.shadingNormal;
    hit.geoNormal = g.geoNormal;
    hit.materialIdx = inst.materialIndex;
    hit.dist = q.CommittedRayT();
    hit.hit = true;
    return hit;
}

/// G9/G10: run the volumetric random walk inside the compute GI loop. All walk math
/// (free-flight, scatter, boundary transmittance, Fresnel exit/TIR) lives in Harmonia's shared
/// `medium_walk` module — identical 1:1 with Hyperion's raygen/miss/closesthit, so it can never
/// drift. Consumes its own bounded budget (does NOT count against the surface bounce budget). On
/// exit, updates origin/dir to the refracted ray leaving the medium and clears insideMedium; on
/// absorption terminates the path.
void runMediumWalk(inout PathState state, inout float3 origin, inout float3 dir) {
    const uint kMaxMediumWalk = 256u;
    const bool pureAbsorption = mediumIsPureAbsorber(state.mediumAlbedo);
    for (uint i = 0u; i < kMaxMediumWalk && state.insideMedium && !state.terminated; ++i) {
        float freeDist = mediumSampleFreeFlight(state.mediumSigmaT, state.mediumAlbedo, state.seed);

        GiHit h = traceClosestHit(origin, dir, freeDist, kMediumEpsilon);
        if (!h.hit) {
            if (pureAbsorption) {
                // No boundary within kInf: the absorber is unbounded along the ray (escaped via
                // non-watertight geometry) — exact transmittance exp(-σ_t·∞) = 0.
                state.terminated = true;
                return;
            }
            float3 newOrigin, newDir;
            mediumScatterEvent(state.throughput, state.seed, state.terminated,
                               state.mediumSigmaT, state.mediumAlbedo, state.mediumG,
                               origin, dir, freeDist, newOrigin, newDir);
            origin = newOrigin;
            dir = newDir;
            continue;
        }

        // Boundary crossing at h.dist — transmittance, then Fresnel exit/TIR.
        mediumBoundaryTransmittance(state.throughput, state.mediumSigmaT, pureAbsorption, h.dist);
        float3 outOrigin, outDir;
        bool exited = mediumResolveExit(state.seed, state.mediumEta, dir, h.geoNormal, h.worldPos,
                                        outOrigin, outDir);
        dir = outDir;
        origin = outOrigin;
        if (exited) {
            state.insideMedium = false;
            return;
        }
    }
    if (state.insideMedium) { state.terminated = true; }
}

/// Hard cap on OpenPBR opacity pass-throughs (null interactions) a single path may
/// take. They consume no bounce budget by design, so this is what guarantees
/// termination — the same role kMaxMediumWalk plays for volumetric segments.
static const uint kMaxNullInteractions = 256u;

/// Bindless-texture fetcher for the shared surface-frame populator. Compute shaders have no
/// screen-space quad, so implicit-LOD .Sample() (which needs SPV_KHR_compute_shader_derivatives)
/// is both unsupported here and meaningless for a secondary ray hit — mip 0 only.
struct TheiaTextureSampler : ISurfaceTextureSampler {
    float4 sampleTex(uint idx, float2 uv) {
        return t_bindless[NonUniformResourceIndex(idx)].SampleLevel(uv, 0.0);
    }
};

/// Interpolated UV of a triangle, from its BLAS primitive index within an instance.
/// Only the UV is reconstructed — the opacity test needs nothing else.
float2 triangleUv(uint instIdx, uint tri, float2 bary) {
    const GpuInstance inst = instances[instIdx];
    const uint base = inst.indexOffset + tri * 3u;
    if (base + 2u >= inst.indexOffset + inst.indexCount) {
        return 0.0.xx;
    }
    const float b1 = bary.x;
    const float b2 = bary.y;
    const float b0 = 1.0 - b1 - b2;
    return vertices[indices[base + 0u]].uv * b0
         + vertices[indices[base + 1u]].uv * b1
         + vertices[indices[base + 2u]].uv * b2;
}

/// OpenPBR presence weight α at a triangle hit — the RayQuery counterpart of
/// Hyperion's shadow any-hit (anyhit.slang). Same shared `evalGeometryOpacity`,
/// so the cutout cannot drift between the two renderers.
float triangleOpacity(uint instIdx, uint tri, float2 bary) {
    TheiaTextureSampler tex;
    return evalGeometryOpacity(materials[instances[instIdx].materialIndex], tex,
                               triangleUv(instIdx, tri, bary));
}

/// Inline-ray-query shadow tracer satisfying the shared estimator's ITracer interface.
///
/// Cut-out geometry is built non-opaque (Harmonia `MeshOpacity::alphaTested`), so its
/// hits arrive as candidates instead of being committed by traversal. That is exactly
/// where the OpenPBR transmittance ∏(1-α) is accumulated — mirroring Hyperion's shadow
/// any-hit — and where an opacity micromap saves the work by resolving uniform
/// microtriangles before a candidate is ever produced. Fully-present surfaces commit,
/// which under ACCEPT_FIRST_HIT ends the traversal exactly as opaque geometry does.
struct TheiaTracer : ITracer {
    float traceTransmittance(float3 origin, float3 direction, float tMax) {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = 0.0;
        ray.TMax = tMax;
        float transmittance = 1.0;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(t_tlas, RAY_FLAG_NONE, kInstanceMaskAll, ray);
        while (q.Proceed()) {
            if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
                continue;
            }
            const float alpha = triangleOpacity(q.CandidateInstanceID(),
                                                q.CandidatePrimitiveIndex(),
                                                q.CandidateTriangleBarycentrics());
            if (alpha >= 1.0) {
                q.CommitNonOpaqueTriangleHit();
            } else {
                transmittance *= (1.0 - alpha);
            }
        }
        return (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : saturate(transmittance);
    }
};

/// Populate a SurfaceHit from a traced hit via the shared OpenPBR texture-map populator
/// (identical 1:1 with Hyperion's shadeSurface, so the frame/C7 logic cannot drift).
SurfaceHit makeSurfaceHit(GiHit h, float3 wo) {
    GpuMaterial mat = materials[h.materialIdx];
    TheiaTextureSampler tex;
    SurfaceFrame f = applyOpenPbrTextureMaps(mat, tex, wo, h.shadingNormal, h.geoNormal, h.tangent, h.uv);

    SurfaceHit surf;
    surf.position  = h.worldPos;
    surf.shadowPos = h.worldPos; // == position (Chiang factor-only terminator scheme; see bsdf_shared.slang)
    surf.normal    = f.normal;
    surf.geoNormal = f.geoNormal;
    surf.backface  = f.backface;
    surf.T = f.T; surf.B = f.B;
    surf.coatN = f.coatN; surf.coatT = f.coatT; surf.coatB = f.coatB;
    surf.mat = mat;
    surf.wo  = wo;
    return surf;
}

// ── A4: ReSTIR DI evaluation helpers ──────────────────────────────────────────

/// Evaluation of a reservoir's chosen emissive-triangle sample AT a given surface:
/// re-derives L_e, bsdf·cos, squared distance and the incident direction from the
/// stored world-space sample point/normal. Used both for the target PDF (p_hat) and
/// for the final shaded contribution. `valid=false` when the sample is behind the
/// surface or the emitter's back face (contributes nothing).
struct RestirSampleEval {
    float3 Le;        // emitter radiance
    float3 bsdfCos;   // bsdf(wo,wi) * cos(theta_surface)
    float  dist2;     // |samplePoint - surf.position|^2 (kInf for env)
    float  dist;      // sqrt(dist2)
    float3 wi;        // normalized surface->light direction
    float  p_hat;     // target PDF at this surface (kind-aware: targetPdf or targetPdfEnv)
    bool   valid;
};

RestirSampleEval evalReservoirSample(SurfaceHit surf, uint lightIdx,
                                     float3 samplePoint, float3 sampleNormal) {
    RestirSampleEval e;
    e.Le = 0.0.xxx; e.bsdfCos = 0.0.xxx; e.dist2 = 0.0; e.dist = 0.0;
    e.wi = 0.0.xxx; e.p_hat = 0.0; e.valid = false;

    // GI2.6: env-NEE sample — samplePoint.xyz holds the env direction (unit vector).
    // MIS-aware: the BSDF continuation in estimateGiSample also reaches env when its
    // bounce ray escapes to infinity. Balance heuristic (Veach 1997 §9.2) prevents
    // double-counting — same MIS the shared evalEnvNEESample applies.
    if (lightIdx == kReservoirEnv) {
        float3 wi = normalize(samplePoint);
        float  cosI = dot(wi, surf.normal);
        if (cosI <= 0.0) return e;
        float3 Le = sampleEnvMapLatLong(t_envMap, s_env, wi) * envLuminanceScale;
        float3 bsdf = evalBSDF(surf.mat, surf.wo, wi,
            surf.normal, surf.T, surf.B, surf.coatN, surf.coatT, surf.coatB, surf.backface);
        // MIS: envPdf from the importance CDF; bsdfPdf from the BSDF model.
        float envPdf = evalEnvImportancePdfDir(envMarginalCdf, envConditionalCdf,
                                                envImportanceWidth, envImportanceHeight, wi);
        float bsdfPdf = pdfBSDF(surf.mat, surf.wo, wi,
            surf.normal, surf.T, surf.B, surf.coatN, surf.coatT, surf.coatB, surf.backface);
        float misW = envPdf / max(envPdf + bsdfPdf, 1.0e-6);
        e.Le = Le;
        e.bsdfCos = bsdf * cosI;
        e.dist2 = kInf;
        e.dist = kInf;
        e.wi = wi;
        e.p_hat = targetPdfEnv(Le, e.bsdfCos) * misW;  // MIS-weighted target
        e.valid = (envPdf > 0.0);
        return e;
    }

    if (lightIdx >= emissiveTriangleCount) {
        return e;
    }
    GpuEmissiveTriangle tri = emissiveTriangles[lightIdx];
    float3 Le = float3(tri.edge1_emitR.w, tri.edge2_emitG.w, tri.normal_emitB.w);

    float3 toLight = samplePoint - surf.position;
    float  dist2   = dot(toLight, toLight);
    float  dist    = sqrt(max(dist2, 1.0e-12));
    float3 wi      = toLight / max(dist, 1.0e-6);
    float  cosI    = dot(wi, surf.normal);
    float  cosL    = dot(-wi, sampleNormal);
    if (cosI <= 0.0 || cosL <= 0.0) {
        return e;
    }
    float3 bsdf = evalBSDF(surf.mat, surf.wo, wi,
        surf.normal, surf.T, surf.B, surf.coatN, surf.coatT, surf.coatB, surf.backface);
    e.Le      = Le;
    e.bsdfCos = bsdf * cosI;
    e.dist2   = dist2;
    e.dist    = dist;
    e.wi      = wi;
    e.p_hat   = targetPdf(Le, e.bsdfCos, dist2);
    e.valid   = true;
    return e;
}

/// Draw one initial light candidate via the power-proportional emissive CDF
/// (the A0b selection), sample a point on the triangle (Shirley sqrt-folding), and
/// return its solid-angle source PDF. Mirrors evalEmissiveTriangleNEE's sampling so
/// the RIS source distribution matches the legacy single-sample estimator.
struct RestirCandidate {
    uint   lightIdx;
    float3 samplePoint;
    float3 sampleNormal;
    float  solidPdf;    // solid-angle PDF of drawing this sample (p_source)
    bool   valid;
};

RestirCandidate sampleEmissiveCandidate(SurfaceHit surf, inout RngState seed) {
    RestirCandidate c;
    c.lightIdx = kReservoirEmpty; c.samplePoint = 0.0.xxx;
    c.sampleNormal = float3(0.0, 1.0, 0.0); c.solidPdf = 0.0; c.valid = false;
    if (emissiveTriangleCount == 0u) {
        return c;
    }

    // Power-proportional triangle selection (binary search on the prefix-sum CDF).
    float u  = randFloat(seed);
    uint  lo = 0u;
    uint  hi = emissiveTriangleCount - 1u;
    for (uint step = 0u; step < 20u; ++step) {
        if (lo >= hi) { break; }
        uint mid = (lo + hi) / 2u;
        if (emissiveCdf[mid] < u) { lo = mid + 1u; } else { hi = mid; }
    }
    uint triIdx = min(lo, emissiveTriangleCount - 1u);
    float triPdf = (triIdx == 0u) ? emissiveCdf[0] : (emissiveCdf[triIdx] - emissiveCdf[triIdx - 1u]);
    if (triPdf <= 0.0) { triPdf = 1.0 / float(emissiveTriangleCount); }

    GpuEmissiveTriangle tri = emissiveTriangles[triIdx];

    // Tagged-union dispatch: analytic sphere emitters are sampled by uniform solid-angle
    // (cone) sampling; the stored samplePoint/sampleNormal let evalReservoirSample (which
    // only reads the emission w-channels + the stored point/normal) re-evaluate them at any
    // surface unchanged. Mirrors evalEmissiveSphereNEE in the shared path_integrator.
    if (tri.kind_pad.x == kEmissiveSphere) {
        float3 center = tri.v0_area.xyz;
        float  radius = tri.v0_area.w;
        float3 toC = center - surf.position;
        float  d   = length(toC);
        if (d <= radius) { return c; }
        float3 wc     = toC / d;
        float  sinMax = min(radius / d, 1.0);
        float  cosMax = safeSqrt(1.0 - sinMax * sinMax);
        // (1 - cosθMax) via the exact identity sin²/(1+cos) — the direct subtraction
        // underflows to 0 for radius/d ≲ 2.4e-4 in fp32 (pdf → +inf, sphere drops out).
        float  oneMinusCosMax = (sinMax * sinMax) / (1.0 + cosMax);
        if (oneMinusCosMax <= 0.0) { return c; }
        float u1               = randFloat(seed);
        float u2               = randFloat(seed);
        float oneMinusCosTheta = u1 * oneMinusCosMax;
        float cosTheta         = 1.0 - oneMinusCosTheta;
        float sinTheta         = safeSqrt(oneMinusCosTheta * (2.0 - oneMinusCosTheta));
        float phi              = 2.0 * kPi * u2;
        float3x3 frame = buildFrameFromZ(wc); // rows: T, B, wc
        float3 wi      = normalize(frame[0] * (sinTheta * cos(phi))
                                + frame[1] * (sinTheta * sin(phi))
                                + wc * cosTheta);
        // Stable discriminant r² - |oc⊥|²: (oc·wi)² - (oc·oc - r²) cancels to noise for a
        // small distant sphere. Matches evalEmissiveSphereNEE bit for bit.
        float3 oc   = surf.position - center;
        float  hb   = dot(oc, wi);
        float3 ocT  = oc - hb * wi;
        float  disc = radius * radius - dot(ocT, ocT);
        if (disc <= 0.0) { return c; }
        float t = -hb - sqrt(disc);
        if (t <= 0.0) { return c; }
        float3 hp     = surf.position + wi * t;
        float3 nLight = normalize(hp - center);
        if (dot(-wi, nLight) <= 0.0 || dot(wi, surf.normal) <= 0.0) { return c; }
        // Source pdf = P(select this sphere) × uniform-cone solid-angle pdf.
        float conePdf = 1.0 / (2.0 * kPi * oneMinusCosMax);
        c.lightIdx    = triIdx;
        c.samplePoint  = hp;
        c.sampleNormal = nLight;
        c.solidPdf     = triPdf * conePdf;
        c.valid        = c.solidPdf > 0.0;
        return c;
    }

    float3 v0    = tri.v0_area.xyz;
    float  area  = tri.v0_area.w;
    float3 edge1 = tri.edge1_emitR.xyz;
    float3 edge2 = tri.edge2_emitG.xyz;
    float3 lNorm = tri.normal_emitB.xyz;

    float  sqrtU = sqrt(randFloat(seed));
    float  xi    = randFloat(seed);
    float3 bary  = float3(1.0 - sqrtU, sqrtU * (1.0 - xi), sqrtU * xi);
    float3 lPos  = v0 + edge1 * bary.y + edge2 * bary.z;

    float3 toLight = lPos - surf.position;
    float  dist    = length(toLight);
    float3 wi      = toLight / max(dist, 1.0e-6);
    float  cosLight = dot(-wi, lNorm);
    if (cosLight <= 0.0 || area <= 0.0) {
        return c;
    }
    // Area PDF (power-weighted) → solid-angle PDF.
    float areaPdf  = triPdf / area;
    float solidPdf = areaPdf * dist * dist / cosLight;

    c.lightIdx     = triIdx;
    c.samplePoint  = lPos;
    c.sampleNormal = lNorm;
    c.solidPdf     = solidPdf;
    c.valid        = solidPdf > 0.0;
    return c;
}

/// GI2.6: Draw one initial env-NEE candidate via the 2D-separable importance CDF.
/// Stores the sampled direction in samplePoint (overloaded — env has no world-space
/// position). The source PDF is the env importance PDF.
RestirCandidate sampleEnvCandidate(SurfaceHit surf, inout RngState seed) {
    RestirCandidate c;
    c.lightIdx = kReservoirEmpty; c.samplePoint = 0.0.xxx;
    c.sampleNormal = float3(0.0, 1.0, 0.0); c.solidPdf = 0.0; c.valid = false;

    const bool envImportance = (hasEnvMap != 0u) && envImportanceWidth > 0u && envImportanceHeight > 0u;
    if (!envImportance) return c;

    float envPdf;
    float2 xi = float2(randFloat(seed), randFloat(seed));
    float3 dir = sampleEnvImportanceDir(envMarginalCdf, envConditionalCdf,
                                         envImportanceWidth, envImportanceHeight, xi, envPdf);
    float cosI = dot(dir, surf.normal);
    if (cosI <= 0.0 || envPdf <= 0.0) return c;

    c.lightIdx     = kReservoirEnv;
    c.samplePoint  = dir;        // env direction (not a world-space point)
    c.sampleNormal = float3(0.0, 1.0, 0.0); // unused for env
    c.solidPdf     = envPdf;
    c.valid        = true;
    return c;
}

/// Combine a source reservoir into the destination (Bitterli 2020, Alg. 4). The
/// source's chosen sample is re-weighted by its target PDF evaluated AT the current
/// surface (passed in as p_hat_at_cur — kind-aware) times its stored W and M.
void combineTwoReservoirs(inout Reservoir r_dst, Reservoir r_src,
                          float p_hat_at_cur,
                          inout RngState rng) {
    float w = p_hat_at_cur * r_src.W * float(r_src.M);
    r_dst.M += r_src.M;
    r_dst.w_sum += w;
    if (w > 0.0 && randFloat(rng) < w / max(r_dst.w_sum, 1.0e-10)) {
        r_dst.lightIdx      = r_src.lightIdx;
        r_dst.samplePoint   = r_src.samplePoint;
        r_dst.sampleNormal  = r_src.sampleNormal;
        r_dst.candidateHash = r_src.candidateHash; // GI2.5: propagate chosen candidate's hash
    }
}

/// A4/GI2 ReSTIR DI direct-lighting estimate for the primary surface.
/// Runs initial-candidate RIS via the power CDF + env importance CDF, temporal reuse,
/// spatial reuse, computes the unbiased weight W, traces ONE shadow ray, and returns
/// the throughput-free direct radiance (pre-exposure) as .xyz plus the reservoir M
/// (effective sample count) as .w — used by the GI2.7 M-aware denoiser via the HDR
/// alpha channel. The final reservoir is written to t_reservoirsCur[pixelIdx].
float4 restirDiDirect(SurfaceHit primary, uint2 px, inout RngState rng) {
    const uint kRestirInitialCandidates = 8u;
    const uint kRestirMCap = 20u;
    const uint pixelIdx = px.y * screenWidth + px.x;

    Reservoir r = makeEmptyReservoir();

    // ── Initial candidates (RIS): scene-type-dependent. Emissive-lit scenes
    //    (cornell etc.): all emissive candidates — env NEE stays in estimateGiSample
    //    (4 independent samples with own shadow rays, better for occluded env).
    //    IBL-only scenes (dragon_teapot etc.): all env candidates — the reservoir
    //    gives env spatiotemporal reuse for the first time. The gate is
    //    emissiveTriangleCount (scene property, not a knob).
    const bool sceneHasEmissive = (emissiveTriangleCount > 0u);
    for (uint i = 0u; i < kRestirInitialCandidates; ++i) {
        RestirCandidate cand;
        if (sceneHasEmissive) {
            cand = sampleEmissiveCandidate(primary, rng);
        } else {
            cand = sampleEnvCandidate(primary, rng);
        }
        if (!cand.valid) {
            r.M += 1u; // count the attempt to keep M consistent with candidate budget
            continue;
        }
        RestirSampleEval e = evalReservoirSample(primary, cand.lightIdx, cand.samplePoint, cand.sampleNormal);
        float weight = 0.0;
        if (e.valid) {
            weight = e.p_hat / max(cand.solidPdf, 1.0e-10);
        }
        reservoirUpdate(r, cand.lightIdx, cand.samplePoint, cand.sampleNormal, weight, rng);
    }

    // ── Temporal reuse: reproject via prevViewProj, merge the previous reservoir. ──
    {
        // GI2 Phase 3: inline temporal reprojection — compute where THIS pixel's world
        // position was on the previous frame's screen, directly from prevViewProj.
        // Replaces the binding-18 motion-vector load (which was always nulled because
        // MotionVectorPass runs AFTER GiPass in the pipeline). The math is identical
        // to MotionVectorPass's, but evaluated only where temporal reuse needs it.
        float4 prevClip = mul(prevViewProj, float4(primary.position, 1.0));
        // Points behind the previous camera (w <= 0) have no valid reprojection — the
        // perspective divide would mirror them to a meaningless screen position. Skip
        // temporal reuse for those (standard practice; was previously abs(w), which
        // mirrored them into plausible-but-wrong pixels).
        if (prevClip.w > 1.0e-6) {
        float2 prevNdc  = prevClip.xy / prevClip.w;
        float2 prevPixel = (prevNdc * 0.5 + 0.5) * float2(screenWidth, screenHeight);
        // B2 fix: use floor (not round) for pixel index. The NDC→pixel mapping produces
        // pixel-center coordinates (px + 0.5); round()'s halfway direction is
        // implementation-defined in SPIR-V (HLSL semantics: away from zero), so it picked
        // a wrong pixel for at least half of all integer positions on any vendor.
        // floor() recovers the integer pixel index unambiguously. v0.7.0 used exact
        // integer subtraction (px - 0 = px) via the nulled motion vectors; this floor()
        // path reproduces that exactly.
        int2 prevCoord = int2(int(floor(prevPixel.x)), int(floor(prevPixel.y)));
        if (prevCoord.x >= 0 && prevCoord.y >= 0 &&
            prevCoord.x < int(screenWidth) && prevCoord.y < int(screenHeight)) {
            uint prevIdx = uint(prevCoord.y) * screenWidth + uint(prevCoord.x);
            Reservoir prev = t_reservoirsPrev[prevIdx];
            if (prev.lightIdx != kReservoirEmpty && prev.M > 0u) {
                prev.M = min(prev.M, kRestirMCap); // cap history length to bound temporal bias
                RestirSampleEval pe = evalReservoirSample(primary, prev.lightIdx, prev.samplePoint, prev.sampleNormal);
                if (pe.valid) {
                    combineTwoReservoirs(r, prev, pe.p_hat, rng);
                } else {
                    r.M += prev.M; // account for history even if its sample is now invalid
                }
            }
        }
        }
    }

    // NOTE: M is deliberately NOT re-capped here. The temporal history is bounded by
    // capping the PREVIOUS reservoir's M (above) before it contributes to w_sum; re-capping
    // the combined M while w_sum reflects the full candidate count would make W diverge over
    // accumulated frames. The stored M (kRestirInitialCandidates + capped history) stays
    // bounded on its own and is re-capped when reused next frame.

    // ── Spatial reuse (Bitterli 2020 Alg. 5; GI2 Phase 2). Merge neighbour reservoirs
    //    from the PREV frame's buffer (same binding as temporal — avoids the read-write
    //    hazard a same-frame spatial pass would create in this single dispatch).
    //    Disocclusion check via G-buffer: world-position distance (view-scaled) and
    //    view-space normal similarity. Compatible neighbours contribute their full
    //    history M (capped); incompatible ones are skipped. The shift is the same
    //    "re-evaluate target PDF at current surface" used for temporal — valid because
    //    the reservoir carries an explicit world-space sample point.
    if (restirDiSpatial != 0u) {
        const uint  kSpatialNeighbors = 5u;
        const float kSpatialRadius    = 32.0;
        const float kPosThresholdRel  = 0.01;   // 1% of view distance
        const float kNormalThreshold  = 0.98;   // ~11.5° half-angle

        // Current pixel's G-buffer edge-stop signals.
        const float3 curWorldPos = t_giBuffer.Load(int3(int(px.x), int(px.y), 0)).xyz;
        const float3 curViewN    = t_gbuffer.Load(int3(int(px.x), int(px.y), 0)).xyz * 2.0 - 1.0;
        const float  viewDist    = length(curWorldPos - cameraPos.xyz);
        const float  posThresh   = max(viewDist * kPosThresholdRel, 1.0e-3);

        // GI2.5: duplication map — track hashes of candidates already merged this
        // spatial pass. Prevents the same initial candidate from being counted twice
        // (e.g. neighbour A and neighbour B both merged neighbour C's sample last
        // frame → merging both A and B into this pixel double-counts C's candidate).
        uint seenHashes[6]; // 1 (own) + up to 5 neighbours
        seenHashes[0] = r.candidateHash;
        uint seenCount = 1u;

        for (uint n = 0u; n < kSpatialNeighbors; ++n) {
            // Random offset in [-radius, +radius]^2 (Bitterli 2020 §5.2 random neighbour).
            const float2 off = (float2(randFloat(rng), randFloat(rng)) * 2.0 - 1.0) * kSpatialRadius;
            const int2 nCoord = int2(int(px.x) + int(round(off.x)), int(px.y) + int(round(off.y)));
            if (nCoord.x < 0 || nCoord.y < 0 ||
                nCoord.x >= int(screenWidth) || nCoord.y >= int(screenHeight)) {
                continue;
            }
            // Disocclusion: skip neighbours on a different surface.
            const float3 nWorldPos = t_giBuffer.Load(int3(nCoord.x, nCoord.y, 0)).xyz;
            const float3 nViewN    = t_gbuffer.Load(int3(nCoord.x, nCoord.y, 0)).xyz * 2.0 - 1.0;
            if (length(curWorldPos - nWorldPos) > posThresh) continue;
            if (dot(curViewN, nViewN) < kNormalThreshold) continue;

            const uint nIdx = uint(nCoord.y) * screenWidth + uint(nCoord.x);
            Reservoir neighbor = t_reservoirsPrev[nIdx];
            if (neighbor.lightIdx != kReservoirEmpty && neighbor.M > 0u) {
                // GI2.5: duplication check — skip if this neighbour's candidate was
                // already merged from a different neighbour (same hash).
                bool duplicate = false;
                for (uint h = 0u; h < seenCount; ++h) {
                    if (seenHashes[h] == neighbor.candidateHash) { duplicate = true; break; }
                }
                if (duplicate) continue;
                if (seenCount < 6u) seenHashes[seenCount++] = neighbor.candidateHash;

                neighbor.M = min(neighbor.M, kRestirMCap);
                RestirSampleEval ne = evalReservoirSample(primary, neighbor.lightIdx,
                                                          neighbor.samplePoint, neighbor.sampleNormal);
                if (ne.valid) {
                    combineTwoReservoirs(r, neighbor, ne.p_hat, rng);
                } else {
                    r.M += neighbor.M; // account for history even if its sample is now invalid
                }
            }
        }
    }

    // ── Final unbiased weight W = w_sum / (M * p_hat(y_chosen)). ──
    float3 contribution = 0.0.xxx;
    RestirSampleEval fe = evalReservoirSample(primary, r.lightIdx, r.samplePoint, r.sampleNormal);
    if (r.lightIdx != kReservoirEmpty && fe.valid && r.M > 0u) {
        r.W = r.w_sum / max(float(r.M) * fe.p_hat, 1.0e-10);

        // Single shadow ray (unbiased single-shadow-ray scheme). For env samples
        // (lightIdx == kReservoirEnv) the ray goes to infinity (kInf); for emissive
        // triangles it goes to the stored sample point.
        TheiaTracer tracer;
        float gTerm;
        float fTerm = shadowTerminatorFactor(primary.normal, primary.geoNormal, fe.wi, gTerm);
        if (fTerm > 0.0) {
            float3 shadowOrigin = primary.position + primary.geoNormal * kRayEpsilon;
            float shadowTMax;
            float misW = 1.0; // MIS weight (1.0 for emissive — no competing BSDF technique at primary)
            if (r.lightIdx == kReservoirEnv) {
                shadowTMax = kInf;  // env is at infinity
                // GI2.6: balance heuristic MIS with the BSDF continuation path (which also
                // reaches env when its bounce ray escapes). The misW was already folded into
                // p_hat during evalReservoirSample; apply it again here as a contribution
                // multiplier (the luminance in p_hat cancels in W, leaving the RGB misW
                // factor to be applied explicitly — standard ReSTIR DI §5.2 formulation).
                float envPdf = evalEnvImportancePdfDir(envMarginalCdf, envConditionalCdf,
                                                        envImportanceWidth, envImportanceHeight, fe.wi);
                float bsdfPdf = pdfBSDF(primary.mat, primary.wo, fe.wi,
                    primary.normal, primary.T, primary.B,
                    primary.coatN, primary.coatT, primary.coatB, primary.backface);
                misW = envPdf / max(envPdf + bsdfPdf, 1.0e-6);
            } else {
                shadowTMax = max(fe.dist - kRayEpsilon * (1.0 + dot(primary.geoNormal, fe.wi)), 0.0);
            }
            const float visibility = tracer.traceTransmittance(shadowOrigin, fe.wi, shadowTMax);
            if (visibility > 0.0) {
                contribution = fe.Le * fe.bsdfCos * r.W * fTerm * misW * visibility;
            }
        }
    } else {
        r.W = 0.0;
    }

    if (!all(isfinite(contribution)) || any(isnan(contribution))) {
        contribution = 0.0.xxx;
    }
    contribution = max(contribution, 0.0.xxx);

    t_reservoirsCur[pixelIdx] = r;
    return float4(contribution, float(r.M));  // GI2.7: .w = M for the M-aware denoiser
}

/// One independent GI estimate for a reconstructed primary surface: primary-vertex env NEE +
/// BSDF continuation + the bounce/medium transport loop, followed by the per-sample firefly
/// clamp. `seed` decorrelates this estimate from the others (c1 adaptive sampling averages
/// several of these). Returns false when the sample produced no usable contribution (degenerate
/// BSDF direction or a non-finite result); callers treat that as a zero contribution. `outEnvNee`
/// is kept separate from `outIndirect` so the firefly clamp never biases the sun term.
bool estimateGiSampleCore(SurfaceHit primary, RngState seed, float adaptiveClamp, bool includePrimaryEnvNee,
                          out float3 outIndirect, out float3 outEnvNee) {
    outIndirect = 0.0.xxx;
    outEnvNee = 0.0.xxx;

    PathState state;
    state.radiance = 0.0.xxx;
    state.throughput = 1.0.xxx;
    state.depth = 0u;
    state.seed = seed;
    state.terminated = false;
    state.specularBounce = false;
    state.lastBsdfPdf = 0.0;
    state.passedThrough = false;
    state.insideMedium = false;
    state.mediumSigmaT = 0.0.xxx;
    state.mediumAlbedo = 0.0.xxx;
    state.mediumG = 0.0;
    state.mediumEta = 1.0;

    const bool envEnabled = (hasEnvMap != 0u);
    const bool envImportance = envEnabled && envImportanceWidth > 0u && envImportanceHeight > 0u;
    TheiaTracer tracer;
    float3 origin = 0.0.xxx;
    float3 dir = 0.0.xxx;

    // Primary-vertex opacity gate — this pass's half of mix(ambient-medium, surface, α).
    // The two passes gate DIFFERENT terms, so they compose rather than double-count: the
    // forward pass discards its fragment with probability (1-α), which weights the DIRECT
    // term by α; here the same draw decides whether this pixel's INDIRECT term comes from
    // the surface (α) or from continuing the camera ray into the background (1-α).
    const float primaryOpacity = saturate(primary.mat.opacityFlagsPad.x);
    const bool primaryPassThrough = randFloat(state.seed) < (1.0 - primaryOpacity);

    // GI2.6: primary-vertex env NEE ownership is scene-type-dependent:
    //   • Emissive-lit scenes (emissiveTriangleCount > 0): env NEE stays here —
    //     4 independent samples with own shadow rays + MIS (better for occluded env,
    //     matches v0.7.0 behaviour). The reservoir only handles emissive triangles.
    //   • IBL-only scenes (emissiveTriangleCount == 0): env NEE moved to the reservoir
    //     (restirDiDirect) — giving it spatial+temporal reuse for the first time.
    // The gate is the scene property emissiveTriangleCount, not a knob.
    // GI2 full PT: path-reservoir candidates pass includePrimaryEnvNee = false — the
    // env NEE term is owned outside the walk, so the reservoir never double-counts it.
    float3 primaryEnvNee = 0.0.xxx;
    if (!primaryPassThrough) {
        if (includePrimaryEnvNee && envImportance && emissiveTriangleCount > 0u) {
            float3 sum = 0.0.xxx;
            const uint count = 4u;
            for (uint i = 0u; i < count; ++i) {
                EnvNeeSample envNee = evalEnvNEESample(
                    tracer, primary,
                    envMarginalCdf, envConditionalCdf,
                    envImportanceWidth, envImportanceHeight, envEnabled, envLuminanceScale,
                    t_envMap, s_env, state.seed);
                sum += envNee.contribution;
            }
            primaryEnvNee = sum / float(count);
            if (!all(isfinite(primaryEnvNee)) || any(isnan(primaryEnvNee))) {
                primaryEnvNee = 0.0.xxx;
            }
        }

        // --- Primary BSDF continuation (env/emissive direct handled per scene type above) ---
        BSDFSample bs = sampleBSDF(primary.mat, primary.wo,
            primary.normal, primary.T, primary.B, primary.coatN, primary.coatT, primary.coatB, primary.backface, state.seed);
        if (bs.pdf <= 0.0 || maxComp(bs.weight) <= 0.0 || !all(isfinite(bs.weight))) {
            return false;
        }
        // Shadow-terminator consistency at the primary vertex (Chiang 2019, see Harmonia
        // bsdf_shared.slang): the continuation throughput carries the same microstructure
        // shadowing that attenuates NEE, keeping the MIS integrand identical across
        // techniques (mirrors the shared integrateSurface, which covers bounce 1+).
        float gPrim;
        float fPrim = shadowTerminatorFactor(primary.normal, primary.geoNormal,
                                             normalize(bs.direction), gPrim);
        if (fPrim <= 0.0) {
            state.terminated = true; // sampled direction self-shadowed by the microstructure
        }
        state.throughput = bs.weight * fPrim;
        state.specularBounce = bs.isDelta;
        state.lastBsdfPdf = bs.isDelta ? 1.0e30 : bs.pdf;
        // G9/G10: capture a primary-vertex medium entry (bulk SSS / transmission scatter) so the
        // shared walk runs on the first segment too, matching Hyperion's primary SSS handling.
        state.insideMedium = bs.entersMedium;
        state.mediumSigmaT = bs.mediumSigmaT;
        state.mediumAlbedo = bs.mediumAlbedo;
        state.mediumG      = bs.mediumG;
        state.mediumEta    = bs.mediumEta;
        origin = primary.position + primary.geoNormal * (dot(bs.direction, primary.geoNormal) >= 0.0 ? kRayEpsilon : -kRayEpsilon);
        dir = normalize(bs.direction);
    } else {
        // Null interaction: resume the SAME camera ray with the continuation epsilon (a
        // scattering-sized offset would step over coincident geometry — kNullInteractionEpsilon).
        const float3 fwd = -primary.wo;
        state.specularBounce = true;
        state.lastBsdfPdf = 1.0e30; // delta pass-through for downstream MIS weighting
        origin = primary.position
               + primary.geoNormal * (dot(fwd, primary.geoNormal) >= 0.0 ? kNullInteractionEpsilon
                                                                          : -kNullInteractionEpsilon);
        dir = normalize(fwd);
        state.passedThrough = true;
    }

    // G9/G10: if the primary continuation entered a medium, walk it before the surface loop.
    if (state.insideMedium) {
        runMediumWalk(state, origin, dir);
    }

    // An OpenPBR opacity pass-through selects the ambient-medium side of
    // mix(ambient, surface, geometry_opacity): no surface interaction happened, so it must
    // NOT consume a bounce (charging it blacks out cut-out geometry wherever a ray grazes
    // several crossings). `iter` is the separate termination guard, mirroring Hyperion's
    // raygen loop and the medium walk's own budget.
    const uint bounces = max(maxDepth, 1u);
    const uint maxIters = bounces + kMaxNullInteractions;
    uint bounce = 0u;
    for (uint iter = 0u; bounce < bounces && !state.terminated && iter < maxIters; ++iter) {
        // A null interaction (opacity pass-through) RESUMES the ray instead of scattering it,
        // so it uses the small continuation epsilon — the scattering guard would skip every
        // surface within 1 mm of the cut-out (see kNullInteractionEpsilon).
        GiHit h = traceClosestHit(origin, dir, kInf,
                                  state.passedThrough ? kNullInteractionEpsilon : kRayEpsilon);
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
        // A3(a): path regularization — widen near-specular GGX lobes on secondary+ bounces
        // to reduce firefly variance (Kaplanyan & Dachsbacher 2013; PBRT 4e §16). Trades a
        // small known bias for much lower variance. Theia-only, flag-gated; Hyperion stays
        // the unbiased ground truth.
        //
        // Exempt smooth dielectric transmission (glass): those surfaces refract via the exact
        // Snell delta lobe (see bsdf.slang, baseRough < 0.05). Regularizing their roughness up
        // to 0.3 would frost the glass and push it onto the microfacet-transmission path,
        // broadening/strengthening refracted colour (e.g. the green cast of a gem on the floor)
        // and diverging from Hyperion. Keep them sharp to preserve OpenPBR parity.
        const bool smoothDielectricTransmission =
            saturate(surf.mat.transmissionColorWeight.w) > 0.1 &&
            surf.mat.specularRoughAnisoIor.x < 0.05;
        if (a3RegularizationEnabled != 0u && bounce > 0u && !smoothDielectricTransmission) {
            const float regStrength = 0.3; // roughness floor the lobe is widened toward
            const float curRough = surf.mat.specularRoughAnisoIor.x;
            // Smooth ramp: full effect at roughness 0, fades to nothing by regStrength.
            const float regT = saturate((regStrength - curRough) / max(regStrength, 1.0e-4));
            const float regRough = lerp(curRough, max(curRough, regStrength), regT);
            surf.mat.specularRoughAnisoIor.x = regRough;
            surf.mat.coatRoughAnisoIorDark.x = max(surf.mat.coatRoughAnisoIorDark.x, regRough * 0.5);
        }
        state.depth = bounce + 1u;
        integrateSurface(tracer, surf,
            emissiveTriangleCount, emissiveTriangles, emissiveCdf,
            envMarginalCdf, envConditionalCdf,
            envImportanceWidth, envImportanceHeight, envEnabled, envLuminanceScale,
            t_envMap, s_env, state);

        if (state.terminated) {
            break;
        }
        origin = state.nextOrigin;
        dir = state.nextDirection;
        if (state.passedThrough) {
            continue; // null interaction: same ray, same bounce index
        }
        ++bounce;

        // G9/G10: if the BSDF sample entered a bulk subsurface / transmission-scatter medium,
        // run the shared volumetric random walk here (its own bounded budget) before resuming
        // the surface bounce loop from the refracted exit ray.
        if (state.insideMedium) {
            runMediumWalk(state, origin, dir);
            if (state.terminated) {
                break;
            }
        }
    }

    float3 indirectPrimaryEnvNee = primaryEnvNee * exposure;
    if (!all(isfinite(indirectPrimaryEnvNee)) || any(isnan(indirectPrimaryEnvNee))) {
        indirectPrimaryEnvNee = 0.0.xxx;
    }
    float3 indirect = state.radiance * exposure;
    // Per-sample NaN/Inf guard: discard this sample's indirect term (report invalid) so the
    // caller keeps the pixel's accumulated value instead of poisoning it with a bad bounce.
    if (!all(isfinite(indirect)) || any(isnan(indirect))) {
        return false;
    }
    indirect = max(indirect, 0.0.xxx);
    // Hard ceiling on top of the per-sample firefly suppression below.
    indirect = clamp(indirect, 0.0.xxx, (adaptiveClamp * 3.0).xxx);
    // Per-sample firefly suppression (averaged across accumulated frames + adaptive samples).
    const float avg = (indirect.x + indirect.y + indirect.z) * (1.0 / 3.0);
    if (avg > adaptiveClamp) {
        indirect *= adaptiveClamp / avg;
    }

    outIndirect = indirect;
    outEnvNee = indirectPrimaryEnvNee;
    return true;
}

bool estimateGiSample(SurfaceHit primary, RngState seed, float adaptiveClamp,
                      out float3 outIndirect, out float3 outEnvNee) {
    return estimateGiSampleCore(primary, seed, adaptiveClamp, true, outIndirect, outEnvNee);
}

/// GI2 full PT: one replayable multi-bounce indirect path candidate. Returns the
/// walk's full indirect estimate g (throughput carries 1/pdf → g is the per-seed
/// path-integral estimator value). Primary env NEE is excluded — it is owned by
/// estimateGiSample's envNee channel (emissive scenes) or the DI reservoir
/// (IBL-only scenes), so the path reservoir never double-counts it.
bool traceGiPathCandidate(SurfaceHit primary, RngState seed, float adaptiveClamp, out float3 outG) {
    float3 envNeeUnused;
    return estimateGiSampleCore(primary, seed, adaptiveClamp, false, outG, envNeeUnused);
}

/// GI2 full PT: primary-vertex env NEE WITHOUT the multi-bounce walk, for the
/// path-reservoir mode. Mirrors the opacity gate + 4-sample env NEE block of
/// estimateGiSampleCore exactly (emissive-lit scenes only; IBL-only scenes route
/// env through the DI reservoir). The opacity pass-through draw is independent
/// from the path candidates' — marginally unbiased per term.
float3 primaryEnvNeeOnly(SurfaceHit primary, RngState seed) {
    const bool envEnabled = (hasEnvMap != 0u);
    const bool envImportance = envEnabled && envImportanceWidth > 0u && envImportanceHeight > 0u;
    if (!envImportance || emissiveTriangleCount == 0u) {
        return 0.0.xxx;
    }
    TheiaTracer tracer;
    const float primaryOpacity = saturate(primary.mat.opacityFlagsPad.x);
    if (randFloat(seed) < (1.0 - primaryOpacity)) {
        return 0.0.xxx; // stochastic geometry_opacity pass-through: no primary shading
    }
    float3 sum = 0.0.xxx;
    const uint count = 4u;
    for (uint i = 0u; i < count; ++i) {
        EnvNeeSample envNee = evalEnvNEESample(
            tracer, primary,
            envMarginalCdf, envConditionalCdf,
            envImportanceWidth, envImportanceHeight, envEnabled, envLuminanceScale,
            t_envMap, s_env, seed);
        sum += envNee.contribution;
    }
    float3 result = (sum / float(count)) * exposure;
    if (!all(isfinite(result)) || any(isnan(result))) {
        return 0.0.xxx;
    }
    return result;
}

/// GI2 full PT: multi-bounce path reservoir for the indirect term (ReSTIR PT,
/// GRIS random-replay shift). RIS over replayable path candidates: local walks +
/// spatial reuse (G-buffer gated neighbours), all replayed at the CURRENT primary
/// surface. Returns float4(indirectContribution, reservoirM).
///
/// NO TEMPORAL MERGE — by design. A streamed temporal merge feeds yesterday's
/// MERGED reservoir back as today's input (feedback recursion); with heavy-tailed
/// path candidates the re-selected history double-counts and the per-frame
/// estimator drifts a few % (measured: −2.8% darker on cornell indirect at
/// 1024 frames, confirmed by Monte-Carlo simulation of the merge; M-cap sweeps
/// do not fix it). Theia's progressive accumulation already integrates the
/// temporal axis unbiasedly, so temporal reservoir reuse is redundant here.
/// Instead, spatial reuse reads neighbours' UNMERGED local reservoirs (written
/// last frame, never the merged output): a neighbour's local candidate replayed
/// at this surface is statistically identical to a fresh local candidate
/// (seed-space uniformity), so the merged estimator is plain unbiased RIS over
/// ≤ 1+3 independent candidates and the frame average converges to ground truth.
float4 restirPtIndirect(SurfaceHit primary, uint2 px, float adaptiveClamp, inout RngState rng) {
    const uint kPathLocalCandidates = 1u;   // local walks per pixel per frame
    const uint pixelIdx = px.y * screenWidth + px.x;

    // ── Local candidates — stored UNMERGED for next frame's spatial reuse. ──
    PathReservoir rLocal = makeEmptyPathReservoir();
    for (uint i = 0u; i < kPathLocalCandidates; ++i) {
        const RngState candSeed = pathCandidateSeed(px, i);
        float3 g;
        const bool valid = traceGiPathCandidate(primary, candSeed, adaptiveClamp, g);
        pathReservoirUpdate(rLocal, candSeed, g, valid, rng);
    }
    // Unbiased weight of the local reservoir (neighbours merge with w = p̂·W·M).
    {
        const float pHatL = restirTargetLuma(rLocal.g);
        rLocal.W = (rLocal.M > 0u && pHatL > 0.0)
            ? rLocal.w_sum / max(float(rLocal.M) * pHatL, 1.0e-10) : 0.0;
    }
    t_pathReservoirsCur[pixelIdx] = rLocal;

    // ── Spatial reuse: merge neighbours' LOCAL reservoirs (independent candidates). ──
    PathReservoir r = rLocal;
    if (restirDiSpatial != 0u) {
        const uint  kPathSpatialNeighbors = 3u;
        const float kSpatialRadius        = 32.0;
        const float kPosThresholdRel      = 0.01;
        const float kNormalThreshold      = 0.98;

        const float3 curWorldPos = t_giBuffer.Load(int3(int(px.x), int(px.y), 0)).xyz;
        const float3 curViewN    = t_gbuffer.Load(int3(int(px.x), int(px.y), 0)).xyz * 2.0 - 1.0;
        const float  viewDist    = length(curWorldPos - cameraPos.xyz);
        const float  posThresh   = max(viewDist * kPosThresholdRel, 1.0e-3);

        // Duplication map (GI2.5): skip a neighbour whose selected candidate was
        // already merged from a different neighbour (same replay token).
        uint seenHashes[4]; // 1 (own) + up to 3 neighbours
        seenHashes[0] = r.candidateHash;
        uint seenCount = 1u;

        for (uint n = 0u; n < kPathSpatialNeighbors; ++n) {
            const float2 off = (float2(randFloat(rng), randFloat(rng)) * 2.0 - 1.0) * kSpatialRadius;
            const int2 nCoord = int2(int(px.x) + int(round(off.x)), int(px.y) + int(round(off.y)));
            if (nCoord.x < 0 || nCoord.y < 0 ||
                nCoord.x >= int(screenWidth) || nCoord.y >= int(screenHeight)) {
                continue;
            }
            const float3 nWorldPos = t_giBuffer.Load(int3(nCoord.x, nCoord.y, 0)).xyz;
            const float3 nViewN    = t_gbuffer.Load(int3(nCoord.x, nCoord.y, 0)).xyz * 2.0 - 1.0;
            if (length(curWorldPos - nWorldPos) > posThresh) continue;
            if (dot(curViewN, nViewN) < kNormalThreshold) continue;

            PathReservoir neighbor = t_pathReservoirsPrev[uint(nCoord.y) * screenWidth + uint(nCoord.x)];
            if (neighbor.M == 0u) continue;
            if (neighbor.W == 0.0) {
                // Empty local reservoir (dark walk): counts as a candidate attempt
                // but carries no sample — no replay walk needed.
                r.M += neighbor.M;
                continue;
            }
            bool duplicate = false;
            for (uint h = 0u; h < seenCount; ++h) {
                if (seenHashes[h] == neighbor.candidateHash) { duplicate = true; break; }
            }
            if (duplicate) continue;
            if (seenCount < 4u) seenHashes[seenCount++] = neighbor.candidateHash;

            float3 gN;
            const bool vn = traceGiPathCandidate(primary, pathReplaySeed(neighbor.rngKey, neighbor.rngIndex),
                                                 adaptiveClamp, gN);
            pathReservoirCombine(r, neighbor, gN, vn, rng);
        }
    }

    // ── Final unbiased weight W = w_sum / (M · p̂(y)); contribution = g(y)·W. ──
    float3 contribution = 0.0.xxx;
    const float pHat = restirTargetLuma(r.g);
    if (r.M > 0u && pHat > 0.0) {
        r.W = r.w_sum / max(float(r.M) * pHat, 1.0e-10);
        contribution = r.g * r.W;
    } else {
        r.W = 0.0;
    }
    if (!all(isfinite(contribution)) || any(isnan(contribution))) {
        contribution = 0.0.xxx;
    }
    // Hard firefly ceiling matching the DI reservoir / indirect path.
    contribution = clamp(contribution, 0.0.xxx, (adaptiveClamp * 3.0).xxx);

    return float4(contribution, float(r.M));
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
    // w = asfloat(instanceIdx+1): 0 means background / sky (A1b: changed from materialIdx+1).
    const uint instIdxPlus1 = asuint(giSample.w);
    if (instIdxPlus1 == 0u) {
        return; // background / non-geometry pixel (and transparent pixels keep the sentinel)
    }
    const uint primaryInstIdx = instIdxPlus1 - 1u;
    // Derive the material index from the instance table — the ray-query primary hit
    // below will use its own instIdx, but the fallback path needs this as a best-guess.
    const uint materialIdx = instances[primaryInstIdx].materialIndex;
    const float3 worldPos = giSample.xyz;

    // Reconstruct the primary surface by re-tracing the camera ray to the recorded world
    // position; this works for opaque AND transparent (glass) pixels alike. The view-space
    // G-buffer normal is only needed as a fallback when the re-trace misses (e.g. the
    // transparent pass does not write the normal G-buffer), so it is computed lazily below.
    GiHit primaryHit = traceClosestHit(cameraPos.xyz, worldPos - cameraPos.xyz, kInf, kRayEpsilon);
    // Trust the re-traced closest hit as the authoritative nearest surface. The rasterized
    // G-buffer materialIdx is draw-order dependent for transparent geometry (depth-write
    // disabled), so overlapping transparent surfaces can record the farther one; the inline
    // ray query is order-independent and always returns the geometrically nearest surface.
    // Matches industry practice (Q2RTX reflect_refract, PowerVR HybridTransparency, SEED
    // "Hybrid Rendering for Real-Time Ray Tracing" RTG ch.25). The matIdxPlus1==0 sentinel
    // check above already handles background pixels; for non-background, the re-trace wins.
    const bool primaryHitValid = primaryHit.hit;
    const float3 primaryWorldPos = primaryHitValid ? primaryHit.worldPos : worldPos;

    const float3 wo = normalize(cameraPos.xyz - primaryWorldPos);
    SurfaceHit primary;
    if (primaryHitValid) {
        primary = makeSurfaceHit(primaryHit, wo);
    } else {
        // Fallback: reconstruct the shading normal from the view-space G-buffer normal.
        // viewN = mul(viewRot, N) in the forward pass (R_cpu·N = N_view, column-vector form).
        // The inverse for an orthonormal rotation is mul(viewN, viewRot) (row-vector form;
        // computes R_cpu^T·viewN = N_world). When the normal G-buffer is unavailable
        // (zero, e.g. a transparent pixel against the sky), there is nothing to reconstruct
        // here → skip the pixel.
        const float3 viewN = t_gbuffer.Load(loadCoord).xyz * 2.0 - 1.0;
        const float3x3 viewRot = float3x3(view[0].xyz, view[1].xyz, view[2].xyz);
        float3 worldN = mul(viewN, viewRot);
        if (dot(worldN, worldN) < 1.0e-8) {
            return;
        }
        worldN = normalize(worldN);
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
        if (dot(worldN, wo) < 0.0) {
            worldN = -worldN;
        }
        const float3x3 tbn = buildTBN(worldN);
        primary.position = primaryWorldPos;
        primary.shadowPos = primaryWorldPos;
        primary.normal = worldN;
        primary.geoNormal = worldN;
        primary.T = tbn[0];
        primary.B = tbn[1];
        primary.coatN = worldN;
        primary.coatT = tbn[0];
        primary.coatB = tbn[1];
        primary.backface = false; // G-buffer fallback: rasterized camera-facing surface
        primary.mat = mat;
        primary.wo = wo;
    }

    // Skip self-emissive primary surfaces: their radiance is the emitter itself, shown
    // directly by the forward pass; bounced light off an emitter is negligible.
    const float3 primaryEmission = max(primary.mat.emissionColorLum.xyz, 0.0.xxx) *
                                   max(primary.mat.emissionColorLum.w, 0.0);
    if (maxComp(primaryEmission) > 0.0) {
        return;
    }

    // A3(b): variance-aware adaptive firefly clamp. When the A-SVGF gradient/variance guide is
    // bound, high local variance (genuine bright/complex region) relaxes the clamp while low
    // variance keeps the tight legacy threshold, suppressing isolated outliers. With
    // hasGradientVariance == 0 this reduces exactly to the legacy fixed kFireflyClampDisplay.
    float variance = 0.0;
    float adaptiveClamp = kFireflyClampDisplay;
    if (hasGradientVariance != 0u) {
        variance = t_gradientVariance.Load(loadCoord).g;
        // Higher local variance → allow higher values (genuine bright spot);
        // lower local variance → tighter clamp (isolated outlier).
        adaptiveClamp = kFireflyClampDisplay * max(1.0, 1.0 + variance * 20.0);
    }

    // c1: variance-guided adaptive sampling. Spend extra unbiased GI samples in high-variance
    // pixels (as flagged by the A-SVGF guide from prior frames — independent of this frame's
    // realizations, so the per-pixel average stays unbiased). Without the guide, maxSamples
    // collapses to 1 and the seed reduces to composeRngSeed(px, frameSampleIndex, ...): the
    // estimator is then bit-identical to the legacy single-sample path.
    const uint maxSamples = (hasGradientVariance != 0u) ? max(giAdaptiveMaxSamples, 1u) : 1u;
    uint sampleCount = 1u;
    if (maxSamples > 1u) {
        const float t = saturate(variance * kAdaptiveVarianceScale);
        sampleCount = 1u + uint(float(maxSamples - 1u) * t + 0.5);
    }

    // A4/GI2: direct lighting from emissive triangles at the primary vertex. Both the
    // legacy ReSTIR DI and the GI2 ReSTIR PT paths route through the SAME reservoir
    // (restirDiDirect) — PT absorbs DI per the sprint plan GI2 conflict resolution. The
    // reservoir does initial-candidate RIS + temporal reuse + spatial reuse (Phase 2),
    // computes the unbiased weight W, and traces ONE shadow ray. The forward pass skips
    // its emissive-derived rect lights for both paths (shared setRestirDiActive semantic).
    //    restirDiEnabled is set by GiPass when EITHER useRestirDi OR useRestirPt is true
    //    and the reservoir buffers are valid; restirDiSpatial is set when EITHER path
    //    requests spatial reuse. PT enables spatial reuse by default (Application.cpp).
    float3 restirDirect = 0.0.xxx;
    float  restirM = 0.0;  // GI2.7: reservoir effective sample count for the M-aware denoiser
    // GI2.6: reservoir runs for BOTH scene types. Emissive-lit scenes: reservoir handles
    // emissive-triangle direct only (v0.7.0 behaviour — env NEE stays in estimateGiSample
    // because the reservoir's single-shadow-ray scheme wastes candidates on occluded env
    // directions in enclosed scenes). IBL-only scenes: reservoir handles env direct
    // (giving it spatiotemporal reuse for the first time). The gate is the scene property
    // emissiveTriangleCount, evaluated inside restirDiDirect's candidate generation.
    if (restirDiEnabled != 0u && (emissiveTriangleCount > 0u || hasEnvMap != 0u)) {
        RngState restirSeed = composeRngSeed(px, frameSampleIndex, 0xA4u, rngBaseSeed);
        const float4 restirResult = restirDiDirect(primary, px, restirSeed);
        restirDirect = restirResult.xyz * exposure;
        restirM = restirResult.w;  // GI2.7: carry M via the HDR alpha channel
        if (!all(isfinite(restirDirect)) || any(isnan(restirDirect))) {
            restirDirect = 0.0.xxx;
        }
        // Hard firefly ceiling matching the indirect path (adaptiveClamp*3).
        restirDirect = clamp(restirDirect, 0.0.xxx, (adaptiveClamp * 3.0).xxx);
    }

    // GI2 full PT: multi-bounce path reservoir owns the indirect walk. Replaces the
    // per-sample estimateGiSample multi-bounce loop below; primary env NEE (emissive
    // scenes) still runs per-sample via primaryEnvNeeOnly. The M channel accumulates
    // the path reservoir's M alongside the DI reservoir's for the M-aware denoiser.
    float3 restirPathIndirect = 0.0.xxx;
    if (restirPtPathEnabled != 0u) {
        RngState pathSeed = composeRngSeed(px, frameSampleIndex, 0xB7u, rngBaseSeed);
        const float4 pathResult = restirPtIndirect(primary, px, adaptiveClamp, pathSeed);
        restirPathIndirect = pathResult.xyz;
        restirM += pathResult.w;
    }

    float3 sumIndirect = 0.0.xxx;
    float3 sumEnvNee = 0.0.xxx;
    for (uint s = 0u; s < sampleCount; ++s) {
        // Decorrelate each extra sample with a distinct Sobol' index. Reserving maxSamples
        // indices per frame keeps samples from overlapping across frames or within a pixel.
        RngState sampleSeed = composeRngSeed(px, frameSampleIndex * maxSamples + s, 0u, rngBaseSeed);
        if (restirPtPathEnabled != 0u) {
            // Path reservoir owns the multi-bounce walk; only primary env NEE remains.
            sumEnvNee += primaryEnvNeeOnly(primary, sampleSeed);
            continue;
        }
        float3 sampleIndirect;
        float3 sampleEnvNee;
        const bool valid = estimateGiSample(primary, sampleSeed, adaptiveClamp, sampleIndirect, sampleEnvNee);
        // Preserve legacy behaviour on the non-adaptive path: a single invalid sample leaves
        // the pixel's accumulated indirect value untouched. The direct term (ReSTIR DI or
        // ReSTIR PT — whichever ran above) is still composited so direct keeps converging
        // even when the GI bounce degenerates.
        if (maxSamples == 1u && !valid) {
            if (restirDirect.x != 0.0 || restirDirect.y != 0.0 || restirDirect.z != 0.0) {
                // GI2.7: M in alpha — OVERWRITE (not +=) so the forward pass's
                // indirectWeight doesn't pollute the M channel. RGB still adds to
                // the forward pass's base color via the read in the same expression.
                hdrImage[px] = float4(hdrImage[px].xyz + restirDirect, restirM);
            }
            return;
        }
        // Invalid samples on the adaptive path contribute zero (degenerate/non-finite outcome).
        sumIndirect += sampleIndirect;
        sumEnvNee += sampleEnvNee;
    }

    const float invN = 1.0 / float(sampleCount);
    // GI2.7: M in alpha — OVERWRITE (not +=) so the forward pass's indirectWeight
    // doesn't pollute the M channel. RGB still adds to the forward pass's base color.
    hdrImage[px] = float4(hdrImage[px].xyz + (sumIndirect + sumEnvNee) * invN + restirDirect + restirPathIndirect, restirM);
}
