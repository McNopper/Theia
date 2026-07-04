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
    uint     a3RegularizationEnabled;      // A3(a): secondary-bounce roughness regularization (Theia-only bias)
    uint     a3ChromaticImportanceEnabled; // A3(c): σ_t-importance-weighted hero channel selection
    uint     hasGradientVariance;          // A3(b): 1 when t_gradientVariance holds a real A-SVGF guide
    uint     giAdaptiveMaxSamples;         // c1: max GI samples/pixel for variance-guided adaptive sampling (1 = off)
};

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

// Per-sample firefly clamp on the indirect contribution (display-referred, post-exposure),
// matching the forward pass kFireflyClampDisplay threshold.
static const float kFireflyClampDisplay = 100.0;

// c1: variance→extra-samples mapping. Local A-SVGF variance is scaled by this factor and
// saturated to [0,1]; the result linearly interpolates the per-pixel GI sample count between
// 1 and giAdaptiveMaxSamples. Matches the magnitude the A3(b) clamp uses (variance * 20).
static const float kAdaptiveVarianceScale = 20.0;

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
    float3 shadingNormal;
    float3 geoNormal; // geometric normal, oriented to face the incoming ray's origin
    float2 uv;
    float3 tangent;
    float  dist;      // committed ray T (used for volumetric medium boundary transmittance)
};

float maxComp(float3 v) { return max(v.x, max(v.y, v.z)); }

/// Closest-hit query against the scene TLAS, reconstructing position/material/normal.
/// `maxDist` bounds the ray (kInf for ordinary bounces; a sampled free-flight distance for
/// volumetric medium segments — a miss within maxDist is an in-medium scattering event).
GiHit traceClosestHit(float3 origin, float3 dir, float maxDist) {
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
    ray.TMin = kRayEpsilon;
    ray.TMax = maxDist;

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
    const float3 n0 = vertices[i0].normal;
    const float3 n1 = vertices[i1].normal;
    const float3 n2 = vertices[i2].normal;
    const float2 uv0 = vertices[i0].uv;
    const float2 uv1 = vertices[i1].uv;
    const float2 uv2 = vertices[i2].uv;

    const float2 b = q.CommittedTriangleBarycentrics();
    const float3 bary = float3(1.0 - b.x - b.y, b.x, b.y);
    float3 Ng = normalize(cross(p1 - p0, p2 - p0));
    if (!frontFace) {
        Ng = -Ng;
    }

    hit.worldPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    hit.uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    const float2 duv1 = uv1 - uv0;
    const float2 duv2 = uv2 - uv0;
    const float det = duv1.x * duv2.y - duv2.x * duv1.y;
    hit.tangent = abs(det) > 1.0e-8 ? ((p1 - p0) * duv2.y - (p2 - p0) * duv1.y) / det : float3(1.0, 0.0, 0.0);
    float3 Ns = n0 * bary.x + n1 * bary.y + n2 * bary.z;
    if (dot(Ns, Ns) > 1.0e-8) {
        Ns = normalize(Ns);
    } else {
        Ns = Ng;
    }
    if (dot(Ns, Ng) < 0.0) {
        Ns = -Ns;
    }
    hit.shadingNormal = Ns;
    hit.geoNormal = Ng;
    hit.materialIdx = inst.materialIndex;
    hit.dist = q.CommittedRayT();
    hit.hit = true;
    return hit;
}

/// G9/G10: run the shared volumetric random walk (chromatic hero-wavelength estimator) inside
/// the compute GI loop, mirroring Hyperion's raygen/miss/closesthit. Consumes its own bounded
/// budget (does NOT count against the surface bounce budget). On exit, updates origin/dir to the
/// refracted ray leaving the medium and clears insideMedium; on absorption terminates the path.
void runMediumWalk(inout PathState state, inout float3 origin, inout float3 dir) {
    const uint kMaxMediumWalk = 256u;
    for (uint i = 0u; i < kMaxMediumWalk && state.insideMedium && !state.terminated; ++i) {
        float3 st = max(state.mediumSigmaT, 1.0e-6.xxx);
        // A3(c): σ_t-importance-weighted hero channel selection (flag-gated, Theia-only knob).
        // Legacy estimator picks each channel uniformly (1/3); with the flag on, each channel
        // is selected proportionally to its extinction so samples are not wasted on
        // near-transparent (σ_t ≈ 0) channels — Kutz et al. 2017 (spectral tracking),
        // adapted from spectral to RGB channels.
        //
        // Unbiasedness: the free-flight distance is drawn from the hero channel's exponential
        // distribution, so marginalized over the channel choice the sampling pdfs are the
        // mixtures  p_coll(t) = Σ_c chProb_c · σ_c e^{-σ_c t}   (collision at t)
        //           P_surf(d) = Σ_c chProb_c · e^{-σ_c d}       (flight past the boundary at d)
        // Dividing the spectral integrand by these mixture pdfs below is the standard
        // one-sample balance-heuristic estimator, exactly unbiased for ANY chProb > 0 —
        // the flag only changes variance, never the expected value. With the flag off,
        // chProb = 1/3 reproduces the legacy uniform estimator bit-for-bit.
        float3 chProb = (1.0 / 3.0).xxx;
        if (a3ChromaticImportanceEnabled != 0u) {
            chProb = st / (st.x + st.y + st.z);
        }
        float chSel = randFloat(state.seed);
        float sigmaHero = chSel < chProb.x ? st.x : (chSel < chProb.x + chProb.y ? st.y : st.z);
        float u = randFloat(state.seed);
        float freeDist = -log(max(1.0 - u, 1.0e-6)) / max(sigmaHero, 1.0e-6);

        GiHit h = traceClosestHit(origin, dir, freeDist);
        if (!h.hit) {
            // Scatter event at freeDist — spectral factor sigma_s*T/pdf_coll.
            float3 T = exp(-st * freeDist);
            float3 sigmaS = state.mediumAlbedo * st;
            float pdfColl = dot(chProb, st * T);
            state.throughput *= (sigmaS * T) / max(pdfColl, 1.0e-8);
            float q = saturate(maxComp(state.throughput));
            if (randFloat(state.seed) > q) { state.terminated = true; return; }
            state.throughput /= max(q, 1.0e-3);
            float3 scatterPos = origin + dir * freeDist;
            float3 newDir = sampleHenyeyGreenstein(dir, state.mediumG, randFloat2(state.seed));
            origin = scatterPos + newDir * kRayEpsilon;
            dir = newDir;
            continue;
        }

        // Boundary crossing at h.dist — spectral transmittance factor, then Fresnel exit/TIR.
        float3 Tb = exp(-st * h.dist);
        float pdfSurf = dot(chProb, Tb);
        state.throughput *= Tb / max(pdfSurf, 1.0e-8);

        float3 outwardN = h.geoNormal;
        float cosI = abs(dot(normalize(dir), outwardN));
        float F = fresnelDielectric(-cosI, state.mediumEta);
        float3 refracted = refract(normalize(dir), -outwardN, state.mediumEta);
        bool tir = dot(refracted, refracted) < 1.0e-9;
        if (tir || randFloat(state.seed) < F) {
            dir = reflect(normalize(dir), outwardN);
            origin = h.worldPos - outwardN * kRayEpsilon;
            continue;
        }
        dir = normalize(refracted);
        origin = h.worldPos + outwardN * kRayEpsilon;
        state.insideMedium = false;
        return;
    }
    if (state.insideMedium) { state.terminated = true; }
}

float4 sampleBindless(uint idx, float2 uv) {
    // Compute shaders have no screen-space quad, so implicit-LOD .Sample() (which needs
    // SPV_KHR_compute_shader_derivatives) is both unsupported here and meaningless for a
    // secondary ray hit. Sample mip 0 explicitly — GI hits carry no pixel footprint.
    return t_bindless[NonUniformResourceIndex(idx)].SampleLevel(uv, 0.0);
}

/// Populate a SurfaceHit from a traced hit, using the same texture-mapped material
/// state as the path tracer while keeping the geometric normal for robust offsets.
SurfaceHit makeSurfaceHit(GiHit h, float3 wo) {
    SurfaceHit surf;
    GpuMaterial mat = materials[h.materialIdx];

    if (mat.textureIndices.x != kNoTexture) {
        mat.baseColorWeight.xyz = sampleBindless(mat.textureIndices.x, h.uv).rgb;
    }
    if (mat.textureIndices.z != kNoTexture) {
        float3 orm = sampleBindless(mat.textureIndices.z, h.uv).rgb;
        mat.specularRoughAnisoIor.x = orm.g;
        mat.baseMetalnessDiffRough.x = orm.b;
    }
    if (mat.textureIndices.w != kNoTexture) {
        mat.emissionColorLum.xyz *= sampleBindless(mat.textureIndices.w, h.uv).rgb;
    }

    float3 N = h.shadingNormal;
    float3 Ng = h.geoNormal;
    if (dot(N, wo) < 0.0) {
        N = -N;
    }
    if (dot(Ng, wo) < 0.0) {
        Ng = -Ng;
    }
    if (mat.textureIndices.y != kNoTexture) {
        float3 nTex = sampleBindless(mat.textureIndices.y, h.uv).xyz * 2.0 - 1.0;
        float3 Tt = h.tangent - N * dot(N, h.tangent);
        if (dot(Tt, Tt) > 1.0e-12) {
            Tt = normalize(Tt);
            float3 Bt = cross(N, Tt);
            float3 Np = normalize(Tt * nTex.x + Bt * nTex.y + N * nTex.z);
            N = dot(wo, Np) >= 0.0 ? Np : N;
        }
    }
    const float3x3 tbn = buildTBN(N);
    surf.position = h.worldPos;
    surf.normal = N;
    surf.geoNormal = Ng;
    surf.T = tbn[0];
    surf.B = tbn[1];
    surf.coatN = N;
    surf.coatT = tbn[0];
    surf.coatB = tbn[1];
    if (mat.textureIndices2.x != kNoTexture) {
        float3 nTex = sampleBindless(mat.textureIndices2.x, h.uv).xyz * 2.0 - 1.0;
        float3 Tt = h.tangent - N * dot(N, h.tangent);
        if (dot(Tt, Tt) > 1.0e-12) {
            Tt = normalize(Tt);
            float3 Bt = cross(N, Tt);
            float3 Np = normalize(Tt * nTex.x + Bt * nTex.y + N * nTex.z);
            surf.coatN = dot(wo, Np) >= 0.0 ? Np : N;
            const float3x3 coatTbn = buildTBN(surf.coatN);
            surf.coatT = coatTbn[0];
            surf.coatB = coatTbn[1];
        }
    }
    if (mat.textureIndices2.z != kNoTexture) {
        float2 tDir = sampleBindless(mat.textureIndices2.z, h.uv).xy * 2.0 - 1.0;
        float3 Tuv = h.tangent - surf.coatN * dot(surf.coatN, h.tangent);
        if (dot(Tuv, Tuv) > 1.0e-12 && dot(tDir, tDir) > 1.0e-12) {
            Tuv = normalize(Tuv);
            float3 Buv = cross(surf.coatN, Tuv);
            float3 Tn = normalize(Tuv * tDir.x + Buv * tDir.y);
            float3 cT = Tn - surf.coatN * dot(surf.coatN, Tn);
            if (dot(cT, cT) > 1.0e-12) {
                surf.coatT = normalize(cT);
                surf.coatB = cross(surf.coatN, surf.coatT);
            }
        }
    }
    surf.mat = mat;
    surf.wo = wo;
    return surf;
}

/// One independent GI estimate for a reconstructed primary surface: primary-vertex env NEE +
/// BSDF continuation + the bounce/medium transport loop, followed by the per-sample firefly
/// clamp. `seed` decorrelates this estimate from the others (c1 adaptive sampling averages
/// several of these). Returns false when the sample produced no usable contribution (degenerate
/// BSDF direction or a non-finite result); callers treat that as a zero contribution. `outEnvNee`
/// is kept separate from `outIndirect` so the firefly clamp never biases the sun term.
bool estimateGiSample(SurfaceHit primary, RngState seed, float adaptiveClamp,
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

    // Primary-vertex stochastic geometry_opacity pass-through gate (shared-integrator contract):
    // with probability (1-opacity), continue the camera ray unchanged with delta semantics.
    const float primaryOpacity = saturate(primary.mat.opacityFlagsPad.x);
    const bool primaryPassThrough = randFloat(state.seed) < (1.0 - primaryOpacity);

    // Primary-vertex environment NEE only (shared estimator conventions). Keep this in a
    // separate accumulator so firefly suppression does not bias the sun term.
    float3 primaryEnvNee = 0.0.xxx;
    if (!primaryPassThrough) {
        if (envImportance) {
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

        // --- Primary BSDF continuation (primary env NEE above; analytic/emissive direct handled by forward) ---
        BSDFSample bs = sampleBSDF(primary.mat, primary.wo,
            primary.normal, primary.T, primary.B, primary.coatN, primary.coatT, primary.coatB, state.seed);
        if (bs.pdf <= 0.0 || maxComp(bs.weight) <= 0.0 || !all(isfinite(bs.weight))) {
            return false;
        }
        state.throughput = bs.weight;
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
        const float3 fwd = -primary.wo;
        state.specularBounce = true;
        state.lastBsdfPdf = 1.0e30; // delta pass-through for downstream MIS weighting
        origin = primary.position + primary.geoNormal * (dot(fwd, primary.geoNormal) >= 0.0 ? kRayEpsilon : -kRayEpsilon);
        dir = normalize(fwd);
    }

    // G9/G10: if the primary continuation entered a medium, walk it before the surface loop.
    if (state.insideMedium) {
        runMediumWalk(state, origin, dir);
    }

    const uint bounces = max(maxDepth, 1u);
    for (uint bounce = 0u; bounce < bounces && !state.terminated; ++bounce) {
        GiHit h = traceClosestHit(origin, dir, kInf);
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
    GiHit primaryHit = traceClosestHit(cameraPos.xyz, worldPos - cameraPos.xyz, kInf);
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
        // viewN = mul(N, viewRot) in the forward pass (row-vector); the inverse for an
        // orthonormal rotation is mul(viewRot, viewN) (column-vector). When the normal
        // G-buffer is unavailable (zero, e.g. a transparent pixel against the sky), there
        // is nothing to reconstruct here -> skip the pixel.
        const float3 viewN = t_gbuffer.Load(loadCoord).xyz * 2.0 - 1.0;
        const float3x3 viewRot = float3x3(view[0].xyz, view[1].xyz, view[2].xyz);
        float3 worldN = mul(viewRot, viewN);
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
        primary.normal = worldN;
        primary.geoNormal = worldN;
        primary.T = tbn[0];
        primary.B = tbn[1];
        primary.coatN = worldN;
        primary.coatT = tbn[0];
        primary.coatB = tbn[1];
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

    float3 sumIndirect = 0.0.xxx;
    float3 sumEnvNee = 0.0.xxx;
    for (uint s = 0u; s < sampleCount; ++s) {
        // Decorrelate each extra sample with a distinct Sobol' index. Reserving maxSamples
        // indices per frame keeps samples from overlapping across frames or within a pixel.
        RngState sampleSeed = composeRngSeed(px, frameSampleIndex * maxSamples + s, 0u, rngBaseSeed);
        float3 sampleIndirect;
        float3 sampleEnvNee;
        const bool valid = estimateGiSample(primary, sampleSeed, adaptiveClamp, sampleIndirect, sampleEnvNee);
        // Preserve legacy behaviour on the non-adaptive path: a single invalid sample leaves
        // the pixel's accumulated value untouched (no write) instead of adding zero.
        if (maxSamples == 1u && !valid) {
            return;
        }
        // Invalid samples on the adaptive path contribute zero (degenerate/non-finite outcome).
        sumIndirect += sampleIndirect;
        sumEnvNee += sampleEnvNee;
    }

    const float invN = 1.0 / float(sampleCount);
    hdrImage[px] += float4((sumIndirect + sumEnvNee) * invN, 0.0);
}
