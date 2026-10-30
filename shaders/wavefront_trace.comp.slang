// wavefront_trace.comp.slang — PERF5 stage 10c: the wavefront TRACE pass.
//
// Stage-1 design (PLAN.md "PERF5 wavefront design (stage 10a)" §1): closest-hit
// query for every live path in the current wave's ray buffer → hit buffer. The
// RayQuery loop below is the VERBATIM copy of gi.comp.slang's
// `traceClosestHit` traversal (same flags — RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES
// with all non-opaque candidates committed, same cull mask kInstanceMaskAll,
// same tMin/tMax semantics — the record carries the ray's own epsilon/infinity
// pair). The record's `direction` is the unnormalized camera delta exactly as
// the pre-split entry traced it; it is normalized HERE at the same point the
// old code normalized it (inside the trace, before the query) — bit contract.
//
// Vertex reconstruction does NOT happen here: the pass writes the raw
// committed-hit data (instance/primitive id, barycentrics, ray T, front-face
// bit) or the miss sentinel into WaveHit, and the consumer repeats
// traceClosestHit's vertex fetch + reconstruction statements verbatim
// (reconstructGiHit, gi.comp.slang). For rays
// that miss — or that hit a triangle with a degenerate index range, which the
// pre-split code treated as a miss — the pass writes the kWaveHitMiss sentinel
// (Q1); the consumer then applies its existing rasterized-worldPos fallback
// unchanged.
//
// 10c DISPATCH STATUS: this pass is fully live and CONSUMED — dispatched
// generate → trace → GI (the trace→GI write→read barrier in place), and the
// shade-side GI dispatch reconstructs its wave-0 primary surface from record 0
// (reconstructGiHit, gi.comp.slang — this pass's verbatim continuation); the
// miss sentinel (Q1) triggers the consumer's G-buffer fallback, and the inline
// primary trace in gi.comp.slang is gone — this pass is the single source of
// wave-0 hits. Verified under the CALIBRATED gate (PLAN.md "PERF5 wavefront
// design" §3, amended from byte-equality): the driver recomputes committed-hit
// barycentrics/rayT at the RayQuery site and FP contraction follows the
// surrounding compiled context — Slang emits no NoContraction decorations — so
// consuming these values outside traceClosestHit's own compiled context shifts
// the primary surface by 1–2 ULP on ~10% of the pixels (7387/76800 px on
// cornell_classic @256f, max 2.6e-7 linear; four bisect experiments → exactly
// two driver-codegen classes: the inline-trace context, and any restructure).
// One deterministic artifact, byte-equal across ALL restructured variants —
// not estimator nondeterminism (same RNG streams, same draws, same statement
// order). Upstream-reportable to Slang: missing NoContraction on ray-query
// recomputes. Dead slots (alive bit clear) are skipped: the buffers are never
// cleared per frame, and no consumer reads the hit of a dead record.
//
// Push constants: the SAME cbuffer declaration as gi.comp.slang — all three
// wavefront pipelines share one pipeline layout, so this copy must stay
// field-for-field identical.

import common;       // GpuInstance, kInstanceMaskAll
import wave_common;  // WavePath / WaveHit / kWaveHitMiss + slot/flag constants

[[vk::binding(0, 0)]]  RaytracingAccelerationStructure       t_tlas;
[[vk::binding(8, 0)]]  StructuredBuffer<GpuInstance>         instances;
[[vk::binding(25, 0)]] StructuredBuffer<WavePath>            t_wavePathsCur; // wave-N ray buffer (read)
[[vk::binding(27, 0)]] RWStructuredBuffer<WaveHit>           t_waveHits;     // trace-pass output

[[vk::push_constant]]
cbuffer GiPC {
    float4x4 view;                  // (unused here; layout parity)
    float4x4 prevViewProj;          // (unused here; layout parity)
    float4   cameraPos;             // (unused here — the ray origin rides in the record; layout parity)
    float    exposure;              // (unused here; layout parity)
    uint     frameSampleIndex;      // (unused here; layout parity)
    uint     rngBaseSeed;           // (unused here; layout parity)
    uint     emissiveTriangleCount; // (unused here; layout parity)
    uint     envImportanceWidth;    // (unused here; layout parity)
    uint     envImportanceHeight;   // (unused here; layout parity)
    uint     hasEnvMap;             // (unused here; layout parity)
    float    envLuminanceScale;     // (unused here; layout parity)
    uint     maxDepth;              // (unused here; layout parity)
    uint     screenWidth;
    uint     screenHeight;
    uint     _pad0;
    uint     a3RegularizationEnabled;      // (unused here; layout parity)
    uint     hasGradientVariance;          // (unused here; layout parity)
    uint     giAdaptiveMaxSamples;         // (unused here; layout parity)
    uint     restirDiEnabled;              // (unused here; layout parity)
    uint     restirDiSpatial;              // (unused here; layout parity)
    uint     restirHasMotion;              // (unused here; layout parity)
    uint     restirPtEnabled;              // (unused here; layout parity)
    uint     restirPtPathEnabled;          // (unused here; layout parity)
    uint     fireflyClampEnabled;          // (unused here; layout parity)
    uint     pairingPerm0;                 // (unused here; layout parity)
    uint     pairingPerm1;                 // (unused here; layout parity)
    uint     pairingPerm2;                 // (unused here; layout parity)
    uint     pairingEnabled;               // (unused here; layout parity)
    uint     reconnectShiftEnabled;        // (unused here; layout parity)
};

/// Closest-hit query for one live path record — the verbatim `traceClosestHit`
/// traversal, emitting the raw committed-hit data instead of a reconstructed
/// GiHit. `rec.tMin` / `rec.tMax` carry the ray's epsilon/infinity pair
/// (kRayEpsilon | kInf for the wave-0 camera rays; the same values the pre-split
/// call sites passed). KEEP IN SYNC with gi.comp.slang's traceClosestHit: the
/// flags, the cull mask, the commit policy, the normalize point, and the
/// degenerate-index treat-as-miss rejection are the bit contract (10d+ may
/// consolidate the traversal into a shared module — only behind the calibrated
/// gate, PLAN.md design §3).
WaveHit traceRecord(WavePath rec) {
    WaveHit hit;
    hit.barycentrics = float2(0.0, 0.0);
    hit.t = 0.0;
    hit.instanceIdx = kWaveHitMiss;
    hit.primitiveIdx = 0u;
    hit.frontFace = 0u;
    hit.pad = 0u;

    const float3 rayDir = normalize(rec.direction);
    RayDesc ray;
    ray.Origin = rec.origin;
    ray.Direction = rayDir;
    ray.TMin = rec.tMin;
    ray.TMax = rec.tMax;

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
        return hit; // miss sentinel (Q1) — the consumer applies its fallback
    }

    const uint instIdx = q.CommittedInstanceID();
    const GpuInstance inst = instances[instIdx];
    const uint tri = q.CommittedPrimitiveIndex();
    const uint base = inst.indexOffset + tri * 3u;
    if (base + 2u >= inst.indexOffset + inst.indexCount) {
        return hit; // degenerate index range: the pre-split semantic is treat-as-miss
    }
    hit.instanceIdx  = instIdx;
    hit.primitiveIdx = tri;
    hit.barycentrics = q.CommittedTriangleBarycentrics();
    hit.t            = q.CommittedRayT();
    // The front-face bit is stored per the design; the consumer reconstructs
    // normals from the vertices (the RAW outward-winding geoNormal contract) and
    // derives the side from dot(geoNormal, wo) — this bit is carried for the
    // later stages, unconsumed in 10c.
    hit.frontFace    = q.CommittedTriangleFrontFace() ? 1u : 0u;
    return hit;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= screenWidth || tid.y >= screenHeight) {
        return;
    }
    const uint pixelIndex = tid.y * screenWidth + tid.x;
    const uint slotBase = pixelIndex * kWaveSlotsPerPixel;

    // Full-width wave traversal (design §1: closest-hit for EVERY live path in
    // the wave's ray buffer). Dead slots exit on the alive flag; kinds 0..3 are
    // the pixel's adaptive-sample camera rays in 10c, kinds 4..7 stay dead
    // records (their enqueue is a later stage) and cost one flag test here.
    for (uint k = 0u; k < kWaveSlotsPerPixel; ++k) {
        const WavePath rec = t_wavePathsCur[slotBase + k];
        if (!wavePathAlive(rec)) {
            continue; // dead slot — its hit record is never read
        }
        t_waveHits[slotBase + k] = traceRecord(rec);
    }
}
