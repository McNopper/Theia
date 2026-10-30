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
// 10d WAVE LOOP: the pass now traces EVERY wave — wave 0's camera rays, then
// the wave N+1 successor records the shade dispatch regenerated into the
// ping-pong side (which half is `cur` flips per wave; selected by the
// waveIndex push constant, uniform across the dispatch). At wave 0 only kind 0
// is traced: the four adaptive-sample slots share the IDENTICAL camera ray
// (same origin/direction/tMin/tMax — the per-sample RNG crate and collect
// index are not trace inputs), so their queries are four copies of the same
// computation and the shade consumer reconstructs the primary ONCE from record
// 0's hit for all samples. The slots diverge at first shade (the BSDF/env-NEE
// draws differ per sample crate), never at trace time — confirmed in
// makeCameraWavePath: every field the query consumes is sample-independent.
// Waves >= 1 trace all live sample slots (kinds 4..7 stay dead records until
// their enqueue lands in a later stage).
//
// Push constants: the SAME cbuffer declaration as gi.comp.slang — all three
// wavefront pipelines share one pipeline layout, so this copy must stay
// field-for-field identical.

import common;       // GpuInstance, kInstanceMaskAll
import wave_common;  // WavePath / WaveHit / kWaveHitMiss + slot/flag constants

[[vk::binding(0, 0)]]  RaytracingAccelerationStructure       t_tlas;
[[vk::binding(8, 0)]]  StructuredBuffer<GpuInstance>         instances;
// 10d: ONE physical ray buffer for every wave (the regenerate is in-place —
// the shade rewrites its own pixel's slots between the wave barriers).
[[vk::binding(25, 0)]] StructuredBuffer<WavePath>            t_wavePaths;
[[vk::binding(27, 0)]] RWStructuredBuffer<WaveHit>           t_waveHits; // trace-pass output

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
    uint     waveIndex;                    // current wave (0..waveCount-1)
    uint     waveCount;                    // dispatched wave count (layout parity)
};

/// The wave's read side: which physical ray-buffer half holds this wave's
/// records (binding 25 = even waves — generate writes wave 0 there; binding 26
/// = odd waves — the shade's regenerate side). Uniform select on the push
/// constant; both bindings exist in the layout.
WavePath loadWavePath(uint slot) {
    return t_wavePaths[slot];
}

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
    // the wave's ray buffer). At wave 0 the four sample slots share one physical
    // ray — see the header note: only kind 0 is traced (the slot-0 single-trace
    // optimization), the shade reuses its hit for all samples. At waves >= 1 the
    // sample slots have diverged (each carries its own BSDF/env-NEE draws), so
    // every live kind 0..3 is traced; kinds 4..7 stay dead records until their
    // enqueue lands (later stage) and are not scanned. The alive gate is a
    // 4-byte member probe — the full 160-B record is only fetched for live rays.
    const uint kindCount = (waveIndex == 0u) ? 1u : kWaveSampleSlots;
    for (uint k = 0u; k < kindCount; ++k) {
        const uint slot = slotBase + k;
        // The alive gate as a member probe — a 4-byte read; the full 160-B
        // record is only fetched for live rays.
        if ((t_wavePaths[slot].flags & kWaveFlagAlive) == 0u) {
            continue; // dead slot — its hit record is never read
        }
        t_waveHits[slot] = traceRecord(loadWavePath(slot));
    }
}
