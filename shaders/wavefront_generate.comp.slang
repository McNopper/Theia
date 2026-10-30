// wavefront_generate.comp.slang — PERF5 stage 10b: the wavefront GENERATE pass.
//
// Stage-1 design (PLAN.md "PERF5 wavefront design (stage 10a)" §1): per frame,
// once, ahead of the trace + GI dispatches — G-buffer load (t_giBuffer worldPos /
// instance sentinel), camera-ray enqueue for every non-background pixel (the
// pixel's adaptive-sample slots, kinds 0–3), collect-buffer init. No tracing, no
// shading, and no RNG consumed beyond the wave-0 seed composition: the seeds are
// composed with the EXACT pre-split call-site expression (the sample-walk
// composeRngSeed in gi.comp.slang's main loop), so the Sobol streams cross the
// pass boundary unchanged — the RNG crate rides in the record (design §3).
//
// Stage 10c scope (orchestrator decision Q2): ALL EIGHT slots per pixel are
// initialized every frame — kinds 0..3 as camera rays (or dead records on
// background pixels), kinds 4..7 (the later stages' reservoir-candidate slots)
// as dead records — so every slot's first read is defined; the buffers are
// never cleared per frame, the alive bit is the truth. The GI dispatch reads
// its record at slot pixelIndex * kWaveSlotsPerPixel + sampleIndex; the trace
// pass consumes the same records. The ReSTIR reservoir passes are untouched
// (they keep their own seeds and G-buffer loads inside gi.comp).
//
// Stage 10d: this pass runs ONCE per frame (wave 0 only) — the shade dispatch
// regenerates waves 1..maxDepth into the ping-pong side (binding 26) itself.
// The collect init (sampleEnvNee chain added) is CONSUMED by the 10d shade
// waves + composite.
//
// Push constants: the SAME cbuffer declaration as gi.comp.slang — all wavefront
// pipelines share one pipeline layout, so this copy must stay field-for-field
// identical.

import common;       // kRayEpsilon, kInf (Harmonia gpu_types, re-exported)
import rng;          // composeRngSeed
import wave_common;  // WavePath / WaveCollect + slot/flag constants

[[vk::binding(2, 0)]]  Texture2D<float4>                t_giBuffer;     // xyz = worldPos, w = asfloat(instanceIdx+1)
[[vk::binding(25, 0)]] RWStructuredBuffer<WavePath>     t_wavePathsCur; // wave-0 write (the GI dispatch reads this binding)
[[vk::binding(28, 0)]] RWStructuredBuffer<WaveCollect>  t_waveCollect;  // per-pixel collect init (consumed from 10d)

[[vk::push_constant]]
cbuffer GiPC {
    float4x4 view;                  // view matrix (row-major; mul(M, v) convention — matches -matrix-layout-row-major)
    float4x4 prevViewProj;          // GI2 Phase 3: previous-frame view-projection (unused here; layout parity with gi.comp)
    float4   cameraPos;             // xyz = world-space camera position
    float    exposure;              // 1 / (1.2 * 2^EV100) (unused here; layout parity)
    uint     frameSampleIndex;      // per-frame sample counter (drives RNG decorrelation across accumulated frames)
    uint     rngBaseSeed;           // base seed for composeRngSeed()
    uint     emissiveTriangleCount; // (unused here; layout parity)
    uint     envImportanceWidth;    // (unused here; layout parity)
    uint     envImportanceHeight;   // (unused here; layout parity)
    uint     hasEnvMap;             // (unused here; layout parity)
    float    envLuminanceScale;     // (unused here; layout parity)
    uint     maxDepth;              // maximum number of traced bounce segments (wave-0 budget)
    uint     screenWidth;
    uint     screenHeight;
    uint     _pad0;
    uint     a3RegularizationEnabled;      // (unused here; layout parity)
    uint     hasGradientVariance;          // A3(b): 1 when t_gradientVariance holds a real A-SVGF guide
    uint     giAdaptiveMaxSamples;         // c1: max GI samples/pixel (1 = off) — the seed stride
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
    uint     waveIndex;                    // (unused here — generate IS wave 0; layout parity)
    uint     waveCount;                    // (unused here; layout parity)
};

/// Wave-0 record for one adaptive-sample slot: the camera ray plus the sample's
/// RNG crate, seeded with the pre-split call-site semantics. Throughput = 1 and
/// radiance = 0 — the megakernel's state initialization — and the medium fields
/// are zero (the 10d shade consumes the record's medium fields per wave; the
/// walk still starts and completes inside one shade invocation, so no wave ever
/// reads a live insideMedium record — the fields are state carriers, not a
/// suspension point).
WavePath makeCameraWavePath(uint2 px, uint pixelIndex, uint sampleIndex, float3 cameraDelta, uint maxSamples) {
    WavePath rec;
    rec.origin          = cameraPos.xyz;
    // EXACT pre-split expression: gi.comp.slang traced `worldPos - cameraPos` and
    // traceClosestHit normalizes internally, so the record carries the same
    // unnormalized delta — the consumer's FP is bit-identical to the inline path.
    rec.direction       = cameraDelta;
    rec.throughput      = float3(1.0, 1.0, 1.0);
    rec.radiance        = float3(0.0, 0.0, 0.0);
    rec.mediumSigmaT    = float3(0.0, 0.0, 0.0); // medium fields: state carriers (10d)
    rec.mediumAlbedo    = float3(0.0, 0.0, 0.0);
    rec.lastBsdfPdf     = 0.0;                   // no BSDF sampled yet (megakernel init)
    rec.tMin            = kRayEpsilon;           // the camera re-trace's self-intersection guard
    rec.tMax            = kInf;
    // The megakernel budget pre-charges the primary vertex (primaryCharge) — but
    // that charge is decided by the opacity pass-through draw, which happens in
    // shade. Wave-0 carries the raw budget; the first shade applies the charge.
    rec.bouncesRemaining = maxDepth;
    rec.iters           = 0u;                    // null-interaction budget (consumed by a later stage)
    rec.depth           = 0u;                    // absolute surface depth at the camera
    rec.pixelIndex      = pixelIndex;
    rec.sampleIndex     = sampleIndex;
    // The RNG crate: composed exactly where the pre-split GI loop composed it.
    const RngState seed = composeRngSeed(px, frameSampleIndex * maxSamples + sampleIndex, 0u, rngBaseSeed);
    rec.rng             = uint3(seed.key, seed.index, seed.dim);
    rec.mediumG         = 0.0;
    rec.mediumEta       = 1.0;
    rec.mediumWalkIters = 0u;
    rec.flags           = kWaveFlagAlive;
    return rec;
}

// Dead slots are flags-only kills (waveSlotAlive probes read nothing else) —
// there is no dead-record builder anymore.

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= screenWidth || tid.y >= screenHeight) {
        return;
    }
    const uint2 px = tid.xy;
    const uint pixelIndex = px.y * screenWidth + px.x;
    const uint slotBase = pixelIndex * kWaveSlotsPerPixel;
    // Collect init (design §1): the per-pixel termination/composite record starts
    // empty every frame. CONSUMED by the 10d shade waves + composite (the
    // composite's activity gate is sampleCount == 0 — this init IS what marks
    // background pixels inactive).
    WaveCollect col;
    col.sampleIndirect[0]  = float3(0.0, 0.0, 0.0);
    col.sampleIndirect[1]  = float3(0.0, 0.0, 0.0);
    col.sampleIndirect[2]  = float3(0.0, 0.0, 0.0);
    col.sampleIndirect[3]  = float3(0.0, 0.0, 0.0);
    col.sampleEnvNee[0]    = float3(0.0, 0.0, 0.0);
    col.sampleEnvNee[1]    = float3(0.0, 0.0, 0.0);
    col.sampleEnvNee[2]    = float3(0.0, 0.0, 0.0);
    col.sampleEnvNee[3]    = float3(0.0, 0.0, 0.0);
    col.restirDirect       = float3(0.0, 0.0, 0.0);
    col.restirPathIndirect = float3(0.0, 0.0, 0.0);
    col.restirM            = 0.0;
    col.sampleCount        = 0u;
    col.flags              = 0u;
    col.pad                = 0u;
    t_waveCollect[pixelIndex] = col;

    // --- Primary surface probe from the forward G-buffer (moved verbatim from
    //     gi.comp.slang's entry) ---
    const float4 giSample = t_giBuffer.Load(int3(int(px.x), int(px.y), 0));
    // w = asfloat(instanceIdx+1): 0 means background / sky (A1b: changed from materialIdx+1).
    const uint instIdxPlus1 = asuint(giSample.w);
    if (instIdxPlus1 == 0u) {
        // Background / non-geometry pixel (and transparent pixels keep the
        // sentinel): kill ALL eight slots. A flags-only member store suffices —
        // no consumer of a dead record reads beyond the alive bit (the shade
        // exits on kind 0's alive probe; kinds 4..7 are probed by no stage this
        // one enqueues), and at 160 B per record the full dead-record writes
        // would be the generate pass's dominant cost on unified memory.
        for (uint k = 0u; k < kWaveSlotsPerPixel; ++k) {
            t_wavePathsCur[slotBase + k].flags = 0u;
        }
        return;
    }

    // maxSamples EXACTLY as gi.comp.slang's sample loop computes it — the seed
    // stride (frameSampleIndex * maxSamples) must match bit for bit. Q3 clamp:
    // bounded by the pixel's sample-slot window (kWaveSampleSlots = 4) so the
    // enqueued records and the shade-side consumer can never disagree about the
    // window (a push-constant value above 4 would read into the next pixel's
    // slots). No-op for every current configuration (giAdaptiveMaxSamples ≤ 4).
    const uint maxSamples = (hasGradientVariance != 0u)
                                ? max(min(giAdaptiveMaxSamples, kWaveSampleSlots), 1u)
                                : 1u;
    const float3 cameraDelta = giSample.xyz - cameraPos.xyz;
    // Camera-ray enqueue for ALL four sample slots (fixed per-pixel slots, design
    // §2): slot s's record is consumed by the GI dispatch for sample s; the slots
    // beyond this pixel's sampleCount stay enqueued-but-unread this frame. Kinds
    // 4..7 (reservoir-candidate slots of the later stages) get flags-only kills —
    // their enqueue is a later stage, and no consumer of a dead record reads
    // beyond the alive bit.
    for (uint s = 0u; s < kWaveSampleSlots; ++s) {
        t_wavePathsCur[slotBase + s] = makeCameraWavePath(px, pixelIndex, s, cameraDelta, maxSamples);
    }
    for (uint k = kWaveSampleSlots; k < kWaveSlotsPerPixel; ++k) {
        t_wavePathsCur[slotBase + k].flags = 0u;
    }
}
