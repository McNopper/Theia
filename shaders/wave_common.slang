// wave_common.slang — PERF5 wavefront path-record vocabulary.
//
// The record structs are the binding spec from the stage-1 design (PLAN.md
// "PERF5 wavefront design (stage 10a)", §2 buffer layout); every wavefront pass
// shares them — the generate pass writes the wave-0 records
// (wavefront_generate.comp.slang), the GI dispatch consumes them
// (gi.comp.slang), and the later stages (10c trace/shade, 10d shadow/medium
// gather, 10e compaction) build on the same layout. The byte math is pinned by
// the design (Slang float3 = 16 B stride/align in structured buffers, std430):
//   WavePath    156 B → 160 B (16-aligned)
//   WaveHit      28 B →  32 B (8-aligned)
//   WaveCollect       → 112 B
// Pure data on purpose: no imports, so any entry shader can import this module
// without pulling the estimator DAG.
//
// NOTE for later stages: changing ANY struct member re-layouts the GPU buffers
// sized in GiPass (kWavePathStride / kWaveHitStride / kWaveCollectStride) —
// keep the design's byte math and the C++ strides in lockstep.

// ── Fixed per-pixel slot window (design §2 "Regeneration queue") ─────────────
// Stage 1 has no separate queue: slots are fixed per pixel (no allocation
// atomics — a deterministic index space), the ray buffer ping-pongs per wave,
// and the buffers are never cleared per frame — the alive bit is the truth.
//   kinds 0..3 = the pixel's adaptive-sample paths (wave-0 camera rays; c1
//                maxSamples ≤ 4 — the same bound WaveCollect's 4 slots pin)
//   kind 4     = the path-reservoir local candidate (queued in 10c)
//   kinds 5..7 = the three spatial-replay candidates (queued in 10c)
static const uint kWaveSlotsPerPixel = 8u;
static const uint kWaveSampleSlots = 4u; // kinds 0..3: the c1 adaptive samples
static const uint kWaveSlotLocalCandidate = 4u; // 10c: reservoir local candidate
static const uint kWaveSlotReplay0 = 5u; // 10c: first spatial-replay candidate

// WavePath.flags bits (design §2).
static const uint kWaveFlagAlive = (1u << 0); // live path — the dispatch consumes the slot
static const uint kWaveFlagTerminated = (1u << 1); // path ended; radiance is final (collect write, 10d)
static const uint kWaveFlagPassedThrough = (1u << 2); // OpenPBR opacity null interaction (ray resumes)
static const uint kWaveFlagSpecularBounce = (1u << 3); // last bounce was a delta lobe (MIS input)
static const uint kWaveFlagInsideMedium = (1u << 4); // suspended inside a volumetric walk (10d)

/// One suspended path segment (the megakernel loop state). The RNG crate rides
/// in the record — seeds are composed once (at the call site that owns them) and
/// `dim` advances exactly as the megakernel consumes draws, so no draw is added,
/// removed, or reordered by the pass split (design §3).
struct WavePath {
    float3 origin;          // 16  suspended ray origin
    float3 direction;       // 16  suspended ray direction
    float3 throughput;      // 16
    float3 radiance;        // 16  accumulated IN the record — per-path FP order preserved
    float3 mediumSigmaT;    // 16  medium-walk suspend/resume (10d; written at BSDF medium entry)
    float3 mediumAlbedo;    // 16
    float  lastBsdfPdf;     // 4   MIS input
    float  tMin;            // 4   kRayEpsilon | kNullInteractionEpsilon | kMediumEpsilon
    float  tMax;            // 4   kInf for surface bounces; free-flight dist for medium segments (10d)
    uint   bouncesRemaining; // 4  megakernel `bounce` budget (primaryCharge pre-applied)
    uint   iters;           // 4   null-interaction guard (kMaxNullInteractions budget)
    uint   depth;           // 4   absolute surface depth (evalEmissionGuarded / kRRMinDepth)
    uint   pixelIndex;      // 4
    uint   sampleIndex;     // 4   c1 adaptive sample slot (collect index)
    uint3  rng;             // 12  RngState{key,index,dim} — THE RNG CRATE rides in the record
    float  mediumG;         // 4   medium-walk suspend/resume
    float  mediumEta;       // 4
    uint   mediumWalkIters; // 4   kMaxMediumWalk budget consumed this entry
    uint   flags;           // 4   kWaveFlag* bits — bit0 alive is the slot's truth
};                          // 156 B → 160 B (16-aligned)

/// Trace-pass output, same index space as WavePath (10c; allocated in 10b).
struct WaveHit {
    float2 barycentrics; // 8
    float  t;            // 4   CommittedRayT (medium boundary transmittance needs it)
    uint   instanceIdx;  // 4   0xFFFFFFFF = miss
    uint   primitiveIdx; // 4
    uint   frontFace;    // 4   committed-hit front-face bit
    uint   pad;          // 4
};                       // 32 B

/// Per-pixel termination + composite inputs (initialized by the generate pass;
/// consumed from 10d — stage 10b keeps the existing HDR accumulation path).
struct WaveCollect {
    float3 sampleIndirect[4]; // 64  one slot per c1 adaptive sample (maxSamples = 4)
    float3 restirDirect;      // 16
    float3 restirPathIndirect; // 16
    float  restirM;           // 4   M-aware denoiser alpha
    uint   sampleCount;       // 4
    uint   pad[2];            // 8
};                            // 112 B

/// The slot's truth: buffers are never cleared per frame, so the alive bit —
/// not the buffer contents' age — decides whether a dispatch consumes the slot.
bool wavePathAlive(WavePath p) {
    return (p.flags & kWaveFlagAlive) != 0u;
}
