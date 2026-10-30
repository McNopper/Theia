// wave_common.slang — PERF5 wavefront path-record vocabulary.
//
// The record structs are the binding spec from the stage-1 design (PLAN.md
// "PERF5 wavefront design (stage 10a)", §2 buffer layout); every wavefront pass
// shares them — the generate pass writes the wave-0 records
// (wavefront_generate.comp.slang), the trace pass consumes them and produces the
// hit records (wavefront_trace.comp.slang), the GI dispatch consumes record +
// hit and (since 10d) writes the wave N+1 successor records back (the
// regenerate loop), and the later stages (10e compaction) build on the same
// layout. The byte math is pinned by the design (Slang float3 = 16 B stride/align
// in structured buffers, std430):
//   WavePath    156 B → 160 B (16-aligned)
//   WaveHit      28 B →  32 B (8-aligned)
//   WaveCollect       → 176 B (10d: was 112 B — see the struct comment)
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
static const uint kWaveFlagTerminated = (1u << 1); // path ended; radiance is final (collect write — 10d)
static const uint kWaveFlagPassedThrough = (1u << 2); // OpenPBR opacity null interaction (ray resumes)
static const uint kWaveFlagSpecularBounce = (1u << 3); // last bounce was a delta lobe (MIS input)
static const uint kWaveFlagInsideMedium = (1u << 4); // carried medium state (10d: consumed, never suspended)

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

/// Trace-pass output, same index space as WavePath (written by the 10c trace
/// pass, consumed by the shade-side GI dispatch).
struct WaveHit {
    float2 barycentrics; // 8
    float  t;            // 4   CommittedRayT (medium boundary transmittance needs it)
    uint   instanceIdx;  // 4   kWaveHitMiss = miss
    uint   primitiveIdx; // 4
    uint   frontFace;    // 4   committed-hit front-face bit
    uint   pad;          // 4
};                       // 32 B

/// WaveHit.instanceIdx miss sentinel (design §2): the trace pass writes it for
/// rays that miss (or hit a degenerate-index triangle — the pre-split
/// treat-as-miss semantic); the consuming dispatch then applies its fallback
/// (wired in 10c: the wave-0 primary's Q1 G-buffer fallback in gi.comp.slang).
static const uint kWaveHitMiss = 0xFFFFFFFFu;

/// Per-pixel termination + composite inputs (initialized by the generate pass;
/// consumed by the 10d shade waves + composite). 10d AMENDMENT (design §2
/// struct, sizes restated): the design's single `sampleIndirect[4]` array cannot
/// express the composite's TWO summation chains bit-exactly — the megakernel
/// accumulates `sumIndirect` and `sumEnvNee` as two separate left-associated
/// chains (per-sample `+=` in s order) and the final expression adds the two
/// chain totals; folding the per-sample env NEE into the same slot (or into one
/// combined per-sample value) reassociates those additions. The collect
/// therefore carries both per-sample chains, and the design's `pad` hosts the
/// composite's activity/validity flags:
///   sampleIndirect[s] — the sample's finalized indirect (the per-path radiance
///                       chain of record s, finalized at its terminating wave,
///                       today's estimateGiSampleCore epilogue verbatim);
///   sampleEnvNee[s]   — the sample's primary env NEE (exposure-applied,
///                       guarded; written once at wave 0 — the whole chain is a
///                       primary-vertex term and never crosses waves).
/// The composite sums each chain in s order and applies today's final
/// expression — FP-identical to the megakernel's loop.
struct WaveCollect {
    float3 sampleIndirect[4];  // 64  one slot per c1 adaptive sample (maxSamples = 4)
    float3 sampleEnvNee[4];    // 64  per-sample primary env NEE (wave-0 write)
    float3 restirDirect;       // 16
    float3 restirPathIndirect; // 16
    float  restirM;            // 4   M-aware denoiser alpha
    uint   sampleCount;        // 4   composite activity gate: 0 = pixel inactive (no composite)
    uint   flags;              // 4   kWaveCollectFlag* bits
    uint   pad;                // 4
};                             // 176 B (16-aligned)

/// WaveCollect.flags: the sample degenerated and maxSamples == 1 — today's
/// `maxSamples == 1u && !valid` early return, where the pixel's indirect AND
/// env-NEE terms are discarded entirely (the composite applies the legacy
/// restirDirect-only write instead of the final expression).
static const uint kWaveCollectFlagSingleInvalid = (1u << 0);

// The alive gate lives at the consumers as a 4-byte MEMBER PROBE on the buffer
// (`t_wavePaths[slot].flags` in gi.comp / wavefront_trace) — loading the whole
// 160-B record just to test one bit is the most expensive no-op of the
// fixed-width wave dispatches, so no whole-record `wavePathAlive` helper is
// provided.
