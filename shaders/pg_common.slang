#ifndef PG_COMMON_SLANG
#define PG_COMMON_SLANG

// ── GI-PG (ReSTIR PG, SIGGRAPH Asia 2025) — shared guiding-grid definitions ───
// Consumed by pg_fit.comp.slang (S2: splat E-step + M-step fit) and, from S3 on,
// gi.comp.slang (guided candidate sampling).
//
// Determinism + consistency design (dependency audit, Harmonia/PLAN.md GI-PG row):
//  * Keyless direct-mapped buckets: slot = hash(pos-cell, materialIdx) & mask.
//    A hash collision MIXES two cells' directions (quality-only effect) — never a
//    correctness/consistency issue, because the fitted distribution is a pure
//    PROPOSAL: one-sample MIS only requires that sampling and evaluation use the
//    same parameters, which holds by construction (fit runs at end of frame N,
//    sampling reads the frozen params during frame N+1 — no in-frame mutation).
//  * Fixed-point integer atomicAdd for the E-step statistics: integer addition is
//    order-independent, so splat accumulation is deterministic run-to-run.
//  * The M-step is a single writer per slot (deterministic given the stats).
//  * No eviction/claim races exist by design (buckets never change identity).

static const uint kPgSlots = 1u << 19; ///< 524288 buckets (MUST match GiPass.hpp)
static const uint kPgLobes = 4u;       ///< vMF components per cell (paper: 4)
static const float kPgCellBase = 0.5;  ///< world meters at the reference distance
static const float kPgDistScale = 0.25; ///< cellSize = base * max(1, dist*scale):
                                       ///< coarser buckets with viewing distance
                                       ///< ("resolution adapts with viewing distance")
static const float kPgFixedScale = 1024.0;  ///< fixed-point scale for atomic stats
static const uint kPgMinSamples = 8u;       ///< min samples/frame before a cell is valid
static const float kPgKappaMin = 0.01;      ///< concentration clamp (near-uniform floor)
static const float kPgKappaMax = 64.0;      ///< concentration clamp (~sharp lobe cap)

// Per-slot buffer strides.
static const uint kPgCellFloat4 = 6u; // cells: [muKappa0..3] + (pi0..3) + (valid,...)
static const uint kPgStatInts = 20u;  // stats: 4 lobes * int4 + count + pad = 5 int4

// S1 selected-path chain layout (flat float4; written by gi.comp's candidate walk,
// read by splatMain). MUST match GiPass.hpp kPgChainSlots.
static const uint kPgChainDepth = 10u;               // chain cap (scenes run <= 9 bounces)
static const uint kPgChainSlots = 1u + 3u * kPgChainDepth; // 31 x 16 B = 496 B/pixel

/// Direct-mapped bucket for a chain vertex (position + material/"BSDF ID" encoded —
/// paper §4.2: the hash explicitly encodes the BSDF ID so same-BSDF cells group).
uint pgCellSlot(float3 pos, float3 camPos, uint materialIdx) {
    const float dist = length(pos - camPos);
    const float cell = kPgCellBase * max(1.0, dist * kPgDistScale);
    const int3 c = int3(floor(pos / cell));
    uint h = (uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u) ^
             ((materialIdx + 1u) * 2246822519u);
    return h & (kPgSlots - 1u);
}

/// Unnormalized vMF direction density exp(kappa * mu·omega) (the kappa/(4pi sinh
/// kappa) factor cancels inside the E-step's responsibility normalization).
/// Exponent clamped to ±80 (fp32-safe; kappa <= kPgKappaMax=64 needs only ±64) —
/// a tighter ±40 clamp truncated high-kappa responsibilities (see pgGuidePdf).
float pgVmfExp(float3 mu, float kappa, float3 omega) {
    return exp(clamp(kappa * dot(mu, omega), -80.0, 80.0));
}

#endif // PG_COMMON_SLANG
