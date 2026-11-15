// pg_fit.comp.slang — GI-PG stage S2: splat (EM E-step) + fit (M-step).
//
// splatMain: every pixel reads its selected-path chain (S1, binding 0), computes
// the E-step responsibilities of each incident direction against the cell's CURRENT
// mixture (previous frame's fit) and atomically accumulates fixed-point sufficient
// statistics into the direct-mapped bucket (binding 2).
// fitMain:   every bucket converts its statistics into updated 4-vMF parameters
// (M-step: mu = normalize(r), kappa from the mean resultant length, pi = w/N) into
// the persistent cells buffer (binding 1), validity-gated on kPgMinSamples.
// S3 (guided candidate sampling in gi.comp) consumes the cells NEXT frame.
//
// Runs only while pathGuidingEnabled (dispatched from GiPass::record).

#include "pg_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<float4> t_pgChainData; // S1 per-pixel chains
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> t_pgCells;   // persistent params (float4 stride)
[[vk::binding(2, 0)]] RWStructuredBuffer<int> t_pgStats;      // per-frame fixed-point stats (int stride)

[[vk::push_constant]]
cbuffer PGPC {
    uint  screenWidth;
    uint  screenHeight;
    float4 cameraPos; // xyz = camera position (view-distance-adaptive cell size)
}

/// Splat + E-step. One thread per pixel: loop this pixel's chain vertices.
[shader("compute")]
[numthreads(8, 8, 1)]
void splatMain(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= uint(screenWidth) || tid.y >= uint(screenHeight)) {
        return;
    }
    const uint pixelIdx = tid.y * uint(screenWidth) + tid.x;
    const uint base = pixelIdx * kPgChainSlots; // S1 stride (pg_common.slang)
    const float rawCount = t_pgChainData[base].x;
    // Defensive: stale/garbage headers (walks that early-returned) are clamped.
    const uint count = min(uint(max(rawCount, 0.0)), kPgChainDepth);

    for (uint i = 0u; i < count; ++i) {
        const float4 posId = t_pgChainData[base + 1u + 3u * i];
        const float4 dir4 = t_pgChainData[base + 2u + 3u * i];
        const float3 dir = dir4.xyz;
        // Terminal vertices keep dir = 0 (S1 discipline) — no direction to learn.
        if (dot(dir, dir) < 1.0e-8) {
            continue;
        }
        const float3 pos = posId.xyz;
        const uint matIdx = asuint(posId.w);
        const uint slot = pgCellSlot(pos, cameraPos.xyz, matIdx);
        const uint sbase = slot * kPgStatInts;

        // E-step: responsibilities gamma_k ∝ pi_k * exp(kappa_k * mu_k·omega).
        // Zero-initialized cells bootstrap with the uniform limit (1/4 each) —
        // the paper's "no additional warm-up phase required".
        const uint cbase = slot * kPgCellFloat4;
        float sum = 0.0;
        float g[kPgLobes];
        [unroll]
        for (uint k = 0u; k < kPgLobes; ++k) {
            const float4 mk = t_pgCells[cbase + k];
            const float pi = t_pgCells[cbase + 4u][k];
            const float e = (pi > 0.0) ? pi * pgVmfExp(mk.xyz, mk.w, dir) : 0.0;
            g[k] = e;
            sum += e;
        }
        [unroll]
        for (uint k = 0u; k < kPgLobes; ++k) {
            g[k] = (sum > 0.0) ? g[k] / sum : 0.25;
        }

        // Fixed-point E-step accumulation (integer atomicAdd = order-independent).
        [unroll]
        for (uint k = 0u; k < kPgLobes; ++k) {
            const int wg = int(g[k] * kPgFixedScale);
            const int wr = int(g[k] * dir.x * kPgFixedScale);
            const int wb = int(g[k] * dir.y * kPgFixedScale);
            const int wc = int(g[k] * dir.z * kPgFixedScale);
            int dst;
            InterlockedAdd(t_pgStats[sbase + k * 4u + 0u], wr, dst);
            InterlockedAdd(t_pgStats[sbase + k * 4u + 1u], wb, dst);
            InterlockedAdd(t_pgStats[sbase + k * 4u + 2u], wc, dst);
            InterlockedAdd(t_pgStats[sbase + k * 4u + 3u], wg, dst);
        }
        int nDst;
        InterlockedAdd(t_pgStats[sbase + 16u], 1, nDst);
    }
}

/// Fit + M-step. One thread per bucket; single writer (deterministic given stats).
[shader("compute")]
[numthreads(64, 1, 1)]
void fitMain(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= kPgSlots) {
        return;
    }
    const uint slot = tid.x;
    const uint sbase = slot * kPgStatInts;
    const uint cbase = slot * kPgCellFloat4;
    const int rawCount = t_pgStats[sbase + 16u];
    if (rawCount < int(kPgMinSamples)) {
        // Too little traffic this frame: previous params stay (if any) but the cell
        // is marked invalid — S3 falls back to pure BSDF sampling for it (the
        // one-sample MIS keeps consistency regardless).
        t_pgCells[cbase + 5u] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    const float n = float(rawCount);
    float sumPi = 0.0;
    [unroll]
    for (uint k = 0u; k < kPgLobes; ++k) {
        const float rX = float(t_pgStats[sbase + k * 4u + 0u]) / kPgFixedScale;
        const float rY = float(t_pgStats[sbase + k * 4u + 1u]) / kPgFixedScale;
        const float rZ = float(t_pgStats[sbase + k * 4u + 2u]) / kPgFixedScale;
        const float w = float(t_pgStats[sbase + k * 4u + 3u]) / kPgFixedScale;

        const float rLen = length(float3(rX, rY, rZ));
        const float rBar = clamp(rLen / max(n, 1.0), 0.0, 0.999);

        const float4 prev = t_pgCells[cbase + k];
        float3 mu = prev.xyz;
        if (rLen > 1.0e-5) {
            mu = float3(rX, rY, rZ) / rLen;
        }
        // Standard vMF M-step for d = 3 (with regularization/clamping per the paper:
        // "appropriate regularization and clamping ... similar to Ruppert et al. 2020").
        const float kappa = clamp(rBar * (3.0 - rBar * rBar) / max(1.0 - rBar * rBar, 1.0e-6),
                                  kPgKappaMin, kPgKappaMax);
        t_pgCells[cbase + k] = float4(mu, kappa);

        // Park the unnormalized weight in the pi slot (read-modify-write of the full
        // float4 — a component lvalue through the buffer binding is not portable).
        float4 piVec = t_pgCells[cbase + 4u];
        piVec[k] = max(w, 0.0);
        t_pgCells[cbase + 4u] = piVec;
        sumPi += max(w, 0.0);
    }
    // Normalize pi in place (single thread, safe) and publish validity.
    if (sumPi > 0.0) {
        const float4 piUnnorm = t_pgCells[cbase + 4u];
        t_pgCells[cbase + 4u] = piUnnorm / sumPi;
    } else {
        t_pgCells[cbase + 4u] = float4(0.25, 0.25, 0.25, 0.25);
    }
    t_pgCells[cbase + 5u] = float4(1.0, n, 0.0, 0.0); // x = valid, y = samples this frame
}

