// Precomputes the Lambertian (diffuse) irradiance map E(N) for every normal
// direction N in the equirectangular output atlas.
//
// Unlike simple cosine-weighted hemisphere sampling, this shader uses the same
// 2D separable CDF env-importance-sampling technique as Hyperion's path tracer
// (PBR Book 4th ed. §12.5).  That is essential for high-dynamic-range env maps
// whose peak/mean ratio can exceed 10,000× (e.g. an outdoor panorama with a
// visible sun disc): cosine-weighted sampling would almost never hit the sun and
// produce a heavily biased (too dark, wrong colour) irradiance estimate.
//
// MC estimator:
//   E(N) = integral_sphere[ Li(w) * max(0, dot(N,w)) dw ]
//        ≈ (1/S) * Σ_i  Li(wi) * max(0, dot(N,wi)) / pdf_env(wi)
//
// pdf_env is the solid-angle PDF of the prebuilt 2D CDF — proportional to
// luminance*sinTheta so bright (high-energy) directions are sampled more
// frequently.  Half of the importance samples fall below the horizon for any
// given N; they contribute 0 via max(0, dot), so no wasted energy enters the
// estimate.  The factor of N in the denominator naturally accounts for this
// without any reweighting.
//
// Ref: Pharr, Jakob & Humphreys — "Physically Based Rendering" 4th ed. §12.5,
//      §13.4.3 (MIS balance heuristic); Veach — "Robust Monte Carlo Methods" §9.2.

static const float kPi    = 3.14159265359;
static const float kInvPi = 0.31830988618;

// Set 0:
// 0 = env panorama (Texture2D RGBA32F, working colour space)
// 1 = env sampler  (REPEAT in U, CLAMP_TO_EDGE in V)
// 2 = output irradiance atlas (RWTexture2D RGBA16F, same layout as the panorama)
// 3 = marginal CDF over rows    — (H+1) float32 values, normalised
// 4 = conditional CDF per row   — H*(W+1) float32 values, normalised
[[vk::binding(0, 0)]] Texture2D<float4>          envMap;
[[vk::binding(1, 0)]] SamplerState               envSampler;
[[vk::binding(2, 0)]] RWTexture2D<float4>        outIrrad;
[[vk::binding(3, 0)]] StructuredBuffer<float>    marginalCdf;
[[vk::binding(4, 0)]] StructuredBuffer<float>    conditionalCdf;

[[vk::push_constant]]
cbuffer PC {
    float    envScale;  ///< env_unit_nits — physical cd/m² per raw EXR unit
    uint     cdfWidth;  ///< CDF grid width  (same W used when the CDF was built)
    uint     cdfHeight; ///< CDF grid height (same H)
    uint     _pad;
};

// ── Hammersley low-discrepancy sequence ────────────────────────────────────────
float radicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
float2 hammersley2d(uint i, uint N)
{
    return float2(float(i) / float(N), radicalInverse_VdC(i));
}

// ── Lat-long ↔ direction (Hyperion env.slang convention) ───────────────────────
float2 dirToLatlong(float3 d)
{
    return float2(atan2(d.z, d.x) * (kInvPi * 0.5) + 0.5,
                  acos(clamp(d.y, -1.0, 1.0)) * kInvPi);
}

// ── 2D separable CDF importance sampling (mirrors Hyperion env.slang) ─────────
/// Binary search: largest i ∈ [0,n-1] with cdf[base+i+1] <= r.
uint sampleCdf1D(StructuredBuffer<float> cdf, uint base, uint n, float r)
{
    uint lo = 0u, hi = n - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (cdf[base + mid + 1u] <= r) lo = mid + 1u;
        else                            hi = mid;
    }
    return lo;
}

/// Sample a world-space direction via the env-map luminance CDF.
/// Returns the direction and its solid-angle PDF.
float3 sampleEnvImportance(float2 xi, out float pdfOmega)
{
    const uint W = cdfWidth;
    const uint H = cdfHeight;

    // Sample row via marginal CDF
    uint row = sampleCdf1D(marginalCdf, 0u, H, xi.x);
    // Sample column via conditional CDF for that row
    uint col = sampleCdf1D(conditionalCdf, row * (W + 1u), W, xi.y);

    // Continuous inversion within the selected bin
    float rowLo = marginalCdf[row];
    float rowHi = marginalCdf[row + 1u];
    float tRow  = (rowHi > rowLo) ? clamp((xi.x - rowLo) / (rowHi - rowLo), 0.0, 1.0) : 0.5;

    uint  cBase = row * (W + 1u);
    float colLo = conditionalCdf[cBase + col];
    float colHi = conditionalCdf[cBase + col + 1u];
    float tCol  = (colHi > colLo) ? clamp((xi.y - colLo) / (colHi - colLo), 0.0, 1.0) : 0.5;

    float uFrac = (float(col) + tCol) / float(W);
    float vFrac = (float(row) + tRow) / float(H);

    // Solid-angle PDF:  p_UV * W * H / (2 * pi^2 * sinTheta)
    float pRow     = rowHi - rowLo;
    float pCol     = colHi - colLo;
    float theta    = kPi * vFrac;
    float sinTheta = max(sin(theta), 1.0e-5);
    pdfOmega = (pRow * pCol * float(W) * float(H)) / (2.0 * kPi * kPi * sinTheta);

    // Reconstruct direction (same as Hyperion env.slang)
    float phi = 2.0 * kPi * (uFrac - 0.5);
    float3 dir;
    dir.y = cos(theta);
    dir.x = sinTheta * cos(phi);
    dir.z = sinTheta * sin(phi);
    return normalize(dir);
}

// ── Main ───────────────────────────────────────────────────────────────────────
[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= 256 || tid.y >= 128) {
        return;
    }

    // Output texel → surface normal direction for which we integrate irradiance.
    float2 uvN  = (float2(tid.xy) + 0.5) / float2(256.0, 128.0);
    float  phi0 = (uvN.x - 0.5) * (2.0 * kPi);
    float  th0  = uvN.y * kPi;
    float  sinT = sin(th0);
    float3 N    = normalize(float3(sinT * cos(phi0), cos(th0), sinT * sin(phi0)));

    // Importance-sample the env map over the full sphere; the hemisphere
    // constraint is enforced by the max(0, dot(N, wi)) factor.
    const uint sampleCount = 1024u;
    float3 irradiance = 0.0.xxx;

    for (uint i = 0u; i < sampleCount; ++i) {
        float  pdfOmega;
        float3 wi      = sampleEnvImportance(hammersley2d(i, sampleCount), pdfOmega);
        float  cosTheta = dot(N, wi);
        if (cosTheta > 0.0 && pdfOmega > 0.0) {
            float3 Li = max(envMap.SampleLevel(envSampler, dirToLatlong(wi), 0.0).xyz, 0.0.xxx);
            irradiance += Li * cosTheta / pdfOmega;
        }
    }

    // Divide by sample count (MC average) and apply the physical scale.
    outIrrad[tid.xy] = float4(irradiance / float(sampleCount) * envScale, 1.0);
}
