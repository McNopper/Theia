// taa.comp.slang — TAA temporal anti-aliasing compute pass.
//
// Reads: current HDR (GENERAL, sampled), motion vectors (GENERAL, sampled),
//        TAA history (GENERAL, sampled).
// Writes: TAA output buffer (GENERAL, storage) — copied back into the HDR image
//         after dispatch so the Harmonia denoiser receives a temporally stable input.
//
// Algorithm: YCoCg-space 3x3 neighbourhood AABB clamping + alpha=0.1 temporal blend.
// Motion vector convention: mv = prevPixel - curPixel  (produced by MotionVectorPass A1b).

[[vk::push_constant]]
cbuffer TaaPC {
    uint  screenWidth;
    uint  screenHeight;
    float alpha;      // current-frame blend weight (default 0.1: 10% current, 90% history)
    uint  firstFrame; // 1 → write current directly and skip history blend
};

[[vk::binding(0, 0)]] Texture2D<float4>   t_hdrCurrent;    // current-frame HDR (GI accumulated)
[[vk::binding(1, 0)]] Texture2D<float2>   t_motionVectors; // R=dx, G=dy pixel-space (prevPx - curPx)
[[vk::binding(2, 0)]] Texture2D<float4>   t_history;       // previous frame's TAA output
[[vk::binding(3, 0)]] [vk::image_format("rgba32f")] RWTexture2D<float4> t_output;

// --- YCoCg colour-space helpers ---
float3 rgbToYCoCg(float3 c) {
    return float3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r              - 0.5  * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
float3 YCoCgToRgb(float3 y) {
    float t = y.x - y.z;
    return float3(t + y.y, y.x + y.z, t - y.y);
}

// Build 3x3 neighbourhood AABB in YCoCg space to clamp history (suppresses ghosting).
void neighbourhoodAABB(int2 coord, int2 dim, out float3 aabbMin, out float3 aabbMax) {
    aabbMin = float3( 1e9,  1e9,  1e9);
    aabbMax = float3(-1e9, -1e9, -1e9);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 nc = clamp(coord + int2(dx, dy), int2(0, 0), dim - int2(1, 1));
            float3 s = rgbToYCoCg(t_hdrCurrent[nc].rgb);
            aabbMin = min(aabbMin, s);
            aabbMax = max(aabbMax, s);
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    const int2 coord = int2(tid.xy);
    const int2 dim   = int2(screenWidth, screenHeight);
    if (coord.x >= dim.x || coord.y >= dim.y) return;

    float4 current = t_hdrCurrent[coord];

    if (firstFrame != 0u) {
        t_output[coord] = current;
        return;
    }

    // Reproject into the previous frame.
    // mv = prevPixel - curPixel  →  prevPixel = curPixel + mv
    float2 mv         = t_motionVectors[coord];
    float2 prevPixelF = float2(coord) + mv;          // fractional previous pixel position
    int2   prevBase   = int2(floor(prevPixelF));      // bottom-left corner of 2x2 bilinear quad
    float2 frac       = prevPixelF - float2(prevBase);
    prevBase = clamp(prevBase, int2(0, 0), dim - int2(2, 2));

    // Manual bilinear reconstruction from history (avoids sampler dependency).
    float4 h00 = t_history[prevBase + int2(0, 0)];
    float4 h10 = t_history[prevBase + int2(1, 0)];
    float4 h01 = t_history[prevBase + int2(0, 1)];
    float4 h11 = t_history[prevBase + int2(1, 1)];
    float4 hist = lerp(lerp(h00, h10, frac.x), lerp(h01, h11, frac.x), frac.y);

    // Clamp history to 3x3 neighbourhood AABB in YCoCg to suppress ghosting.
    float3 aabbMin, aabbMax;
    neighbourhoodAABB(coord, dim, aabbMin, aabbMax);
    float3 histClamped = YCoCgToRgb(clamp(rgbToYCoCg(hist.rgb), aabbMin, aabbMax));

    // Temporal blend: alpha = current weight (0.1), (1-alpha) = history weight (0.9).
    t_output[coord] = float4(lerp(histClamped, current.rgb, alpha), current.a);
}
