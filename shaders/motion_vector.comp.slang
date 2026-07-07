// motion_vector.comp.slang — compute per-pixel motion vectors for temporal reprojection.
// Reads worldPos from giBuffer (set by forward pass), transforms by prev+cur VP matrices,
// and outputs pixel-space (dx, dy): where did this pixel come from in the previous frame?
//
// A1b: per-instance transform history.  The giBuffer w channel now stores
// asfloat(instanceIdx+1) (0 = sky).  For each pixel we look up that instance's
// previous-frame world transform and compute a corrected previous world position:
//
//   prevWorldPos = prevT * worldPos
//
// where prevT = prevInstanceTransforms[instanceIdx].  Since all current Theia
// scenes bake world transforms into vertex positions at load time (geometry xform
// = identity), prevT is always identity and prevWorldPos == worldPos for all
// existing scenes — i.e. A1a static-camera behavior is fully preserved.
// When non-identity per-instance transforms are introduced the caller should keep
// prevInstanceTransforms updated each frame.
//
// Runs AFTER GiPass (giBuffer is already in SHADER_READ_ONLY_OPTIMAL).
// Output image is R32G32_SFLOAT in VK_IMAGE_LAYOUT_GENERAL.

[[vk::push_constant]]
cbuffer MotionVectorPC {
    float4x4 curViewProj;     // row-major (slang-math convention) VP for current frame
    float4x4 prevViewProj;    // row-major (slang-math convention) VP for previous frame
    float4x4 invCurViewProj;  // row-major inverse of curViewProj (background reprojection)
};

// GI GBuffer: xyz = world-space primary hit position, w = asfloat(instanceIdx+1); 0 = sky.
[[vk::binding(0, 0)]] Texture2D<float4>   t_giBuffer;
// Output: pixel-space (dx, dy) motion vector.
[[vk::binding(1, 0)]] [vk::image_format("rg32f")] RWTexture2D<float2> t_motionVec;
// Per-instance previous-frame world transforms (one float4x4 per instance, row-major (slang-math convention)).
// Identity for static scenes (current Theia default: transforms baked into vertex positions).
[[vk::binding(2, 0)]] StructuredBuffer<float4x4> prevInstanceTransforms;

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint screenWidth, screenHeight;
    t_motionVec.GetDimensions(screenWidth, screenHeight);

    if (tid.x >= screenWidth || tid.y >= screenHeight) {
        return;
    }

    float4 gb = t_giBuffer.Load(int3(tid.xy, 0));
    float3 wpos = gb.xyz;
    // w = asfloat(instanceIdx+1): 0 means sky/background, no valid world position.
    uint instanceIdxPlus1 = asuint(gb.w);
    bool hasHit = (instanceIdxPlus1 > 0u);

    if (!hasHit) {
        // Background / sky: no world position, but it is NOT static under camera motion.
        // Treating it as a zero motion vector (the old behaviour) makes TAA fetch history
        // from the same pixel, so the background ghosts and jumps whenever the camera moves.
        // Reproject it as an infinitely-distant point (rotation-only parallax): reconstruct
        // this pixel's world-space view ray from the current inverse-VP, then project that
        // direction (w = 0 → point at infinity) through the previous-frame VP.
        float2 ndc = ((float2(tid.xy) + 0.5) / float2(screenWidth, screenHeight)) * 2.0 - 1.0;
        float4 farW  = mul(invCurViewProj, float4(ndc, 1.0, 1.0));
        float4 nearW = mul(invCurViewProj, float4(ndc, 0.0, 1.0));
        float3 dir = farW.xyz / farW.w - nearW.xyz / nearW.w;

        float4 prevClip = mul(prevViewProj, float4(dir, 0.0));
        if (prevClip.w <= 1e-6) {
            // Direction projects to/behind the previous camera plane — no reliable history.
            t_motionVec[tid.xy] = float2(0.0, 0.0);
            return;
        }
        float2 prevNdc = prevClip.xy / prevClip.w;
        float2 curPx   = float2(tid.xy) + 0.5;
        float2 prevPx  = (prevNdc * 0.5 + 0.5) * float2(screenWidth, screenHeight);
        t_motionVec[tid.xy] = prevPx - curPx;
        return;
    }

    // Per-instance motion correction: apply the previous-frame world transform to
    // get the world position this point occupied last frame.
    // For static scenes (prevT == identity): prevWorldPos == wpos (no-op).
    uint instanceIdx = instanceIdxPlus1 - 1u;
    float4x4 prevT = prevInstanceTransforms[instanceIdx];
    float3 prevWorldPos = mul(prevT, float4(wpos, 1.0)).xyz;

    // Clip-space positions (homogeneous).
    // Row-major matrices; mul(M, v) column-vector convention — matches the mesh shader.
    float4 curClip  = mul(curViewProj,  float4(wpos, 1.0));
    float4 prevClip = mul(prevViewProj, float4(prevWorldPos, 1.0));

    // Perspective divide to NDC [-1, 1].
    float2 curNdc  = curClip.xy  / max(curClip.w,  1e-6);
    float2 prevNdc = prevClip.xy / max(prevClip.w, 1e-6);

    // NDC → pixel coordinates.  Vulkan NDC: +x = right, +y = down.
    // NDC (-1,-1) = top-left pixel (0, 0).
    float2 curPx  = (curNdc  * 0.5 + 0.5) * float2(screenWidth, screenHeight);
    float2 prevPx = (prevNdc * 0.5 + 0.5) * float2(screenWidth, screenHeight);

    // Delta: where in the previous frame does this pixel come from?
    t_motionVec[tid.xy] = prevPx - curPx;
}
