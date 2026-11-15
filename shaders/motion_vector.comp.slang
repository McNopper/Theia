// motion_vector.comp.slang — compute per-pixel motion vectors for temporal reprojection.
// Reads worldPos from giBuffer (set by forward pass), transforms by prev+cur VP matrices,
// and outputs pixel-space (dx, dy): where did this pixel come from in the previous frame?
//
// A1b/ANI3: per-instance transform history.  The giBuffer w channel stores
// asfloat(instanceIdx+1) (0 = sky).  For each pixel we look up that instance's
// previous-frame world transform and compute a corrected previous world position:
//
//   prevWorldPos = prevT * inverse(curT) * wpos
//
// where prevT = prevInstanceTransforms[instanceIdx], curT = instanceTransforms[idx]
// and wpos was shaded with the CURRENT transform (the mesh/GI shaders apply o2w), so
// the current transform is undone before the previous one is applied.  Scene::updateTlas
// (ANI3) keeps both buffers in step with the animation stamp: CURRENT = this frame's
// stamped pose, PREV = last rendered frame's.  Static scenes (or the build pose before
// the first sync) hold IDENTICAL matrices in both buffers; that case is detected
// exactly and keeps the pre-ANI3 static path bit-identical (no matrix inverse).
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
// ANI3: Scene::updateTlas refreshes this each stamped frame (PREV = last rendered pose).
[[vk::binding(2, 0)]] StructuredBuffer<float4x4> prevInstanceTransforms;
// Per-instance CURRENT-frame world transforms (the buffer the mesh/GI shaders shade
// with). Needed to undo the current transform before applying the previous one.
[[vk::binding(3, 0)]] StructuredBuffer<float4x4> instanceTransforms;

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

    // Per-instance motion correction (ANI3).  The world position read from giBuffer is
    // in the CURRENT frame's world space (the mesh/GI shaders apply the instance's
    // object→world transform o2w = instanceTransforms[idx]), so the surface point sat
    // at prevT·inverse(curT)·wpos last frame.  Static scenes (and the build pose before
    // the first sync) hold identical matrices in both buffers - detected exactly below,
    // preserving the pre-ANI3 static path bit-identically (no matrix inverse).
    const uint instanceIdx = instanceIdxPlus1 - 1u;
    const float4x4 curT = instanceTransforms[instanceIdx];
    const float4x4 prevT = prevInstanceTransforms[instanceIdx];
    const bool sameTransform = all(curT[0] == prevT[0]) && all(curT[1] == prevT[1]) &&
                               all(curT[2] == prevT[2]) && all(curT[3] == prevT[3]);
    float3 prevWorldPos = wpos;
    if (!sameTransform) {
        // Affine inverse of curT: instance transforms are TRS (last row = 0 0 0 1), so
        // inv = [A⁻¹, -A⁻¹·t; 0 1]. Slang has NO inverse() intrinsic (verified against
        // slangc 1.4.363), so the adjugate form of common.slang's inverse3x3 is inlined
        // here to keep this standalone pass shader's include graph unchanged. A singular
        // A (zero scale - already a broken render) falls back to no object motion.
        const float3 c0 = float3(curT[0].x, curT[1].x, curT[2].x); // columns of A
        const float3 c1 = float3(curT[0].y, curT[1].y, curT[2].y);
        const float3 c2 = float3(curT[0].z, curT[1].z, curT[2].z);
        const float3 r0 = cross(c1, c2); // rows of adj(A)
        const float3 r1 = cross(c2, c0);
        const float3 r2 = cross(c0, c1);
        const float det = dot(c0, r0);
        if (abs(det) > 1.0e-20) {
            const float invDet = 1.0 / det;
            const float3 t = float3(curT[0].w, curT[1].w, curT[2].w);
            const float3 it = -float3(dot(r0, t), dot(r1, t), dot(r2, t)) * invDet;
            const float4x4 invCur = float4x4(float4(r0 * invDet, it.x),
                                             float4(r1 * invDet, it.y),
                                             float4(r2 * invDet, it.z),
                                             float4(0.0, 0.0, 0.0, 1.0));
            prevWorldPos = mul(prevT, mul(invCur, float4(wpos, 1.0))).xyz;
        }
    }

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
