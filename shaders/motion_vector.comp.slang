// motion_vector.comp.slang — compute per-pixel motion vectors for temporal reprojection.
// Reads worldPos from giBuffer (set by forward pass), transforms by prev+cur VP matrices,
// and outputs pixel-space (dx, dy): where did this pixel come from in the previous frame?
//
// Runs AFTER GiPass (giBuffer is already in SHADER_READ_ONLY_OPTIMAL).
// Output image is R32G32_SFLOAT in VK_IMAGE_LAYOUT_GENERAL.

[[vk::push_constant]]
cbuffer MotionVectorPC {
    float4x4 curViewProj;   // glm::transpose(proj * view) for current frame
    float4x4 prevViewProj;  // glm::transpose(proj * view) for previous frame
};

// GI GBuffer: xyz = world-space primary hit position, w = asfloat(matIdx+1); 0 = sky.
[[vk::binding(0, 0)]] Texture2D<float4>   t_giBuffer;
// Output: pixel-space (dx, dy) motion vector.
[[vk::binding(1, 0)]] [vk::image_format("rg32f")] RWTexture2D<float2> t_motionVec;

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint screenWidth, screenHeight;
    t_motionVec.GetDimensions(screenWidth, screenHeight);

    if (tid.x >= screenWidth || tid.y >= screenHeight) {
        return;
    }

    float4 gb = t_giBuffer.Load(int3(tid.xy, 0));
    float3 wpos = gb.xyz;
    // w = asfloat(matIdx+1): 0 means sky/background, no valid world position.
    bool hasHit = (gb.w > 0.0);

    if (!hasHit) {
        t_motionVec[tid.xy] = float2(0.0, 0.0);
        return;
    }

    // Clip-space positions (homogeneous) using VP × world convention.
    // curViewProj / prevViewProj are glm::transpose(VP), so mul(M, v) = VP × v.
    float4 curClip  = mul(curViewProj,  float4(wpos, 1.0));
    float4 prevClip = mul(prevViewProj, float4(wpos, 1.0));

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
