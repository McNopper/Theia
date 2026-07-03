// hiz_build.comp.slang — build one level of the Hi-Z (hierarchical-Z) depth pyramid.
//
// Two-pass occlusion culling (B4) needs a max-depth mip chain of the current frame's
// depth buffer (produced by render pass 1). Each Hi-Z texel holds the MAXIMUM (farthest)
// depth over the base-image region it covers. Max reduction is the conservative choice for
// occlusion: a bounding sphere is culled only when its nearest depth is behind the farthest
// occluder in its whole screen footprint, so it can never wrongly cull visible geometry.
//
// The dispatch runs this shader once per mip level, halving the extent each time:
//   * copyMode != 0 (level 0): copy the D32 depth buffer 1:1 into R32F mip 0.
//   * copyMode == 0 (level L): dst = max over the 2x2 (or 3x3 on odd extents) of level L-1.
//
// Depth cannot be sampled as a storage image on all hardware, so level 0 reads the depth
// image as a sampled Texture2D and the pyramid itself lives in a separate R32F image.

[[vk::push_constant]]
cbuffer HiZPC {
    uint2 srcSize;   // extent of the source level (depth buffer for level 0)
    uint2 dstSize;   // extent of the destination level being written
    uint  copyMode;  // 1 = copy depth into mip 0, 0 = reduce previous mip
    uint  _pad;
};

// Level 0 source: the D32_SFLOAT depth buffer, read as a single-channel sampled texture.
[[vk::binding(0, 0)]] Texture2D<float> srcDepth;
// Level L>0 source: previous R32F mip (bound as storage for integer texel loads).
[[vk::binding(1, 0)]] [vk::image_format("r32f")] RWTexture2D<float> srcMip;
// Destination: the R32F mip level being written this dispatch.
[[vk::binding(2, 0)]] [vk::image_format("r32f")] RWTexture2D<float> dstMip;

float loadSrc(int2 c) {
    // Clamp to the valid source range so out-of-bounds neighbours (odd extents) safely
    // re-read an edge texel — harmless for a max reduction.
    int2 m = clamp(c, int2(0, 0), int2(srcSize) - int2(1, 1));
    return srcMip.Load(m);
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= dstSize.x || tid.y >= dstSize.y) {
        return;
    }

    float m;
    if (copyMode != 0u) {
        // Level 0: 1:1 copy of the depth buffer.
        m = srcDepth.Load(int3(int2(tid.xy), 0));
    } else {
        const int2 s = int2(tid.xy) * 2;
        m = max(max(loadSrc(s + int2(0, 0)), loadSrc(s + int2(1, 0))),
                max(loadSrc(s + int2(0, 1)), loadSrc(s + int2(1, 1))));

        // When the source extent is odd, the last destination texel must also fold in the
        // trailing column/row so the pyramid stays conservative (covers every base texel).
        const bool oddW = (srcSize.x & 1u) != 0u;
        const bool oddH = (srcSize.y & 1u) != 0u;
        if (oddW) {
            m = max(m, max(loadSrc(s + int2(2, 0)), loadSrc(s + int2(2, 1))));
        }
        if (oddH) {
            m = max(m, max(loadSrc(s + int2(0, 2)), loadSrc(s + int2(1, 2))));
        }
        if (oddW && oddH) {
            m = max(m, loadSrc(s + int2(2, 2)));
        }
    }

    dstMip[tid.xy] = m;
}
