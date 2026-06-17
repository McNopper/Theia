// Conservative min-depth pyramid for hierarchical screen-space tracing.
//
// Each dispatch writes one mip level. Mip 0 copies the depth buffer. Higher mips
// downsample the previous mip with a 2x2 min filter so coarser levels remain
// conservative for SSR ray marching.

[[vk::push_constant]]
cbuffer PyramidPC {
    uint srcMip;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(0, 0)]] Texture2D<float>  t_srcDepth;
[[vk::binding(1, 0)]] RWTexture2D<float> rw_dstDepth;

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 dstDims;
    rw_dstDepth.GetDimensions(dstDims.x, dstDims.y);
    if (any(id.xy >= dstDims)) {
        return;
    }

    uint2 srcDims;
    t_srcDepth.GetDimensions(srcDims.x, srcDims.y);
    const uint2 base = id.xy * 2u;
    const uint2 last = uint2(srcDims.x - 1u, srcDims.y - 1u);

    const uint2 p0 = min(base + uint2(0u, 0u), last);
    const uint2 p1 = min(base + uint2(1u, 0u), last);
    const uint2 p2 = min(base + uint2(0u, 1u), last);
    const uint2 p3 = min(base + uint2(1u, 1u), last);

    const float d0 = t_srcDepth.Load(int3(p0, 0));
    const float d1 = t_srcDepth.Load(int3(p1, 0));
    const float d2 = t_srcDepth.Load(int3(p2, 0));
    const float d3 = t_srcDepth.Load(int3(p3, 0));
    rw_dstDepth[id.xy] = min(min(d0, d1), min(d2, d3));
}
