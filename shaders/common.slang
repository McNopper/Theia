// common.slang — Theia rasterization pipeline
// Struct layout is kept bit-for-bit identical to Hyperion so the same bsdf.slang
// and math.slang can be imported without modification.

__exported import gpu_types;

struct GpuInstance {
    uint  meshIndex;
    uint  materialIndex;
    uint  vertexOffset;
    uint  indexOffset;
    uint  indexCount;
    uint  meshletOffset;
    uint  meshletCount;
    uint  geometryKind;
    float sphereRadius;
};

struct GpuMeshlet {
    uint  vertexOffset;
    uint  triangleOffset;
    uint  vertexCount;
    uint  triangleCount;
    float centerX;
    float centerY;
    float centerZ;
    float radius;
};

static const uint kInstanceMaskEmissive    = 0x01u;
static const uint kInstanceMaskTransparent = 0x02u;
static const uint kInstanceMaskOpaque      = 0x04u;
static const uint kInstanceMaskNonEmissive = kInstanceMaskTransparent | kInstanceMaskOpaque;
// kNoTexture lives in gpu_types.slang (re-exported via `__exported import gpu_types`).

/// Inverse of a row-major 3×3 (cross-products-of-columns form). Theia-internal: the raster
/// path derives normal matrices from instance transforms via this (Hyperion uses the RT
/// WorldToObject3x4 builtin instead). Shared by gi.comp and forward_render.mesh.
float3x3 inverse3x3(float3x3 m) {
    const float3x3 col = transpose(m);
    const float3 c0 = col[0], c1 = col[1], c2 = col[2];
    const float3 r0 = cross(c1, c2);
    const float3 r1 = cross(c2, c0);
    const float3 r2 = cross(c0, c1);
    const float invDet = 1.0 / dot(c0, r0);
    return float3x3(r0 * invDet, r1 * invDet, r2 * invDet);
}
