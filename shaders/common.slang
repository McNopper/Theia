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
static const uint kNoTexture = 0xFFFFFFFFu;
