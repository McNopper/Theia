// common.slang — Theia rasterization pipeline
// Struct layout is kept bit-for-bit identical to Hyperion so the same bsdf.slang
// and math.slang can be imported without modification.

struct GpuVertex {
    float3 position;
    float  tangentX;
    float3 normal;
    float  tangentY;
    float2 uv;
    float  tangentZ;
    float  bitangentSign;
};

struct GpuMaterial {
    float4 baseColorWeight;
    float4 baseMetalnessDiffRough;
    float4 specularColorWeight;
    float4 specularRoughAnisoIor;
    float4 transmissionColorWeight;
    float4 transmissionParams;
    float4 transmissionScatter;
    float4 subsurfaceColorWeight;
    float4 subsurfaceRadiusScale;
    uint4  textureIndices;
    float4 thinFilmParams;
    float4 coatColorWeight;
    float4 coatRoughAnisoIorDark;
    float4 fuzzColorWeight;
    float4 fuzzRoughPad;
    float4 emissionColorLum;
    float4 opacityFlagsPad;
    uint4  textureIndices2;
};

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

struct GpuLight {
    float3 position;   float  typeF;
    float3 direction;  float  range;
    float3 color;      float  intensity;
    float  halfWidth;
    float  halfHeight;
    float  cosInner;
    float  cosOuter;
};

struct GpuEmissiveTriangle {
    float4 v0_area;
    float4 edge1_emitR;
    float4 edge2_emitG;
    float4 normal_emitB;
};

/// Light type constants — match LightType C++ enum.
static const uint kLightRect        = 0;
static const uint kLightPoint       = 1;
static const uint kLightSpot        = 2;
static const uint kLightDirectional = 3;
static const uint kLightSky         = 4;

static const float kPi        = 3.14159265359;
static const float kInvPi     = 0.31830988618;
static const float kHalfPi    = 1.57079632679;
static const float kRayEpsilon = 0.001;
static const float kInf        = 1.0e30;

/// Rec.2020 luminance coefficients (ITU-R BT.2020).
static const float3 kRec2020Luma = float3(0.2627, 0.6780, 0.0593);

static const uint kInstanceMaskEmissive    = 0x01u;
static const uint kInstanceMaskTransparent = 0x02u;
static const uint kInstanceMaskOpaque      = 0x04u;
static const uint kInstanceMaskNonEmissive = kInstanceMaskTransparent | kInstanceMaskOpaque;
static const uint kInstanceMaskAll         = 0xFFu;
static const uint kNoTexture = 0xFFFFFFFFu;
