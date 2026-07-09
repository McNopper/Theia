// Task (amplification) shader — one workgroup per scene instance.
// Emits one mesh workgroup per meshlet in the instance (no culling).

[[vk::push_constant]]
cbuffer ForwardPC {
    float4x4 viewProj;
    float4x4 view;
    float4   cameraPos;   // xyz = world-space camera position
    float    exposure;
    uint     lightCount;
    uint     emissiveTriangleCount;
    uint     tilesX;
    uint     tilesY;
    uint     screenWidth;
    uint     screenHeight;
    uint     transparentMaxDepth;
    uint     frameSampleIndex;
    uint     rngBaseSeed;
    uint     rngFlags;
    uint     _padRng;
    uint     envImportanceWidth;
    uint     envImportanceHeight;
    uint     _padEnv;
    float4   sunDirection;
    float4   shadowParams;
    float4   presentationParams;
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

/// Payload: instance index + base offset into the global meshlet buffer.
/// The mesh shader derives its per-meshlet index as: meshletBaseOffset + gid.x
struct TaskPayload {
    uint instanceIdx;
    uint meshletBaseOffset;
};

[[vk::binding(1, 0)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(6, 0)]] StructuredBuffer<GpuMaterial> materials;
/// Compact visible instance list written by GpuCullPass each frame.
/// gid.x indexes into this list to get the actual scene instance index.
[[vk::binding(10, 0)]] StructuredBuffer<uint> compactInstanceList;

[numthreads(1, 1, 1)]
[shader("amplification")]
void main(uint3 gid : SV_GroupID) {
    const uint instIdx = compactInstanceList[gid.x];
    const GpuInstance inst = instances[instIdx];
    const GpuMaterial mat = materials[inst.materialIndex];

    TaskPayload payload;
    payload.instanceIdx       = instIdx;
    payload.meshletBaseOffset = inst.meshletOffset;

    const bool isTransparent = (saturate(mat.opacityFlagsPad.x) < 0.9999) || (mat.transmissionColorWeight.w > 0.0);
    const bool passAccepts = (presentationParams.y < 0.5) ? !isTransparent : isTransparent;
    // Emit 0 workgroups for non-triangle geometry or empty instances.
    const uint count = (inst.geometryKind == 0 && passAccepts) ? inst.meshletCount : 0u;
    DispatchMesh(count, 1, 1, payload);
}
