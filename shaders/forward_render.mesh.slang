// Mesh shader — one workgroup per visible meshlet (dispatched by task shader).
// 64 threads: each transforms one vertex and emits one triangle.

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
    uint     _pad;
    float4   sunDirection;
    float4   shadowParams;
    float4   presentationParams;
};

struct GpuVertex {
    float3 position;
    float  tangentX;
    float3 normal;
    float  tangentY;
    float2 uv;
    float  tangentZ;
    float  bitangentSign;
};

struct GpuInstance {
    uint  meshIndex;
    uint  materialIndex;
    uint  vertexOffset;
    uint  meshletOffset;
    uint  meshletCount;
    uint  geometryKind;
    float sphereRadius;
    uint  _pad;
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

/// Payload from task shader: instance index + base meshlet offset.
struct TaskPayload {
    uint instanceIdx;
    uint meshletBaseOffset;
};

struct MeshOutput {
    float4 pos           : SV_Position;
    float3 normal        : NORMAL;
    float3 tangent       : TANGENT;
    float  bitangentSign : TEXCOORD3;
    float3 worldPos      : TEXCOORD0;
    float2 uv            : TEXCOORD1;
    nointerpolation uint materialIdx : TEXCOORD2;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuVertex>   vertices;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(3, 0)]] StructuredBuffer<GpuMeshlet>  meshlets;
[[vk::binding(4, 0)]] StructuredBuffer<uint>        meshletVertices;
[[vk::binding(5, 0)]] StructuredBuffer<uint>        meshletTriangles;

static const uint kMaxVerts = 64;
static const uint kMaxTris  = 124;

[numthreads(64, 1, 1)]
[shader("mesh")]
[outputtopology("triangle")]
void main(
    uint3 gid : SV_GroupID,
    uint tig : SV_GroupIndex,
    in payload TaskPayload p,
    OutputVertices<MeshOutput, kMaxVerts> verts,
    OutputIndices<uint3, kMaxTris> tris
) {
    const uint instIdx = p.instanceIdx;
    const uint meshletIdx = p.meshletBaseOffset + gid.x;
    const GpuInstance inst = instances[instIdx];
    const GpuMeshlet meshlet = meshlets[meshletIdx];
    SetMeshOutputCounts(meshlet.vertexCount, meshlet.triangleCount);

    // 64 threads cover up to kMaxVerts (64) vertices and kMaxTris (124) triangles.
    // Loop with a stride of the workgroup size so every vertex/triangle is written —
    // otherwise SetMeshOutputCounts would declare triangles that are never emitted,
    // leaving undefined primitive indices (GPU out-of-bounds → device lost).
    for (uint vi = tig; vi < meshlet.vertexCount; vi += 64) {
        const uint globalVertIdx = meshletVertices[meshlet.vertexOffset + vi];
        const GpuVertex v = vertices[globalVertIdx];
        verts[vi].pos           = mul(float4(v.position, 1.0), viewProj);
        verts[vi].normal        = v.normal;
        verts[vi].tangent       = float3(v.tangentX, v.tangentY, v.tangentZ);
        verts[vi].bitangentSign = v.bitangentSign;
        verts[vi].worldPos      = v.position;
        verts[vi].uv            = v.uv;
        verts[vi].materialIdx   = inst.materialIndex;
    }
    for (uint ti = tig; ti < meshlet.triangleCount; ti += 64) {
        const uint byteBase = meshlet.triangleOffset * 4 + ti * 3;
        const uint i0 = (meshletTriangles[byteBase / 4] >> ((byteBase % 4) * 8)) & 0xFF;
        const uint i1 = (meshletTriangles[(byteBase + 1) / 4] >> (((byteBase + 1) % 4) * 8)) & 0xFF;
        const uint i2 = (meshletTriangles[(byteBase + 2) / 4] >> (((byteBase + 2) % 4) * 8)) & 0xFF;
        tris[ti] = uint3(i0, i1, i2);
    }
}
