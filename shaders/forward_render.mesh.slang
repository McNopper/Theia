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
    uint     transparentMaxDepth;
    uint     frameSampleIndex;
    uint     rngBaseSeed;
    uint     rngFlags;
    uint     cullPhase;            // 0 = draw all (transparent / fallback), 1 = pass 1 (prev-visible), 2 = pass 2 (Hi-Z)
    uint     envImportanceWidth;
    uint     envImportanceHeight;
    uint     hiZMipCount;          // Hi-Z mip levels; 0 disables the occlusion test (first frame / camera cut)
    float4   sunDirection;
    float4   shadowParams;         // x = ray tMin, y = sky ambient floor, z = env_unit_nits, w = |proj[0][0]|
    float4   presentationParams;   // x = indirect ambient, y = pass flag, z = debug ray-hit, w = |proj[1][1]|
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
    nointerpolation uint instanceIdx : TEXCOORD4;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuVertex>   vertices;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(3, 0)]] StructuredBuffer<GpuMeshlet>  meshlets;
[[vk::binding(4, 0)]] StructuredBuffer<uint>        meshletVertices;
[[vk::binding(5, 0)]] StructuredBuffer<uint>        meshletTriangles;
// Two-pass Hi-Z occlusion culling (B4):
//   binding 7 = per-meshlet visibility from the PREVIOUS frame (read-only filter)
//   binding 8 = per-meshlet visibility for THIS frame (written when a meshlet is drawn)
//   binding 9 = current-frame Hi-Z max-depth pyramid (built from pass 1's depth)
[[vk::binding(7, 0)]] StructuredBuffer<uint>        prevVisibility;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint>      currVisibility;
[[vk::binding(9, 0)]] Texture2D<float>              t_hiZ;

static const uint kMaxVerts = 64;
static const uint kMaxTris  = 124;

/// Conservative Hi-Z occlusion test for a meshlet bounding sphere against the current-frame
/// max-depth pyramid. Returns true only when the sphere is fully behind the farthest
/// occluder over its entire screen footprint (never culls visible geometry).
bool hiZOccluded(GpuMeshlet ml) {
    if (hiZMipCount == 0u) {
        return false; // Hi-Z disabled: draw conservatively.
    }
    const float3 center = float3(ml.centerX, ml.centerY, ml.centerZ);
    const float  radius = ml.radius;

    // viewProj is row-major VP, so mul(vector, viewProj) = VP * vector = clip space.
    const float4 clip = mul(float4(center, 1.0), viewProj);
    if (clip.w <= radius) {
        return false; // sphere straddles / is behind the near plane → keep it.
    }

    // Nearest depth of the sphere (project the point closest to the camera).
    const float3 toCam = normalize(cameraPos.xyz - center);
    const float4 nearClip = mul(float4(center + toCam * radius, 1.0), viewProj);
    const float  nearDepth = nearClip.z / max(nearClip.w, 1e-6);

    // Screen-space AABB of the sphere. Using (w - radius) as the denominator enlarges the
    // footprint slightly, which keeps texel coverage conservative.
    const float2 centerUV = (clip.xy / clip.w) * 0.5 + 0.5;
    const float  denom = max(clip.w - radius, 1e-4);
    const float2 uvHalf = float2(shadowParams.w, presentationParams.w) * (radius / denom) * 0.5;
    const float2 minUV = clamp(centerUV - uvHalf, 0.0, 1.0);
    const float2 maxUV = clamp(centerUV + uvHalf, 0.0, 1.0);

    // Pick the mip whose texel covers the footprint, then max the (up to) 2x2 covering texels.
    const float2 sizePx = (maxUV - minUV) * float2(screenWidth, screenHeight);
    const float  maxSize = max(max(sizePx.x, sizePx.y), 1.0);
    int mip = (int)ceil(log2(maxSize));
    mip = clamp(mip, 0, (int)hiZMipCount - 1);

    uint mw, mh, mlv;
    t_hiZ.GetDimensions((uint)mip, mw, mh, mlv);
    const float2 dim = float2(mw, mh);
    int2 t0 = clamp(int2(minUV * dim), int2(0, 0), int2(mw - 1u, mh - 1u));
    int2 t1 = clamp(int2(maxUV * dim), int2(0, 0), int2(mw - 1u, mh - 1u));

    const float hz = max(max(t_hiZ.Load(int3(t0.x, t0.y, mip)), t_hiZ.Load(int3(t1.x, t0.y, mip))),
                         max(t_hiZ.Load(int3(t0.x, t1.y, mip)), t_hiZ.Load(int3(t1.x, t1.y, mip))));

    // Occluded when the nearest point of the sphere is behind the farthest occluder.
    return nearDepth > hz;
}

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

    // Two-pass Hi-Z occlusion culling. This decision is meshlet-uniform (identical across
    // all 64 threads), so any thread reaching the loops has already been kept.
    bool drawMeshlet;
    if (cullPhase == 0u) {
        drawMeshlet = true; // transparent pass / fallback: draw everything.
    } else if (cullPhase == 1u) {
        // Pass 1: redraw only meshlets that were visible last frame.
        drawMeshlet = (prevVisibility[meshletIdx] != 0u);
    } else {
        // Pass 2: consider only meshlets not already drawn in pass 1, and test them against
        // the current-frame Hi-Z built from pass 1's depth.
        drawMeshlet = (prevVisibility[meshletIdx] == 0u) && !hiZOccluded(meshlet);
    }

    if (!drawMeshlet) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    // Mark this meshlet visible this frame (fed back as prevVisibility next frame). One
    // thread writes to avoid redundant stores; the value is idempotent regardless.
    if (cullPhase != 0u && tig == 0u) {
        currVisibility[meshletIdx] = 1u;
    }

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
        verts[vi].instanceIdx   = instIdx;
    }
    for (uint ti = tig; ti < meshlet.triangleCount; ti += 64) {
        const uint byteBase = meshlet.triangleOffset * 4 + ti * 3;
        const uint i0 = (meshletTriangles[byteBase / 4] >> ((byteBase % 4) * 8)) & 0xFF;
        const uint i1 = (meshletTriangles[(byteBase + 1) / 4] >> (((byteBase + 1) % 4) * 8)) & 0xFF;
        const uint i2 = (meshletTriangles[(byteBase + 2) / 4] >> (((byteBase + 2) % 4) * 8)) & 0xFF;
        tris[ti] = uint3(i0, i1, i2);
    }
}
