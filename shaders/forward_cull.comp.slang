// Per-instance GPU-driven frustum cull compute shader.
// One thread per scene instance. Each thread frustum-culls its instance's world-space
// bounding sphere against the view frustum (4 side planes + near plane).
// Visible instances are compacted into compactInstanceList and the visible count
// accumulated atomically into indirectDrawBuf.groupCountX.
//
// GPU-driven design: the CPU submits no per-instance visibility decisions.
// indirectDrawBuf holds a single VkDrawMeshTasksIndirectCommandEXT = {visibleCount, 1, 1}
// consumed directly by vkCmdDrawMeshTasksIndirectEXT (one draw call).
// The task shader indexes compactInstanceList[gid.x] to get the instance index.
//
// Initialization per frame: vkCmdFillBuffer zeros all 12 bytes; thread 0 of this shader
// restores groupCountY=1 and groupCountZ=1 before any atomic add on groupCountX.
//
// Gribb-Hartmann plane extraction from row-major view-projection matrix M:
//   Left:   M[3] + M[0]   Right:  M[3] - M[0]
//   Bottom: M[3] + M[1]   Top:    M[3] - M[1]
//   Near:   M[2]                              (Vulkan depth in [0,1]; near at z/w=0)
//
// Ref: Gribb & Hartmann — "Fast Extraction of Viewing Frustum Planes from the
//      World-View-Projection Matrix" (2001)

[[vk::push_constant]]
cbuffer CullPC {
    float4x4 viewProj;      // row-major VP matrix (same as ForwardRenderer's pcBase.viewProj)
    uint     instanceCount;
    uint     _pad0, _pad1, _pad2; // pad to 80 bytes total
};

struct GpuInstance {
    uint  meshIndex;
    uint  materialIndex;
    uint  vertexOffset;
    uint  indexOffset;
    uint  indexCount;
    uint  meshletOffset;
    uint  meshletCount;
    uint  geometryKind; ///< 0 = triangle mesh, 1 = analytic sphere (never drawn by mesh shader)
    float sphereRadius; ///< legacy; bounds from instanceBounds
};

struct GpuInstanceBounds {
    float centerX;
    float centerY;
    float centerZ;
    float radius;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuInstance>       instances;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstanceBounds> instanceBounds;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            compactInstanceList;
/// Single VkDrawMeshTasksIndirectCommandEXT = {groupCountX, groupCountY, groupCountZ}.
/// groupCountX is accumulated atomically (= visible instance count).
/// groupCountY and groupCountZ are set to 1 by thread 0 each frame.
[[vk::binding(3, 0)]] RWByteAddressBuffer                 indirectDrawBuf;

/// Normalize a frustum plane (xyz = normal, w = D), returning unit-normal plane.
float4 normalizePlane(float4 p) {
    return p / max(length(p.xyz), 1e-6f);
}

/// Returns true if the bounding sphere (center, radius) overlaps the view frustum.
/// Uses 5 planes: left, right, bottom, top, near. Far plane omitted (conservative).
bool frustumCull(float3 center, float radius) {
    // Gribb-Hartmann extraction from row-major VP matrix.
    // viewProj[r] = r-th row as float4 (Slang row-major indexing matches slang-math).
    const float4 pLeft   = normalizePlane(viewProj[3] + viewProj[0]);
    const float4 pRight  = normalizePlane(viewProj[3] - viewProj[0]);
    const float4 pBottom = normalizePlane(viewProj[3] + viewProj[1]);
    const float4 pTop    = normalizePlane(viewProj[3] - viewProj[1]);
    const float4 pNear   = normalizePlane(viewProj[2]);              // near: z/w >= 0

    // Sphere outside plane if signed distance of centre from plane < -radius.
    if (dot(pLeft.xyz,   center) + pLeft.w   < -radius) return false;
    if (dot(pRight.xyz,  center) + pRight.w  < -radius) return false;
    if (dot(pBottom.xyz, center) + pBottom.w < -radius) return false;
    if (dot(pTop.xyz,    center) + pTop.w    < -radius) return false;
    if (dot(pNear.xyz,   center) + pNear.w   < -radius) return false;
    return true;
}

[numthreads(64, 1, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID) {
    // Thread 0 restores the constant indirect command fields that vkCmdFillBuffer zeroed.
    // groupCountX is left at 0 and will be accumulated via InterlockedAdd below.
    if (dtid.x == 0) {
        indirectDrawBuf.Store(4, 1u);  // groupCountY = 1
        indirectDrawBuf.Store(8, 1u);  // groupCountZ = 1
    }

    const uint instIdx = dtid.x;
    if (instIdx >= instanceCount) return;

    const GpuInstance inst = instances[instIdx];

    // Only draw triangle meshes with at least one meshlet.
    if (inst.geometryKind != 0u || inst.meshletCount == 0u) return;

    const GpuInstanceBounds bounds = instanceBounds[instIdx];

    // Frustum test: skip instances whose bounding sphere is outside the view frustum.
    if (bounds.radius >= 0.0f && !frustumCull(float3(bounds.centerX, bounds.centerY, bounds.centerZ), bounds.radius))
        return;

    // Visible: atomically claim a slot in the compact list and increment the indirect
    // task workgroup count (groupCountX at byte offset 0).
    uint slot = 0;
    indirectDrawBuf.InterlockedAdd(0, 1u, slot);
    compactInstanceList[slot] = instIdx;
}
