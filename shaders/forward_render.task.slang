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
    uint     _pad;
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

/// Payload: instance index + base offset into the global meshlet buffer.
/// The mesh shader derives its per-meshlet index as: meshletBaseOffset + gid.x
struct TaskPayload {
    uint instanceIdx;
    uint meshletBaseOffset;
};

[[vk::binding(1, 0)]] StructuredBuffer<GpuInstance> instances;

[numthreads(1, 1, 1)]
[shader("amplification")]
void main(uint3 gid : SV_GroupID) {
    const uint instIdx = gid.x;
    const GpuInstance inst = instances[instIdx];

    TaskPayload payload;
    payload.instanceIdx       = instIdx;
    payload.meshletBaseOffset = inst.meshletOffset;

    // Emit 0 workgroups for non-triangle geometry or empty instances.
    const uint count = (inst.geometryKind == 0) ? inst.meshletCount : 0u;
    DispatchMesh(count, 1, 1, payload);
}
