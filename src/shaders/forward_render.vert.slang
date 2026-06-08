// Forward rendering vertex shader
// Outputs position, normal, and UV for fragment shader to evaluate OpenPBR BSDF

[[vk::binding(0, 0)]] StructuredBuffer<float4x4> worldMatrices;
[[vk::binding(1, 0)]] StructuredBuffer<uint> materialIndices;

struct VertexInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
    uint vertexIndex : SV_VertexID;
    uint instanceIndex : SV_InstanceID;
};

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPos;
    [[vk::location(1)]] float3 worldNormal;
    [[vk::location(2)]] float2 texCoord;
    [[vk::location(3)]] flat uint materialId;
};

cbuffer Camera : register(b0) {
    float4x4 viewProj;
    float3 cameraPos;
    float _pad0;
};

VertexOutput main(VertexInput input) {
    VertexOutput output;

    // Get world transform for this instance
    float4x4 worldMatrix = worldMatrices[input.instanceIndex];
    float3 worldPosition = mul(float4(input.position, 1.0), worldMatrix).xyz;

    output.position = mul(float4(worldPosition, 1.0), viewProj);
    output.worldPos = worldPosition;
    output.worldNormal = mul(input.normal, (float3x3)worldMatrix);
    output.texCoord = input.texCoord;
    output.materialId = materialIndices[input.instanceIndex];

    return output;
}
