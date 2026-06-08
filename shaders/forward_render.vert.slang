// Camera data passed via push constants
cbuffer PushConstants : register(b0) {
    float4x4 viewProj;
    float3 lightDir;
    float _pad1;
};

// Vertex data from buffer
struct GpuVertex {
    float3 position;
    float tangentX;
    float3 normal;
    float tangentY;
    float2 uv;
    float tangentZ;
    float bitangentSign;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuVertex> vertices;

struct VS_OUT {
    float4 pos : SV_Position;
    float3 normal : NORMAL;
    float3 worldPos : TEXCOORD0;
    float2 uv : TEXCOORD1;
};

VS_OUT main(uint vertexId : SV_VertexID) {
    VS_OUT output;
    
    GpuVertex v = vertices[vertexId];
    
    // Transform to clip space
    output.pos = mul(float4(v.position, 1.0), viewProj);
    
    // Pass world-space data to fragment shader
    output.normal = normalize(v.normal);
    output.worldPos = v.position;
    output.uv = v.uv;
    
    return output;
}
