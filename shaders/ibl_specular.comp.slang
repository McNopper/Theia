static const float kPi = 3.14159265359;

import env_sample;

[[vk::binding(0, 0)]] Texture2D<float4> envMap;
[[vk::binding(1, 0)]] SamplerState envSampler;
[[vk::binding(2, 0)]] RWTexture2D<float4> outPrefilter;

[[vk::push_constant]]
cbuffer PC {
    float roughness;
    float envScale; ///< env_unit_nits — physical cd/m² per raw EXR unit
};

float radicalInverse_VdC(uint bits)
{
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
    bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
    bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley2d(uint i, uint N)
{
    return float2(float(i) / float(N), radicalInverse_VdC(i));
}

float3 importanceSampleGGX(float2 xi, float roughnessValue)
{
    float alpha = max(roughnessValue * roughnessValue, 1.0e-4);
    float alpha2 = alpha * alpha;
    float phi = 2.0 * kPi * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / max(1.0 + (alpha2 - 1.0) * xi.y, 1.0e-4));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float3 tangentSpaceToWorld(float3 v, float3 N)
{
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    return normalize(T * v.x + B * v.y + N * v.z);
}

float3 latlongToDir(float2 uv)
{
    float phi = (uv.x - 0.5) * (2.0 * kPi);
    float theta = uv.y * kPi;            // 0 at top (y=+1), pi at bottom
    float sinTheta = sin(theta);
    return normalize(float3(sinTheta * cos(phi), cos(theta), sinTheta * sin(phi)));
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint width;
    uint height;
    outPrefilter.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    const uint sampleCount = 1024;
    float2 uv = (float2(tid.xy) + 0.5) / float2(float(width), float(height));
    float3 N = latlongToDir(uv);
    float3 V = N;
    float3 prefiltered = 0.0.xxx;
    float totalWeight = 0.0;

    for (uint i = 0; i < sampleCount; ++i) {
        float3 H = tangentSpaceToWorld(importanceSampleGGX(hammersley2d(i, sampleCount), roughness), N);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NoL = saturate(dot(N, L));
        if (NoL > 0.0) {
            prefiltered += envMap.SampleLevel(envSampler, dirToLatlong(L), 0.0).xyz * NoL;
            totalWeight += NoL;
        }
    }

    outPrefilter[tid.xy] = float4(prefiltered / max(totalWeight, 1.0e-4) * envScale, 1.0);
}
