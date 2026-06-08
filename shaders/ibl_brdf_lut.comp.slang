static const float kPi = 3.14159265359;
static const float kInvPi = 0.31830988618;

[[vk::binding(0, 0)]] RWTexture2D<float2> outLut;

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

float3 importanceSampleGGX(float2 xi, float roughness)
{
    float alpha = max(roughness * roughness, 1.0e-4);
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

float2 dirToLatlong(float3 d)
{
    return float2(atan2(d.z, d.x) * (1.0 / (2.0 * kPi)) + 0.5,
                  asin(clamp(d.y, -1.0, 1.0)) / kPi + 0.5);
}

float3 latlongToDir(float2 uv)
{
    float phi = (uv.x - 0.5) * (2.0 * kPi);
    float theta = (uv.y - 0.5) * kPi;
    float cosTheta = cos(theta);
    return normalize(float3(cos(phi) * cosTheta, sin(theta), sin(phi) * cosTheta));
}

float geometrySchlickGGX(float NoV, float roughness)
{
    float a = roughness;
    float k = (a * a) * 0.5;
    return NoV / max(NoV * (1.0 - k) + k, 1.0e-4);
}

float geometrySmith(float NoV, float NoL, float roughness)
{
    return geometrySchlickGGX(NoV, roughness) * geometrySchlickGGX(NoL, roughness);
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= 512 || tid.y >= 512) {
        return;
    }

    const uint sampleCount = 1024;
    float NoV = (float(tid.x) + 0.5) / 512.0;
    float roughness = (float(tid.y) + 0.5) / 512.0;

    float3 V = float3(sqrt(max(1.0 - NoV * NoV, 0.0)), 0.0, NoV);
    float A = 0.0;
    float B = 0.0;

    for (uint i = 0; i < sampleCount; ++i) {
        float2 xi = hammersley2d(i, sampleCount);
        float3 H = importanceSampleGGX(xi, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NoL = saturate(L.z);
        float NoH = saturate(H.z);
        float VoH = saturate(dot(V, H));
        if (NoL > 0.0) {
            float G = geometrySmith(NoV, NoL, roughness);
            float GVis = (G * VoH) / max(NoH * NoV, 1.0e-4);
            float Fc = pow(1.0 - VoH, 5.0);
            A += (1.0 - Fc) * GVis;
            B += Fc * GVis;
        }
    }

    outLut[tid.xy] = float2(A, B) / float(sampleCount);
}
