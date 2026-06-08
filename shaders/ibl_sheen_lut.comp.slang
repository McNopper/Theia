static const float kPi = 3.14159265359;

[[vk::binding(0, 0)]] RWTexture2D<float> outLut;

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

float3 sampleCosineHemisphere(float2 xi)
{
    float r = sqrt(xi.x);
    float phi = 2.0 * kPi * xi.y;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(1.0 - xi.x, 0.0));
    return float3(x, y, z);
}

float charlieNdf(float roughness, float cosTheta)
{
    float alpha = max(roughness * roughness, 1.0e-3);
    float invAlpha = 1.0 / alpha;
    float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    return ((2.0 + invAlpha) * pow(max(sinTheta, 1.0e-4), invAlpha)) / (2.0 * kPi);
}

float lambdaCharlie(float roughness, float cosTheta)
{
    float alpha = max(roughness * roughness, 1.0e-3);
    float sinTheta2 = saturate(1.0 - cosTheta * cosTheta);
    if (sinTheta2 <= 1.0e-6) {
        return 0.0;
    }

    float cot2 = (cosTheta * cosTheta) / max(sinTheta2, 1.0e-6);
    float invAlpha = 1.0 / alpha;
    return 0.5 * (-1.0 + sqrt(1.0 + (invAlpha * invAlpha) / max(cot2, 1.0e-6)));
}

float G_Charlie(float roughness, float cosTheta)
{
    float lambda = lambdaCharlie(roughness, max(cosTheta, 1.0e-4));
    return 1.0 / (1.0 + 2.0 * lambda);
}

float evalSheenAlbedo(float roughness, float cosTh)
{
    float ct = max(cosTh, 1.0e-4);
    return charlieNdf(roughness, ct) * G_Charlie(roughness, ct) / max(4.0 * ct, 1.0e-4);
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= 512 || tid.y >= 512) {
        return;
    }

    const uint sampleCount = 1024;
    float roughness = (float(tid.y) + 0.5) / 512.0;
    float sum = 0.0;

    for (uint i = 0; i < sampleCount; ++i) {
        float2 xi = hammersley2d(i, sampleCount);
        float3 wi = sampleCosineHemisphere(xi);
        sum += evalSheenAlbedo(roughness, wi.z);
    }

    outLut[tid.xy] = kPi * sum / float(sampleCount);
}
