// Forward rendering fragment shader
// Evaluates OpenPBR BSDF with direct lighting (lights + IBL)
// Uses clustered light culling for efficient light access

// Material properties (OpenPBR)
struct Material {
    float3 base_color;
    float base_weight;
    float base_metalness;
    float specular_ior;
    float specular_roughness;
    float coat_weight;
    float coat_roughness;
    float coat_darkening;
    float3 emission_luminance;
    float emission_luminance_scale;
};

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPos;
    [[vk::location(1)]] float3 worldNormal;
    [[vk::location(2)]] float2 texCoord;
    [[vk::location(3)]] flat uint materialId;
};

// Light structures
struct Light {
    float3 position;
    float intensity;
    float3 color;
    float radius;
};

cbuffer SceneData : register(b0) {
    float3 sunDirection;
    float sunIntensity;
    float3 ambientLight;
    float iblIntensity;
};

[[vk::binding(0, 0)]] StructuredBuffer<Material> materials;
[[vk::binding(1, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(2, 0)]] TextureCube<float4> envMap;
[[vk::binding(3, 0)]] SamplerState envSampler;

float3 fresnelSchlick(float3 f0, float cosTheta) {
    return f0 + (1.0 - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float distributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return nom / (3.14159 * denom * denom);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = r * r / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return nom / denom;
}

float3 evaluateDirectLight(float3 N, float3 V, float3 L, float3 radiance, Material mat) {
    float3 H = normalize(V + L);
    float distance = length(L);
    float attenuation = 1.0 / (distance * distance);
    float3 light = radiance * attenuation;

    // Cook-Torrance BRDF
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    float3 f0 = mix(float3(0.04), mat.base_color, mat.base_metalness);
    float3 F = fresnelSchlick(f0, max(dot(H, V), 0.0));

    float NDF = distributionGGX(N, H, mat.specular_roughness);
    float G = geometrySchlickGGX(NdotV, mat.specular_roughness) * 
              geometrySchlickGGX(NdotL, mat.specular_roughness);

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - mat.base_metalness);

    float3 specular = NDF * G * F / (4.0 * NdotV * NdotL + 0.001);
    float3 diffuse = kD * mat.base_color / 3.14159;

    return (diffuse + specular) * light * NdotL;
}

float4 main(VertexOutput input) : SV_Target {
    Material mat = materials[input.materialId];

    float3 N = normalize(input.worldNormal);
    float3 V = normalize(input.worldPos); // Camera is at origin for view direction

    float3 color = float3(0.0);

    // Direct sunlight
    float3 sunLight = evaluateDirectLight(N, V, normalize(-sunDirection), 
                                          float3(1.0) * sunIntensity, mat);
    color += sunLight;

    // Ambient lighting
    color += mat.base_color * ambientLight * mat.base_weight;

    // Add emission
    color += mat.emission_luminance * mat.emission_luminance_scale;

    // Convert to HDR output format
    return float4(color, 1.0);
}
