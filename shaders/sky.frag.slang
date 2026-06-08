// Sky/background fragment shader.
// Samples the GGX-prefiltered environment (mip 0 ≈ sharp panorama) along the
// world-space view ray, in the same lat-long convention as the IBL evaluation.
// Output is scaled by camera exposure so the background matches lit geometry.

[[vk::push_constant]]
cbuffer SkyPC {
    float4x4 invViewProj;
    float4   cameraPos;
    float    exposure;
    uint     hasEnv;
    uint     _pad0;
    uint     _pad1;
};

// Set 0 == forward IBL set (binding 1 = specular prefilter, binding 3 = sampler).
[[vk::binding(1, 0)]] Texture2D<float4> t_iblSpecular;
[[vk::binding(3, 0)]] SamplerState      s_iblLinear;

float2 dirToLatlong(float3 d) {
    // Hyperion env.slang convention (shaders/env.slang): u via atan2(z,x), v via acos(y).
    float longitude = atan2(d.z, d.x) * (1.0 / (2.0 * 3.14159265)) + 0.5;
    float latitude  = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265; // v=0 at top (y=+1)
    return float2(longitude, latitude);
}

[shader("fragment")]
float4 main(float4 pos : SV_Position, float3 viewRay : TEXCOORD0) : SV_Target0 {
    if (hasEnv == 0u) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    const float3 dir = normalize(viewRay);
    const float3 env = t_iblSpecular.SampleLevel(s_iblLinear, dirToLatlong(dir), 0.0).xyz;
    return float4(env * exposure, 1.0);
}
