// Sky/background fragment shader.
// Samples the RAW equirectangular environment panorama along the world-space view
// ray, in the same lat-long convention as Hyperion's env.slang sampleEnvMap().
// This matches Hyperion's primary-ray background exactly (evalEnvironment =
// envLuminanceScale * rawEnv(dir)); it deliberately does NOT sample the GGX
// prefiltered specular map, which is lower-resolution and redistributes energy.
// Output is scaled by env_unit_nits and camera exposure so the background matches
// lit geometry.

[[vk::push_constant]]
cbuffer SkyPC {
    float4x4 invViewProj;
    float4   cameraPos;
    float    exposure;
    uint     hasEnv;
    float    envScale;  ///< env_unit_nits — physical cd/m² per raw EXR unit
    uint     _pad1;
};

// Set 0 == forward IBL set (binding 3 = sampler, binding 4 = raw env panorama).
[[vk::binding(3, 0)]] SamplerState      s_iblLinear;
[[vk::binding(4, 0)]] Texture2D<float4> t_envRaw;

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
    // max(.,0) matches Hyperion sampleEnvMap clamping against negative filtered values.
    // Per-component finite guard: some HDR panoramas carry NaN/Inf texels, and max(NaN, 0)
    // is not guaranteed to return 0 (SPIR-V FMax is undefined for NaN). Without this the
    // sky pass writes NaN straight into the HDR image (the forward geometry path is already
    // protected by safe3), poisoning accumulation, the denoiser and the captured EXR.
    const float3 raw = t_envRaw.SampleLevel(s_iblLinear, dirToLatlong(dir), 0.0).xyz;
    const float3 env = float3(isfinite(raw.x) ? max(raw.x, 0.0) : 0.0,
                              isfinite(raw.y) ? max(raw.y, 0.0) : 0.0,
                              isfinite(raw.z) ? max(raw.z, 0.0) : 0.0);
    return float4(env * envScale * exposure, 1.0);
}
