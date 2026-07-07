// Sky/background fullscreen-triangle vertex shader.
// Generates three vertices covering the viewport at the far plane (z = 1) and
// reconstructs a world-space view ray per corner via the inverse view-projection.
// The fragment shader samples the environment panorama along this ray.

[[vk::push_constant]]
cbuffer SkyPC {
    float4x4 invViewProj; // row-major (VP)^{-1}; mul(M, v) column-vector convention
    float4   cameraPos;    // xyz = world-space camera position
    float    exposure;     // 1 / (1.2 * 2^EV100)
    uint     hasEnv;       // 1 = sample env map, 0 = black
    uint     _pad0;
    uint     _pad1;
};

struct VSOut {
    float4 pos     : SV_Position;
    float3 viewRay : TEXCOORD0;
};

[shader("vertex")]
VSOut main(uint vid : SV_VertexID) {
    const float x = float(((vid << 1U) & 2U)) * 2.0f - 1.0f;
    const float y = float((vid & 2U)) * 2.0f - 1.0f;

    VSOut o;
    o.pos = float4(x, y, 1.0f, 1.0f); // far plane

    // Unproject the far-plane NDC point to world space.
    float4 world = mul(invViewProj, float4(x, y, 1.0f, 1.0f));
    world /= world.w;
    o.viewRay = world.xyz - cameraPos.xyz;
    return o;
}
