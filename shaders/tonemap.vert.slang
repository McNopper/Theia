// Fullscreen triangle vertex shader for the tone-mapping graphics pass.
// No vertex buffer needed — three vertices are generated procedurally from SV_VertexID.
// Produces a triangle that covers the entire viewport in Vulkan NDC.
// Copied from Hyperion.

[shader("vertex")]
float4 main(uint vid : SV_VertexID) : SV_Position {
    float x = float(((vid << 1U) & 2U)) * 2.0f - 1.0f;
    float y = float((vid & 2U)) * 2.0f - 1.0f;
    return float4(x, y, 0.0f, 1.0f);
}
