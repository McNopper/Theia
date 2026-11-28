// forward_tangent.slang — the tangent of the texture coordinates at a rasterized fragment.
//
// The anisotropy of a surface follows its texture coordinates: OpenPBR binds
// `geometry_tangent` to the tangent of the geometry, which Harmonia's surface.slang takes
// as the direction of increasing u in the plane of the shading normal. The GI pass and
// Hyperion read that tangent from the triangle they hit (`reconstructTriangleGeometry`).
// A fragment has no triangle, only the interpolated world position and texture coordinate,
// and their screen-space derivatives stand in for the triangle's edge deltas: over a
// triangle both pairs are the image of the same linear map from UV to position, so they
// give the same direction of increasing u.
//
// Vertex tangents are not that direction (the OBJ and box meshes of the scenes carry none
// at all). They serve the normal map only, never the frame of the anisotropy.

import surface; // buildAnisotropyFrame (Harmonia/shaders/surface.slang)

/// The tangent of the texture coordinates from two position deltas and the UV deltas that
/// belong to them (Lengyel's method). This is the formula and the degeneracy guard of
/// Harmonia's `reconstructTriangleGeometry`, with the deltas of a fragment in place of the
/// edges of a triangle; the two must stay equal
/// (tests/shaders/forward_tangent_test.slang compares them). Texture coordinates without a
/// u direction are degenerate and give +X, which `buildAnisotropyFrame` turns into a
/// vector orthogonal to the normal.
float3 uvTangent(float3 dp1, float3 dp2, float2 duv1, float2 duv2) {
    const float det = duv1.x * duv2.y - duv2.x * duv1.y;
    return abs(det) > 1.0e-8 ? (dp1 * duv2.y - dp2 * duv1.y) / det : float3(1.0, 0.0, 0.0);
}

/// `uvTangent` of a fragment, from the screen-space derivatives of its world position and
/// its texture coordinate. Fragment stage only, and in uniform control flow: once a
/// neighbour of the quad is discarded, the derivatives are undefined.
float3 fragmentUvTangent(float3 worldPos, float2 uv) {
    return uvTangent(ddx(worldPos), ddy(worldPos), ddx(uv), ddy(uv));
}
