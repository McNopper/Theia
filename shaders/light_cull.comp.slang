// Forward+ light culling compute shader.
// Assigns each point/spot light to all 16×16 pixel screen tiles it overlaps.
// One thread per light: each thread iterates over all tiles and checks
// sphere-frustum overlap.
//
// Ref: Olsson & Assarsson — "Tiled Shading" (JCGT 2011)
//      https://jcgt.org/published/0001/01/01/

[[vk::push_constant]]
cbuffer LightCullPC {
    float4x4 proj;          // projection matrix
    float4x4 view;          // world-to-camera matrix
    uint2    tilesXY;       // (tilesX, tilesY)
    uint2    screenSize;    // (width, height) in pixels
    uint     lightCount;
    float    nearZ;         // near clip plane distance (positive)
    float    farZ;          // far clip plane distance (positive)
    uint     _pad;
};

struct GpuLight {
    float3  position;
    float   typeF;
    float3  direction;
    float   range;
    float3  color;
    float   intensity;
    float   halfWidth;
    float   halfHeight;
    float   cosInner;
    float   cosOuter;
};

static const uint kLightRect        = 0;
static const uint kLightPoint       = 1;
static const uint kLightSpot        = 2;
static const uint kLightDirectional = 3;
static const uint kLightSky         = 4;

static const uint kMaxLightsPerTile = 128;

[[vk::binding(0, 0)]] StructuredBuffer<GpuLight>  lights;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>    tileLightCounts;  // [tilesX*tilesY]
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>    tileLightIndices; // [tilesX*tilesY * kMaxLightsPerTile]

/// Compute view-space AABB of a tile defined by its screen-pixel bounds.
/// Returns min/max in view space (z range covers [nearZ, farZ]).
/// We use the classic frustum plane approach for sphere-frustum test.
float4 tileFrustumPlane(uint2 tile, bool right, bool top) {
    // NDC corners of this tile
    const float tileW = 2.0 / float(tilesXY.x);
    const float tileH = 2.0 / float(tilesXY.y);
    // ndc x: -1 + tile.x * tileW (left edge), or + tileW (right edge)
    const float ndcX = -1.0 + float(tile.x + (right ? 1u : 0u)) * tileW;
    const float ndcY = -1.0 + float(tile.y + (top  ? 1u : 0u)) * tileH;
    // Un-project to view space using inverse projection
    const float3 p = float3(ndcX / proj[0][0], ndcY / proj[1][1], -1.0);
    // Build plane: normal = cross of two edge vectors through origin
    return float4(p, 0.0); // unused — see inline plane construction below
}

[numthreads(64, 1, 1)]
[shader("compute")]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint lightIdx = dtid.x;
    if (lightIdx >= lightCount) return;

    const GpuLight light = lights[lightIdx];
    const uint lt = asuint(light.typeF);

    // Directional lights affect all tiles.
    if (lt == kLightDirectional) {
        const uint tileCount = tilesXY.x * tilesXY.y;
        for (uint t = 0; t < tileCount; ++t) {
            uint prev = 0;
            InterlockedAdd(tileLightCounts[t], 1u, prev);
            if (prev < kMaxLightsPerTile) {
                tileLightIndices[t * kMaxLightsPerTile + prev] = lightIdx;
            }
        }
        return;
    }

    // Transform light position to view space.
    // Matrix convention: `view` is uploaded transposed (row-major for Slang), so
    // the row-vector form mul(vec, M) yields V·p — matching the mesh shader.
    const float4 posV4 = mul(float4(light.position, 1.0), view);
    const float3 posV = posV4.xyz;
    const float  radius = (light.range > 0.0) ? light.range : 1e6;

    // Project bounding sphere to screen to find tile range.
    // Use conservative AABB projection (approximate, but fast).
    // Extract projection scale factors: p[0][0] = 2n/(r-l) ≈ 2f/w
    const float sx = proj[0][0]; // = 2*near / (right - left)
    const float sy = proj[1][1]; // = 2*near / (top - bottom)

    // Project sphere centre + radius to NDC (approximate, not exact for off-axis).
    // Only use lights in front of camera (posV.z < 0 in right-handed view space).
    const float depth = -posV.z; // positive when in front
    if (depth + radius <= nearZ || depth - radius >= farZ) return; // depth cull

    float xNdc, yNdc, dxNdc, dyNdc;
    if (depth > 0.001) {
        xNdc  = sx * posV.x / depth;
        yNdc  = sy * posV.y / depth;
        // Extent magnitude must be sign-independent: proj[1][1] is negative due to
        // the Vulkan Y-flip, so use abs() or the tile range inverts (ty0 > ty1).
        dxNdc = abs(sx) * radius / depth;
        dyNdc = abs(sy) * radius / depth;
    } else {
        // Very close: cover all tiles
        xNdc = 0.0; yNdc = 0.0; dxNdc = 2.0; dyNdc = 2.0;
    }

    // Map NDC [-1,1] to tile indices [0, tilesXY).
    const float tx0f = (xNdc - dxNdc + 1.0) * 0.5 * float(tilesXY.x);
    const float tx1f = (xNdc + dxNdc + 1.0) * 0.5 * float(tilesXY.x);
    const float ty0f = (yNdc - dyNdc + 1.0) * 0.5 * float(tilesXY.y);
    const float ty1f = (yNdc + dyNdc + 1.0) * 0.5 * float(tilesXY.y);

    const int tx0 = max(0,                int(floor(tx0f)));
    const int tx1 = min(int(tilesXY.x)-1, int(ceil (tx1f)));
    const int ty0 = max(0,                int(floor(ty0f)));
    const int ty1 = min(int(tilesXY.y)-1, int(ceil (ty1f)));

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            const uint tileId = uint(ty) * tilesXY.x + uint(tx);
            uint prev = 0;
            InterlockedAdd(tileLightCounts[tileId], 1u, prev);
            if (prev < kMaxLightsPerTile) {
                tileLightIndices[tileId * kMaxLightsPerTile + prev] = lightIdx;
            }
        }
    }
}
