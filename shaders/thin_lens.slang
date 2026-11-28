// Deterministic thin-lens primary ray shared by Theia's raster and GI stages.
// A zero radius keeps the exact pinhole origin and direction used before this module existed.
struct LensParams {
    float lensRadius;
    float focusDistance;
    uint lightCount;
    uint analyticLightCount;
};

struct ThinLensPrimaryRay {
    float3 origin;
    float3 direction;
};

ThinLensPrimaryRay makeThinLensPrimaryRay(float3 cameraPosition,
                                           float3 pinDirection,
                                           float3 viewRight,
                                           float3 viewUp,
                                           float lensRadius,
                                           float focusDistance,
                                           float2 diskSample) {
    ThinLensPrimaryRay ray;
    ray.origin = cameraPosition;
    ray.direction = pinDirection;
    if (lensRadius > 0.0) {
        const float3 shiftedOrigin = cameraPosition + viewRight * (diskSample.x * lensRadius) +
                                     viewUp * (diskSample.y * lensRadius);
        const float3 focalPoint = cameraPosition + pinDirection * focusDistance;
        ray.origin = shiftedOrigin;
        ray.direction = normalize(focalPoint - shiftedOrigin);
    }
    return ray;
}
