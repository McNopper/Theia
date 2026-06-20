// Fragment shader: OpenPBR BSDF direct lighting via evalBSDF() from Hyperion.
// Direct lights (point/spot/directional/rect) evaluated analytically.
// Rect lights are treated as point sources at their center for this initial pass;
// LTC area-light integration is a planned optimization (Heitz 2016, Theia roadmap).
// Exposure (1/(1.2*2^EV100)) applied here; ToneMapper curve runs in post.

import common; // GpuMaterial, GpuLight, GpuEmissiveTriangle, kLight* constants
import bsdf;   // evalBSDF() -- imports common + math transitively
import math;   // fresnelDielectric()

[[vk::push_constant]]
cbuffer ForwardPC {
    float4x4 viewProj;
    float4x4 view;
    float4   cameraPos;             // xyz = world-space camera position
    float    exposure;              // 1 / (1.2 * 2^EV100)
    uint     lightCount;
    uint     emissiveTriangleCount;
    uint     tilesX;                // tiles per screen row (screenWidth / 16)
    uint     tilesY;                // tiles per screen column (screenHeight / 16)
    uint     screenWidth;
    uint     screenHeight;
    uint     _pad;
    float4   sunDirection;          // xyz = world dir toward sun, w = shadow strength (0 disables)
    float4   shadowParams;          // x = ray tMin, y = sky ambient floor, z = env_unit_nits
    float4   presentationParams;    // x = indirect ambient strength (presentation-only), yzw reserved
};

// Set 1: material / lighting buffers
[[vk::binding(0, 1)]] StructuredBuffer<GpuMaterial>         materials;
[[vk::binding(1, 1)]] StructuredBuffer<GpuLight>            lights;
[[vk::binding(2, 1)]] StructuredBuffer<GpuEmissiveTriangle> emissiveTriangles;
// Tile-based light lists (Forward+):
// binding 3: tileLightCounts[tilesX*tilesY] — number of lights per tile
// binding 4: tileLightIndices[tilesX*tilesY * kMaxLightsPerTile] — packed light indices
[[vk::binding(3, 1)]] StructuredBuffer<uint> tileLightCounts;
[[vk::binding(4, 1)]] StructuredBuffer<uint> tileLightIndices;
// Scene top-level acceleration structure for ray-traced shadows (inline ray query).
[[vk::binding(5, 1)]] RaytracingAccelerationStructure t_tlas;

// Set 0 (shared geometry buffers): used by mesh shading and ray-hit reconstruction debug.
[[vk::binding(0, 0)]] StructuredBuffer<GpuVertex>   vertices;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(2, 0)]] StructuredBuffer<uint>        indices;

// Set 2: IBL textures
// Bindings: 0=diffuse irradiance, 1=specular prefilter (mipped), 2=sheen LUT, 3=sampler, 4=raw env, 5=GGX BRDF LUT
[[vk::binding(0, 2)]] Texture2D<float4> t_iblDiffuse;
[[vk::binding(1, 2)]] Texture2D<float4> t_iblSpecular;
[[vk::binding(2, 2)]] Texture2D<float>  t_sheenLut;
[[vk::binding(3, 2)]] SamplerState      s_iblLinear;
[[vk::binding(4, 2)]] Texture2D<float4> t_envRaw;
[[vk::binding(5, 2)]] Texture2D<float2> t_brdfLut;

// Set 3: bindless material textures (base_color, normal, ORM, emission).
// Indexed by GpuMaterial::textureIndices; kNoTexture (~0u) means "no map, use constant".
[[vk::binding(0, 3)]] Sampler2D t_bindless[];

struct FS_IN {
    float4 pos           : SV_Position;
    float3 normal        : NORMAL;
    float3 tangent       : TANGENT;
    float  bitangentSign : TEXCOORD3;
    float3 worldPos      : TEXCOORD0;
    float2 uv            : TEXCOORD1;
    nointerpolation uint materialIdx : TEXCOORD2;
};

/// Output: HDR color (target 0) + thin GBuffer (target 1).
/// GBuffer: xyz = view-space normal * 0.5 + 0.5 (range [0,1]), w = base roughness.
/// Stored in VIEW space so the SSR compute shader can ray-march without a view matrix.
struct FS_OUT {
    float4 hdrColor : SV_Target0;
    float4 gbuffer  : SV_Target1; // xyz = viewN [0,1], w = roughness
};

float2 dirToLatlong(float3 d) {
    // Hyperion env.slang convention (shaders/env.slang): u via atan2(z,x), v via acos(y).
    d = normalize(d);
    float longitude = atan2(d.z, d.x) * (1.0 / (2.0 * 3.14159265)) + 0.5;
    float latitude  = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265; // v=0 at top (y=+1)
    return float2(longitude, latitude);
}

float3 sampleEnvRaw(float3 dir, float envScale) {
    const float2 uv = dirToLatlong(dir);
    return envScale * max(t_envRaw.SampleLevel(s_iblLinear, uv, 0.0).xyz, 0.0.xxx);
}

float3 computeTransmissionTint(GpuMaterial mat) {
    // OpenPBR transmission tint used consistently across IBL and RT transparent paths.
    float3 transColor = saturate3(mat.transmissionColorWeight.xyz);
    float3 scatter = saturate3(mat.transmissionScatter.xyz);
    float scatterG = clamp(mat.transmissionScatter.w, -1.0, 1.0);
    float scatterAmt = saturate(dot(scatter, kRec2020Luma)) * saturate(0.5 * (1.0 - scatterG));
    float3 effTransColor = transColor * lerp(1.0.xxx, scatter, scatterAmt);
    float transDepth = max(mat.transmissionParams.x, 0.001);
    float3 absorption = exp(-transDepth * (1.0.xxx - effTransColor));
    return effTransColor * absorption;
}

/// MaterialX analytic GGX directional albedo (rational polynomial fit).
/// Returns (A, B) where the directional albedo = F0*A + F90*B.
/// Ref: MaterialX libraries/pbrlib/genglsl/lib/mx_microfacet_specular.glsl
///      mx_ggx_dir_albedo_analytic()
float2 mx_ggx_dir_albedo_AB(float NdotV, float alpha) {
    float x = NdotV, y = alpha;
    float x2 = x * x, y2 = y * y;
    float4 r = float4( 0.1003,  0.9345, 1.0,    1.0   )
             + float4(-0.6303, -2.323, -1.765,   0.2281) * x
             + float4( 9.748,   2.229,  8.263,  15.94  ) * y
             + float4(-2.038,  -3.748, 11.53,  -55.83  ) * x * y
             + float4(29.34,    1.424, 28.96,   13.08  ) * x2
             + float4(-8.245,  -0.7684,-7.507,  41.26  ) * y2
             + float4(-26.44,   1.436,-36.11,   54.9   ) * x2 * y
             + float4(19.99,    0.2913,15.86,  300.2   ) * x * y2
             + float4(-5.448,   0.6286,33.37, -285.1   ) * x2 * y2;
    return clamp(r.xy / r.zw, 0.0.xx, 1.0.xx);
}

/// Roughness → mip level for the prefiltered specular map (8 mip levels, linear distribution).
float roughnessToLod(float roughness) {
    return roughness * 7.0; // mip 0 = roughness 0, mip 7 = roughness 1
}

float3 sampleSpecularEnv(float3 dir, float roughness, float gain) {
    const float2 uv = dirToLatlong(dir);
    return gain * t_iblSpecular.SampleLevel(s_iblLinear, uv, roughnessToLod(roughness)).xyz;
}

/// IBL evaluation per OpenPBR layer, matching MaterialX split-sum:
///   Lo_indirect ≈ LD(L, alpha) * FG(NdotV, alpha, Fresnel)
///
/// Coat layer attenuation follows MaterialX mx_layer_bsdf:
///   final = coat_IBL + base_IBL * coat_throughput
///   coat_throughput = 1 - coat_weight * coat_dirAlbedo
///
/// Ref: MaterialX libraries/pbrlib/genglsl/lib/mx_environment_prefilter.glsl
///      MaterialX libraries/pbrlib/genglsl/mx_layer_bsdf.glsl
float3 evalIBL(GpuMaterial mat, float3 wo, float3 N, float3 T, float3 B)
{
    float NoV = saturate(dot(N, wo));
    if (NoV <= 0.0) {
        return 0.0.xxx;
    }

    float diffuseWeight;
    float subsurfaceWeight;
    float specularWeight;
    float metalWeight;
    float transmissionWeight;
    float coatWeight;
    float fuzzWeight;
    float diffuseTransWeight;
    bool mirrorMode;
    bool glassMode;
    computeLobeWeights(mat,
                       diffuseWeight,
                       subsurfaceWeight,
                       specularWeight,
                       metalWeight,
                       transmissionWeight,
                       coatWeight,
                       fuzzWeight,
                       diffuseTransWeight,
                       mirrorMode,
                       glassMode);

    float opacity = saturate(mat.opacityFlagsPad.x);
    float ssAniso = clamp(mat.opacityFlagsPad.z, -1.0, 1.0);
    float baseRough = saturate(mat.specularRoughAnisoIor.x);
    float coatRough = saturate(mat.coatRoughAnisoIorDark.x);
    float eta = max(mat.specularRoughAnisoIor.z, 1.01);
    float coatEta = max(mat.coatRoughAnisoIorDark.z, 1.01);
    float coatDark = saturate(mat.coatRoughAnisoIorDark.w);
    float metalness = saturate(mat.baseMetalnessDiffRough.x);
    float diffuseRough = saturate(mat.baseMetalnessDiffRough.y);

    float3 baseColor = saturate3(mat.baseColorWeight.xyz);
    float3 specColor = saturate3(mat.specularColorWeight.xyz);
    float3 subsurfaceColor = saturate3(mat.subsurfaceColorWeight.xyz);
    float3 coatColor = saturate3(mat.coatColorWeight.xyz);
    float3 fuzzColor = saturate3(mat.fuzzColorWeight.xyz);

    float baseLayerScale = opacity * rcp(lerp(1.0, coatEta * coatEta, coatWeight * coatDark));
    baseLayerScale *= 1.0 - 0.35 * fuzzWeight;

    float3 diffuseIrradiance = t_iblDiffuse.SampleLevel(s_iblLinear, dirToLatlong(N), 0.0).xyz;
    float3 R = reflect(-wo, N);
    // Anisotropic specular IBL: bend the base reflection vector toward the bitangent
    // so the prefiltered highlight elongates along the tangent, approximating the
    // anisotropic GGX lobe under environment lighting (Filament getReflectedVector /
    // MaterialX real-time). Matches the BSDF's computeAlpha convention where positive
    // specular_anisotropy widens roughness along the tangent (alphaX > alphaY).
    float baseAniso = saturate(mat.specularRoughAnisoIor.y);
    float3 Rbase = R;
    if (baseAniso > 0.0) {
        float3 anisoTangent = cross(B, wo);
        float3 anisoNormal  = cross(anisoTangent, B);
        float  bend         = baseAniso * saturate(5.0 * baseRough);
        Rbase = reflect(-wo, normalize(lerp(N, anisoNormal, bend)));
    }
    // Compensation for split-sum under-energy vs path-traced multi-bounce specular.
    // Stronger for smooth lobes (where missing high-frequency env energy is most visible),
    // tapering toward rough surfaces.
    const float iblSpecularGainBase = lerp(1.70, 1.00, saturate(baseRough * 1.35));
    const float iblSpecularGainCoat = lerp(1.55, 1.00, saturate(coatRough * 1.35));
    float3 prefilteredBase = sampleSpecularEnv(Rbase, baseRough, iblSpecularGainBase);
    float3 prefilteredCoat = sampleSpecularEnv(R, coatRough, iblSpecularGainCoat);

    // Base layer IBL: specular computed first so its directional albedo (FG) can
    // mask the diffuse — matching MaterialX mx_layer_bsdf energy conservation.
    // Diffuse irradiance map stores E = integral(Li*cos(theta) dw); the OpenPBR
    // Fujii BRDF has Lo_diffuse = kInvPi * E * diffuseColorDirAlbedo(NoV, rough, color),
    // which reduces to the Lambertian kInvPi*E*color when roughness = 0.
    float3 baseLo = 0.0.xxx;
    float3 specularFG = 0.0.xxx;
    if (specularWeight > 0.0 || metalWeight > 0.0) {
        float3 glossyF0 = lerp(iorToF0(eta) * specColor, baseColor, metalness);
        glossyF0 = thinFilmTint(glossyF0, NoV, mat.thinFilmParams.x, max(mat.thinFilmParams.y, 1.0), mat.thinFilmParams.z);
        // F90 = specColor for metallic workflow; 1 for pure dielectric
        float3 specF90 = lerp(1.0.xxx, specColor, metalness);
        float2 AB = t_brdfLut.SampleLevel(s_iblLinear, float2(saturate(NoV), baseRough), 0.0);
        specularFG = (specularWeight + metalWeight) * (glossyF0 * AB.x + specF90 * AB.y);
        baseLo += baseLayerScale * prefilteredBase * specularFG;
    }
    // Energy transmitted through the specular layer reaches the diffuse sublayer.
    float3 diffuseMask = saturate(1.0.xxx - specularFG);
    if (diffuseWeight > 0.0) {
        baseLo += baseLayerScale * diffuseWeight * kInvPi * diffuseIrradiance * diffuseColorDirAlbedo(NoV, diffuseRough, baseColor) * diffuseMask;
    }
    if (subsurfaceWeight > 0.0) {
        float ssScale = max(mat.subsurfaceRadiusScale.w, 0.001);
        float3 ssTint = subsurfaceColor * (1.0 / (1.0 + ssScale *
            (mat.subsurfaceRadiusScale.x + mat.subsurfaceRadiusScale.y + mat.subsurfaceRadiusScale.z)));
        float ssReflScale = (mat.opacityFlagsPad.w > 0.5) ? 1.0 : saturate(1.0 - ssAniso);
        baseLo += baseLayerScale * subsurfaceWeight * ssReflScale * kInvPi * diffuseIrradiance * diffuseColorDirAlbedo(NoV, saturate(0.25 + 0.5 * diffuseRough), ssTint) * diffuseMask;
    }
    // Thin-walled diffuse transmission (translucent base): diffuse light from the
    // BACK hemisphere transmitted through the sheet, tinted by subsurface_color.
    // IBL analogue of bsdf.slang's evalDiffuse on the flipped
    // hemisphere — the back-side irradiance is the cosine-prefiltered diffuse map
    // sampled along -N. Lets thin curtains/leaves/paper glow when lit from behind.
    if (diffuseTransWeight > 0.0) {
        float3 backIrradiance = t_iblDiffuse.SampleLevel(s_iblLinear, dirToLatlong(-N), 0.0).xyz;
        baseLo += baseLayerScale * diffuseTransWeight * kInvPi * backIrradiance * diffuseColorDirAlbedo(NoV, diffuseRough, subsurfaceColor) * diffuseMask;
    }
    if (fuzzWeight > 0.0) {
        // The Disney retroreflective sheen model (evalSheen in bsdf.slang) integrates
        // to approximately Lambertian energy.  Use the diffuse irradiance for a
        // split-sum approximation that matches the path-tracer's energy budget.
        baseLo += fuzzWeight * fuzzColor * kInvPi * diffuseIrradiance;
    }
    // Transmission (glass/dielectric): approximate refraction by sampling the
    // environment in the refracted direction, blended against the Fresnel
    // reflection of the prefiltered base lobe.  This replaces the black glass
    // produced when only the (energy-zero) opaque lobes are evaluated, giving a
    // believable see-through dielectric for env-lit scenes.
    if (transmissionWeight > 0.0) {
        // OpenPBR transmission tint = effective transmission_color (incl. single-scatter
        // re-tint) modulated by Beer-Lambert volume absorption over transmission_depth.
        // Matches Hyperion bsdf.slang effectiveTransColor() * exp(-depth*(1-color)).
        float3 transTint = computeTransmissionTint(mat);
        float Fr = fresnelDielectric(NoV, eta);
        float3 refrDir = refract(-wo, N, 1.0 / eta);
        float3 refracted;
        if (dot(refrDir, refrDir) < 1e-6) {
            // Total internal reflection -> mirror the reflection lobe.
            refracted = prefilteredBase;
        } else {
            // IBL refraction: sample the prefiltered environment radiance in the
            // refracted direction at the transmission roughness (MaterialX
            // mx_environment_prefilter convention for the transmission lobe).
            refracted = sampleSpecularEnv(normalize(refrDir), baseRough, iblSpecularGainBase);
        }
        float3 glass = Fr * prefilteredBase + (1.0 - Fr) * refracted * transTint;
        baseLo += baseLayerScale * transmissionWeight * glass;
    }

    // Coat IBL + throughput (MaterialX mx_layer_bsdf):
    //   final = coat_IBL + base_IBL * coat_throughput
    float3 Lo = baseLo;
    if (coatWeight > 0.0) {
        float3 coatF0 = thinFilmTint(iorToF0(coatEta) * max(coatColor, 0.04.xxx),
                                     NoV,
                                     mat.thinFilmParams.x,
                                     max(mat.thinFilmParams.y, 1.0),
                                     mat.thinFilmParams.z);
        float2 coatAB = t_brdfLut.SampleLevel(s_iblLinear, float2(saturate(NoV), coatRough), 0.0);
        float3 coatFG = coatF0 * coatAB.x + 1.0.xxx * coatAB.y; // F90=1 for dielectric coat
        float3 coatIBL = coatWeight * prefilteredCoat * coatFG;
        float3 coatThroughput = 1.0.xxx - coatWeight * coatFG;
        Lo = coatIBL + baseLo * coatThroughput;
    }

    return max(Lo, 0.0.xxx);
}

/// Trace a shadow ray against the scene TLAS using inline ray query.
/// Returns 1.0 if the ray reaches `tMax` unobstructed (lit), 0.0 if occluded.
/// The scene BLASes are opaque triangle geometry, so a single Proceed() suffices.
float traceShadow(float3 origin, float3 dir, float tMin, float tMax) {
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = dir;
    ray.TMin      = tMin;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(t_tlas, RAY_FLAG_NONE, kInstanceMaskNonEmissive, ray);
    q.Proceed();
    return (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}

/// Evaluate incoming radiance from a single light source.
/// Returns Li; sets wi to the incoming direction (surface to light).
/// Returns zero radiance when light is out of range or behind.
float3 evalLightWi(GpuLight light, float3 worldPos, out float3 wi) {
    const uint lt = asuint(light.typeF);

    if (lt == kLightPoint) {
        const float3 toLight = light.position - worldPos;
        const float  distSq  = dot(toLight, toLight);
        const float  dist    = sqrt(distSq);
        if (light.range > 0.0 && dist > light.range) { wi = 0..xxx; return 0..xxx; }
        wi = toLight / dist;
        return light.color * (light.intensity / distSq);
    }

    if (lt == kLightSpot) {
        const float3 toLight = light.position - worldPos;
        const float  dist    = length(toLight);
        if (light.range > 0.0 && dist > light.range) { wi = 0..xxx; return 0..xxx; }
        wi = toLight / dist;
        const float cosA = dot(-wi, normalize(light.direction));
        const float spot = saturate((cosA - light.cosOuter) / max(light.cosInner - light.cosOuter, 1e-4));
        if (spot <= 0.0) { wi = 0..xxx; return 0..xxx; }
        return light.color * (light.intensity * spot / (dist * dist));
    }

    if (lt == kLightDirectional) {
        wi = -normalize(light.direction);
        return light.color * light.intensity;
    }

    if (lt == kLightRect) {
        // Approximate as point at center (LTC integration planned for Phase 6).
        const float3 toLight = light.position - worldPos;
        const float  distSq  = dot(toLight, toLight);
        const float  dist    = sqrt(distSq);
        wi = toLight / dist;
        // One-sided emission: the area light only emits toward the scene along
        // light.direction (matches Hyperion's emissive-triangle front face). The
        // contribution scales by how much the emitter faces the receiver (cos_l);
        // without this the surfaces directly facing the light blow out to white.
        const float cosL = max(dot(-wi, normalize(light.direction)), 0.0);
        if (cosL <= 0.0) { return 0..xxx; }
        const float area = 4.0 * light.halfWidth * light.halfHeight;
        return light.color * (light.intensity * area * cosL / distSq);
    }

    wi = 0..xxx;
    return 0..xxx; // sky -> IBL
}

/// Shade a rect/area light by numerically integrating the rendering equation over
/// the emitter surface (NxN samples). A single point approximation produces a hard
/// 1/dist² hotspot; integrating over the extended source gives the smooth, even
/// illumination of a real area light (matches Hyperion's emissive-mesh result).
/// Visibility is evaluated per sample to preserve soft-shadow penumbra quality.
/// Lo = ∫ f(wo,wi) · L · cos_i · cos_l / r² dA   (L = light.color·intensity, W/sr/m²)
float3 shadeRectLight(GpuLight light, GpuMaterial mat, float3 worldPos, float3 wo,
                      float3 N, float3 T, float3 B, float3 shadowOrigin, float shadowBias) {
    const float3 Ln = normalize(light.direction); // emitter normal (toward scene)

    // Build an orthonormal tangent frame on the emitter plane.
    const float3 up = abs(Ln.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 Lt = normalize(cross(up, Ln));
    const float3 Lb = cross(Ln, Lt);

    const int   NS = 11;
    const float dA = (4.0 * light.halfWidth * light.halfHeight) / float(NS * NS);

    float3 acc = 0..xxx;
    [unroll]
    for (int sy = 0; sy < NS; ++sy) {
        [unroll]
        for (int sx = 0; sx < NS; ++sx) {
            const float u = ((float(sx) + 0.5) / float(NS)) * 2.0 - 1.0; // [-1,1]
            const float v = ((float(sy) + 0.5) / float(NS)) * 2.0 - 1.0;
            const float3 sp = light.position + Lt * (u * light.halfWidth)
                                             + Lb * (v * light.halfHeight);
            const float3 d      = sp - worldPos;
            const float  distSq = dot(d, d);
            const float  dist   = sqrt(distSq);
            const float3 wi     = d / dist;
            const float  NdotL  = max(dot(N, wi), 0.0);
            if (NdotL <= 0.0) continue;
            const float cosL = max(dot(-wi, Ln), 0.0); // one-sided emission
            if (cosL <= 0.0) continue;
            const float tMax = max(dist - shadowBias, shadowBias * 2.0);
            const float vis = traceShadow(shadowOrigin, wi, shadowBias, tMax);
            if (vis <= 0.0) continue;
            acc += evalBSDF(mat, wo, wi, N, T, B, N, T, B)
                 * (light.intensity * NdotL * cosL * dA * vis / distSq);
        }
    }
    return acc * light.color;
}

/// Full direct-lighting contribution of one light, including shadow + BSDF.
/// Rect lights use area integration; point/spot/directional use the point form.
float3 shadeOneLight(GpuLight light, GpuMaterial mat, float3 worldPos, float3 wo,
                     float3 N, float3 T, float3 B, float3 shadowOrigin, float shadowBias) {
    const uint lt = asuint(light.typeF);
    if (lt == kLightRect) {
        return shadeRectLight(light, mat, worldPos, wo, N, T, B, shadowOrigin, shadowBias);
    }

    float3 wi;
    const float3 Li = evalLightWi(light, worldPos, wi);
    if (dot(Li, Li) < 1e-12) return 0..xxx;
    const float NdotL = max(dot(N, wi), 0.0);
    if (NdotL <= 0.0) return 0..xxx;
    const float tMax = (lt == kLightDirectional)
        ? 1e16
        : max(length(light.position - worldPos) - light.halfWidth - shadowBias, shadowBias * 2.0);
    const float vis = traceShadow(shadowOrigin, wi, shadowBias, tMax);
    if (vis <= 0.0) return 0..xxx;
    return evalBSDF(mat, wo, wi, N, T, B, N, T, B) * Li * NdotL * vis;
}

struct RayHitData {
    bool  hit;
    uint  materialIdx;
    float3 worldPos;
    float3 normal;
};

RayHitData traceClosestHit(float3 originWs, float3 dirWs, float tMin, uint instanceMask) {
    RayHitData hit;
    hit.hit = false;
    hit.materialIdx = 0u;
    hit.worldPos = 0.0.xxx;
    hit.normal = float3(0.0, 1.0, 0.0);

    const float3 rayDir = normalize(dirWs);
    RayDesc ray;
    ray.Origin = originWs;
    ray.Direction = rayDir;
    ray.TMin = tMin;
    ray.TMax = kInf;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(t_tlas, RAY_FLAG_NONE, instanceMask, ray);
    while (q.Proceed()) {}
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return hit;
    }

    const uint instIdx = q.CommittedInstanceID();
    const GpuInstance inst = instances[instIdx];
    const uint tri = q.CommittedPrimitiveIndex();
    const uint base = inst.indexOffset + tri * 3u;
    if (base + 2u >= inst.indexOffset + inst.indexCount) {
        return hit;
    }

    const uint i0 = indices[base + 0u];
    const uint i1 = indices[base + 1u];
    const uint i2 = indices[base + 2u];
    const float3 p0 = vertices[i0].position;
    const float3 p1 = vertices[i1].position;
    const float3 p2 = vertices[i2].position;

    const float2 b = q.CommittedTriangleBarycentrics();
    const float b0 = 1.0 - b.x - b.y;
    const float b1 = b.x;
    const float b2 = b.y;
    hit.worldPos = p0 * b0 + p1 * b1 + p2 * b2;
    float3 Ng = normalize(cross(p1 - p0, p2 - p0));
    if (!q.CommittedTriangleFrontFace()) {
        Ng = -Ng;
    }
    hit.normal = Ng;
    hit.materialIdx = inst.materialIndex;
    hit.hit = true;
    return hit;
}

RayHitData tracePrimaryOpaqueHit(float3 cameraPosWs, float3 targetWorldPos, float tMin) {
    return traceClosestHit(cameraPosWs, targetWorldPos - cameraPosWs, tMin, kInstanceMaskAll);
}

float3 shadeHitRadianceWithWo(RayHitData hit, float3 wo, float shadowBias) {
    const GpuMaterial mat = materials[hit.materialIdx];
    const float3 N = normalize(hit.normal);
    wo = normalize(wo);
    const float3 up = abs(N.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 T = normalize(cross(up, N));
    const float3 B = normalize(cross(N, T));
    const float3 shadowOrigin = hit.worldPos + N * shadowBias;

    float3 directLo = 0.0.xxx;
    for (uint i = 0; i < lightCount; ++i) {
        directLo += shadeOneLight(lights[i], mat, hit.worldPos, wo, N, T, B, shadowOrigin, shadowBias);
    }

    float iblShadow = 1.0;
    const float sunStrength = sunDirection.w;
    if (sunStrength > 0.0) {
        const float3 sunDir = normalize(sunDirection.xyz);
        if (dot(N, sunDir) > 0.0) {
            const float sunVis = traceShadow(shadowOrigin, sunDir, shadowBias, 1e16);
            const float skyFloor = shadowParams.y;
            iblShadow = lerp(1.0, lerp(skyFloor, 1.0, sunVis), sunStrength);
        }
    }
    const float3 indirectLo = evalIBL(mat, wo, N, T, B) * iblShadow;
    const float3 emission = mat.emissionColorLum.xyz * mat.emissionColorLum.w;
    return directLo + indirectLo + emission;
}

float3 shadeHitRadiance(RayHitData hit, float3 viewPos, float shadowBias) {
    return shadeHitRadianceWithWo(hit, viewPos - hit.worldPos, shadowBias);
}

float3 shadeHitRadianceFromRayDir(RayHitData hit, float3 rayDir, float shadowBias) {
    // For traced secondary hits, outgoing direction points toward the previous segment
    // (opposite ray travel), not necessarily toward the camera.
    return shadeHitRadianceWithWo(hit, -rayDir, shadowBias);
}

float3 traceTransparentPath(float3 originWs, float3 dirWs, float3 viewPos, uint maxBounces, float shadowBias, bool skipFirstWeight = false) {
    float3 accum = 0.0.xxx;
    float3 throughput = 1.0.xxx;
    float3 rayOrigin = originWs;
    float3 rayDir = normalize(dirWs);

    [loop]
    for (uint bounce = 0u; bounce < maxBounces; ++bounce) {
        const RayHitData hit = traceClosestHit(rayOrigin, rayDir, kRayEpsilon, kInstanceMaskNonEmissive);
        if (!hit.hit) {
            accum += throughput * sampleEnvRaw(rayDir, shadowParams.z);
            break;
        }

        const GpuMaterial hitMat = materials[hit.materialIdx];
        const float opacity = saturate(hitMat.opacityFlagsPad.x);
        const float transmissionWeight = saturate(hitMat.transmissionColorWeight.w);
        const bool isTransparent = (opacity < 0.9999) || (transmissionWeight > 0.0);
        if (!isTransparent) {
            accum += throughput * shadeHitRadianceFromRayDir(hit, rayDir, shadowBias);
            break;
        }

        float3 N = normalize(hit.normal);
        float Fr = 0.0;
        float3 refrDir = rayDir;
        if (transmissionWeight > 0.0) {
            const float eta = max(hitMat.specularRoughAnisoIor.z, 1.01);
            float etaRatio = 1.0 / eta;
            if (dot(rayDir, N) > 0.0) {
                N = -N;
                etaRatio = eta;
            }

            const float NoV = saturate(abs(dot(-rayDir, N)));
            Fr = fresnelDielectric(NoV, eta);
            refrDir = refract(rayDir, N, etaRatio);
            const float3 reflDir = normalize(reflect(rayDir, N));
            if (dot(refrDir, refrDir) < 1e-6) {
                refrDir = reflDir;
            } else {
                refrDir = normalize(refrDir);
            }

            const float rough = saturate(hitMat.specularRoughAnisoIor.x);
            float3 reflectedEnv = sampleSpecularEnv(reflDir, rough, 1.0);
            if (rough < 0.1) {
                reflectedEnv = sampleEnvRaw(reflDir, shadowParams.z);
                if (rough < 0.05) {
                    const float reflSide = dot(reflDir, N) >= 0.0 ? 1.0 : -1.0;
                    const float3 reflOrigin = hit.worldPos + N * (reflSide * kRayEpsilon);
                    const RayHitData reflHit =
                        traceClosestHit(reflOrigin, reflDir, kRayEpsilon, kInstanceMaskNonEmissive);
                    if (reflHit.hit) {
                        const GpuMaterial reflMat = materials[reflHit.materialIdx];
                        const float reflOpacity = saturate(reflMat.opacityFlagsPad.x);
                        const float reflTransWeight = saturate(reflMat.transmissionColorWeight.w);
                        const bool reflIsTransparent = (reflOpacity < 0.9999) || (reflTransWeight > 0.0);
                        if (!reflIsTransparent) {
                            reflectedEnv = shadeHitRadianceFromRayDir(reflHit, reflDir, shadowBias);
                        }
                    }
                }
            }
            accum += throughput * Fr * reflectedEnv;
        }

        const float3 transTint = computeTransmissionTint(hitMat);
        const float coverageWeight = 1.0 - opacity;
        const float transWeight = transmissionWeight * (1.0 - Fr);
        const bool continueAsTransmission = transmissionWeight > 0.0;
        // The caller applies current-surface geometry_opacity at composition time. When asked,
        // skip weighting only on the first transparent interaction to avoid double-applying alpha.
        const bool skipWeightHere = skipFirstWeight && (bounce == 0u);
        const float continueWeight = skipWeightHere ? 1.0 : (continueAsTransmission ? transWeight : coverageWeight);
        const float3 continueTint = continueAsTransmission ? transTint : 1.0.xxx;
        throughput *= continueWeight * continueTint;
        if (max(throughput.x, max(throughput.y, throughput.z)) < 1e-3) {
            break;
        }

        rayDir = continueAsTransmission ? refrDir : rayDir;
        const float side = dot(rayDir, N) >= 0.0 ? 1.0 : -1.0;
        rayOrigin = hit.worldPos + N * (side * kRayEpsilon);
    }

    return accum;
}

[shader("fragment")]
FS_OUT main(FS_IN input) {
    GpuMaterial mat = materials[input.materialIdx];

    // 26b debug modes (primary-ray hit reconstruction), gated and off by default:
    // presentationParams.z >= 1: show ray-hit base color
    // presentationParams.z >= 2: show ray-hit shaded radiance (no texture normal maps)
    // presentationParams.z >= 3: show instance index as color (debug indexing)
    // presentationParams.z >= 5: show transparent pass surfaceLo
    // presentationParams.z >= 6: show transparent pass behindLo
    if (presentationParams.z >= 3.0 && presentationParams.z < 4.0 && presentationParams.y >= 0.5) {
        const RayHitData hit = tracePrimaryOpaqueHit(cameraPos.xyz, input.worldPos, kRayEpsilon);
        FS_OUT outDbg;
        outDbg.gbuffer = float4(0.5, 0.5, 0.5, 1.0);
        if (!hit.hit) {
            outDbg.hdrColor = float4(0.0, 0.0, 0.0, 1.0);
            return outDbg;
        }
        // Visualize instance index: 0=red, 1=green, 2=blue
        float3 debugColor = float3(0.0, 0.0, 0.0);
        if (hit.materialIdx == 0u) debugColor = float3(1.0, 0.0, 0.0);
        else if (hit.materialIdx == 1u) debugColor = float3(0.0, 1.0, 0.0);
        else if (hit.materialIdx == 2u) debugColor = float3(0.0, 0.0, 1.0);
        else debugColor = float3(1.0, 1.0, 0.0); // yellow for unexpected
        outDbg.hdrColor = float4(debugColor, 1.0);
        return outDbg;
    }
    if (presentationParams.z >= 1.0 && presentationParams.z < 4.0 && presentationParams.y < 0.5) {
        const float shadowBias = shadowParams.x;
        const RayHitData hit = tracePrimaryOpaqueHit(cameraPos.xyz, input.worldPos, kRayEpsilon);
        FS_OUT outDbg;
        outDbg.gbuffer = float4(0.5, 0.5, 0.5, 1.0);
        if (!hit.hit) {
            outDbg.hdrColor = float4(0.0, 0.0, 0.0, 1.0);
            return outDbg;
        }

        const GpuMaterial hitMat = materials[hit.materialIdx];
        if (presentationParams.z < 2.0) {
            outDbg.hdrColor = float4(saturate3(hitMat.baseColorWeight.xyz), 1.0);
            return outDbg;
        }

        outDbg.hdrColor = float4(exposure * shadeHitRadiance(hit, cameraPos.xyz, shadowBias), 1.0);
        return outDbg;
    }

    // Reconstruct orthonormal tangent frame.
    float3 N = normalize(input.normal);
    // Gram-Schmidt re-orthogonalize T against N.
    // Fallback to a frame derived from N when the mesh provides no tangent (e.g. OBJ without UV).
    float3 rawT = input.tangent - dot(input.tangent, N) * N;
    float3 T;
    if (dot(rawT, rawT) < 1e-8) {
        const float3 up = abs(N.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        T = normalize(cross(up, N));
    } else {
        T = normalize(rawT);
    }
    float3 B = normalize(cross(N, T) * input.bitangentSign);

    // --- Bindless material textures (base_color, normal, ORM, emission) ---
    // Textures are pre-converted to linear Rec.2020 at load time, so no sRGB decode here.
    const uint baseColorTex = mat.textureIndices.x;
    const uint normalTex     = mat.textureIndices.y;
    const uint ormTex        = mat.textureIndices.z;
    const uint emissionTex   = mat.textureIndices.w;

    if (baseColorTex != kNoTexture) {
        float4 bc = t_bindless[NonUniformResourceIndex(baseColorTex)].Sample(input.uv);
        mat.baseColorWeight.xyz *= bc.xyz;
        mat.opacityFlagsPad.x   *= bc.w; // modulate opacity by texture alpha
    }
    if (ormTex != kNoTexture) {
        // glTF ORM convention: r = occlusion, g = roughness, b = metalness.
        float3 orm = t_bindless[NonUniformResourceIndex(ormTex)].Sample(input.uv).xyz;
        mat.specularRoughAnisoIor.x = saturate(mat.specularRoughAnisoIor.x * orm.y);
        mat.baseMetalnessDiffRough.x = saturate(mat.baseMetalnessDiffRough.x * orm.z);
    }
    if (emissionTex != kNoTexture) {
        mat.emissionColorLum.xyz *= t_bindless[NonUniformResourceIndex(emissionTex)].Sample(input.uv).xyz;
    }
    if (normalTex != kNoTexture) {
        // Tangent-space normal map → world space using the TBN frame.
        // glTF/OpenGL convention: +Y. Reconstruct Z for robustness against JPG artifacts.
        float3 tn = t_bindless[NonUniformResourceIndex(normalTex)].Sample(input.uv).xyz * 2.0 - 1.0;
        tn.z = sqrt(saturate(1.0 - tn.x * tn.x - tn.y * tn.y));
        float3 perturbed = normalize(tn.x * T + tn.y * B + tn.z * N);
        // Guard against degenerate/encoded normals that would flip the shading normal
        // away from the geometric one (keeps NoV positive for front faces).
        if (dot(perturbed, N) > 0.0) {
            N = perturbed;
            T = normalize(input.tangent - dot(input.tangent, N) * N);
            B = normalize(cross(N, T) * input.bitangentSign);
        }
    }

    // Outgoing direction: surface to camera.
    const float3 wo = normalize(cameraPos.xyz - input.worldPos);

    // 26c transparent pass: straight-through transmission query against opaque-only mask.
    // Opaque pass already filled the background color/depth; this pass resolves transparent
    // front surfaces by querying behind-radiance and writing final HDR RGB (no depth write).
    if (presentationParams.y >= 0.5) {
        // D1 gate: make the transparent surface behave as if it didn't exist.
        // This emits only behind-radiance along the view ray from the front hit.
        if (presentationParams.z >= 4.0 && presentationParams.z < 5.0) {
            const float3 passThroughDir = -wo;
            const float3 passThroughLo = traceTransparentPath(input.worldPos,
                                                              passThroughDir,
                                                              cameraPos.xyz,
                                                              2u,
                                                              shadowParams.x,
                                                              true);
            FS_OUT outNoOp;
            outNoOp.hdrColor = float4(passThroughLo * exposure, 0.0);
            outNoOp.gbuffer = float4(0.5, 0.5, 0.5, 1.0);
            return outNoOp;
        }

        const float opacity = saturate(mat.opacityFlagsPad.x);
        const float transmissionWeight = saturate(mat.transmissionColorWeight.w);
        const float coverageWeight = 1.0 - opacity;
        const float3 coverageRayDir = -wo;
        const float3 coverageBehindLo = traceTransparentPath(input.worldPos,
                                                             coverageRayDir,
                                                             cameraPos.xyz,
                                                             2u,
                                                             shadowParams.x,
                                                             true);

        // Coverage path (`opacity < 1`): alpha-style visibility mix between the current surface
        // shading and the behind-radiance. This keeps geometry_opacity separate from transmission.
        RayHitData frontHit;
        frontHit.hit = true;
        frontHit.materialIdx = input.materialIdx;
        frontHit.worldPos = input.worldPos;
        frontHit.normal = N;
        const float3 frontSurfaceLo = shadeHitRadiance(frontHit, cameraPos.xyz, shadowParams.x);
        float3 surfaceLo = frontSurfaceLo;

        // Transmission path (`transmission_weight > 0`): replace IBL-only "refracted env"
        // approximation with scene-traced behind-radiance. In 26d this ray is bent by
        // Snell/IOR (enter/exit aware) before tracing.
        if (transmissionWeight > 0.0) {
            const float eta = max(mat.specularRoughAnisoIor.z, 1.01);
            const float rough = saturate(mat.specularRoughAnisoIor.x);
            float3 Nf = N;
            float etaRatio = 1.0 / eta; // air -> material
            if (dot(-wo, N) < 0.0) {
                Nf = -N; // exiting medium
                etaRatio = eta;
            }
            float3 transmissionDir = refract(-wo, Nf, etaRatio);
            if (dot(transmissionDir, transmissionDir) < 1e-6) {
                transmissionDir = reflect(-wo, Nf); // total internal reflection fallback
            } else {
                transmissionDir = normalize(transmissionDir);
            }

            // traceTransparentPath already handles Fresnel weighting internally (line 598 adds
            // Fr*reflection, line 603/607 continues with (1-Fr)*transTint throughput), so use result directly.
            const float3 transmissionLo = traceTransparentPath(input.worldPos,
                                                               transmissionDir,
                                                               cameraPos.xyz,
                                                               2u,
                                                               shadowParams.x);
            // Apply transmission on the covered/front surface response first.
            // geometry_opacity then controls visibility of that surface vs behind-radiance.
            surfaceLo = lerp(frontSurfaceLo, transmissionLo, transmissionWeight);
        }

        if (presentationParams.z >= 5.0 && presentationParams.z < 6.0) {
            FS_OUT outSurface;
            outSurface.hdrColor = float4(surfaceLo * exposure, 0.0);
            outSurface.gbuffer = float4(0.5, 0.5, 0.5, 1.0);
            return outSurface;
        }
        if (presentationParams.z >= 6.0) {
            FS_OUT outBehind;
            outBehind.hdrColor = float4(coverageBehindLo * exposure, 0.0);
            outBehind.gbuffer = float4(0.5, 0.5, 0.5, 1.0);
            return outBehind;
        }

        // geometry_opacity coverage compositing: premultiplied-alpha "over" operator.
        // evalIBL/shadeHitRadiance scale the base layer by opacity (matching Hyperion bsdf.slang:56),
        // so surfaceLo is premultiplied; compose via standard over: A + B*(1-alphaA).
        const float3 outLo = surfaceLo + coverageBehindLo * coverageWeight;

        FS_OUT outTransparent;
        outTransparent.hdrColor = float4(outLo * exposure, 0.0);
        outTransparent.gbuffer = float4(0.5, 0.5, 0.5, 1.0);
        return outTransparent;
    }

    // Match Hyperion emissive-hit behavior: emissive triangles are one-sided and
    // terminate primary shading when seen from their emitting side.
    const float3 rawEmission = mat.emissionColorLum.xyz * mat.emissionColorLum.w;
    const float3 emission = (dot(wo, input.normal) > 0.0) ? rawEmission : 0..xxx;
    const bool hitVisibleEmitter = max(emission.x, max(emission.y, emission.z)) > 0.0;

    float3 directLo = 0..xxx;
    float3 indirectLo = 0..xxx;

    // Ray-traced shadows: offset the ray origin slightly along the shading normal to
    // avoid self-intersection. `bias` is a scene-scale epsilon (camera near plane).
    const float  shadowBias   = shadowParams.x;
    const float3 shadowOrigin = input.worldPos + N * shadowBias;

    if (!hitVisibleEmitter) {
        // Tile-based direct lighting (Forward+).
        // When tilesX == 0 (fallback), evaluate all lights directly.
        static const uint kMaxLightsPerTile = 128;
        if (tilesX > 0 && tilesY > 0) {
            const uint tileX = uint(input.pos.x) / 16u;
            const uint tileY = uint(input.pos.y) / 16u;
            const uint tileId = clamp(tileY, 0u, tilesY - 1u) * tilesX
                              + clamp(tileX, 0u, tilesX - 1u);
            const uint tileCount = min(tileLightCounts[tileId], kMaxLightsPerTile);
            for (uint j = 0; j < tileCount; ++j) {
                const uint i = tileLightIndices[tileId * kMaxLightsPerTile + j];
                if (i >= lightCount) continue;
                directLo += shadeOneLight(lights[i], mat, input.worldPos, wo, N, T, B, shadowOrigin, shadowBias);
            }
        } else {
            // Fallback: evaluate all lights (used when light culler hasn't run).
            for (uint i = 0; i < lightCount; ++i) {
                directLo += shadeOneLight(lights[i], mat, input.worldPos, wo, N, T, B, shadowOrigin, shadowBias);
            }
        }

        // IBL with ray-traced directional ("sun") shadow.  The brightest region of the
        // environment is treated as a dominant light; points it cannot see keep only a
        // `skyFloor` fraction of indirect light (residual sky), giving Hyperion-style
        // cast shadows on IBL-lit scenes.  `sunDirection.w` scales the effect by how
        // concentrated the dominant light is (0 → no shadow, e.g. overcast skies).
        float iblShadow = 1.0;
        const float sunStrength = sunDirection.w;
        if (sunStrength > 0.0) {
            const float3 sunDir = normalize(sunDirection.xyz);
            if (dot(N, sunDir) > 0.0) {
                const float sunVis = traceShadow(shadowOrigin, sunDir, shadowBias, 1e16);
                const float skyFloor = shadowParams.y;
                iblShadow = 1.0 - sunStrength * (1.0 - skyFloor) * (1.0 - sunVis);
            }
        }
        indirectLo += evalIBL(mat, wo, N, T, B) * iblShadow;

        const float indirectAmbient = presentationParams.x;
        if (indirectAmbient > 0.0) {
            // Presentation-only fallback for scenes that would otherwise look too
            // contrasty without bounce lighting. Keep disabled for parity runs.
            indirectLo += mat.baseColorWeight.xyz * mat.baseColorWeight.w * indirectAmbient;
        }
    }
    // Emission (self-luminous surfaces, e.g. Cornell box ceiling lamp geometry).
    float3 Lo = directLo + indirectLo + emission;

    // Carry an indirect-light weight in alpha so post-fx can damp ambient/IBL
    // without dimming direct lighting. The tone mapper ignores alpha.
    float indirectWeight = 0.0;
    if (!hitVisibleEmitter) {
        const float directLum = dot(max(directLo, 0.0.xxx), kRec2020Luma);
        const float indirectLum = dot(max(indirectLo, 0.0.xxx), kRec2020Luma);
        indirectWeight = indirectLum / max(directLum + indirectLum, 1e-6);
    }

    // Apply physical camera exposure: maps scene-linear radiance into [0,1] range
    // for the ToneMapper.  Same formula as Hyperion's PhysicalCamera.
    Lo *= exposure;

    // --- GBuffer (target 1): view-space normal + base roughness for SSR ---
    // Transform world-space N into view space.
    // push-constant `view` = glm::transpose(V), so extracting rows gives V's columns.
    // mul(worldN, rows_of_V^T) = worldN * V^T = (V * worldN^T)^T = viewN^T.
    const float3x3 viewRot = float3x3(view[0].xyz, view[1].xyz, view[2].xyz); // rows of V^T
    const float3   viewN   = normalize(mul(N, viewRot)); // view-space normal
    const float    roughness = saturate(mat.specularRoughAnisoIor.x);

    FS_OUT output;
    output.hdrColor = float4(Lo, indirectWeight);
    output.gbuffer  = float4(viewN * 0.5 + 0.5, roughness); // encode to [0,1]
    return output;
}
