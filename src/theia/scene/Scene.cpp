#include "theia/scene/Scene.hpp"

#include <slang-math/slang-math.hpp>
#include <volk/volk.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <meshoptimizer.h>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vma/vk_mem_alloc.h>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/scene/EmissiveBuilder.hpp"
#include "harmonia/scene/Geometry.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"

uint32_t Scene::addMesh(const DeviceContext& ctx,
                        const CommandPool& pool,
                        MeshData&& data,
                        uint32_t materialIdx,
                        std::string_view name) {
    const uint32_t instanceIndex = static_cast<uint32_t>(m_geometries.size());
    const std::string debugName =
        name.empty() ? std::string{"mesh."} + std::to_string(instanceIndex) : std::string{name};

    auto mesh = TriangleMesh::create(ctx, pool, std::move(data), materialIdx, debugName);
    if (!mesh) {
        return std::numeric_limits<uint32_t>::max();
    }

    m_geometries.push_back(std::move(*mesh));
    m_instances.push_back(GpuInstance{
        .meshIndex = instanceIndex,
        .materialIndex = materialIdx,
        .vertexOffset = 0,
        .indexOffset = 0,
        .indexCount = 0,
        .meshletOffset = 0,
        .meshletCount = 0,
        .geometryKind = 0,
        .sphereRadius = 0.0f,
    });
    return instanceIndex;
}

uint32_t Scene::addSphere(const DeviceContext& ctx,
                          const CommandPool& pool,
                          sm::float3 center,
                          float radius,
                          uint32_t materialIdx) {
    // Theia is a rasterizer (mesh-shader pipeline), so an analytic sphere cannot
    // be drawn directly — it is tessellated into a triangle mesh on insertion.
    // (Hyperion keeps the analytic sphere; that divergence lives in each Scene.)
    MeshData mesh = ProceduralGeometry::makeSphere(center, radius);
    return addMesh(ctx, pool, std::move(mesh), materialIdx, "sphere");
}

VkResult Scene::build(const DeviceContext& ctx, const CommandPool& pool) {
    if (m_geometries.empty()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    assert(m_geometries.size() == m_instances.size());
    if (const VkResult result = buildSceneBuffers(ctx, pool); result != VK_SUCCESS) {
        return result;
    }
    for (auto& geo : m_geometries) {
        if (const VkResult result = geo->buildBlas(ctx, pool); result != VK_SUCCESS) {
            return result;
        }
    }
    return buildTlas(ctx, pool);
}

VkResult Scene::buildSceneBuffers(const DeviceContext& ctx, const CommandPool& pool) {
    std::vector<GpuMaterial> gpuMaterials;
    gpuMaterials.reserve(std::max<size_t>(m_materials.size(), 1));
    for (const Material& material : m_materials) {
        gpuMaterials.push_back(material.gpu());
    }
    if (gpuMaterials.empty()) {
        gpuMaterials.push_back(GpuMaterial{});
    }

    std::vector<GpuVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<GpuMeshlet> gpuMeshlets;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint32_t> meshletTriangles;
    vertices.reserve(std::max<size_t>(m_geometries.size(), 1));

    for (size_t i = 0; i < m_geometries.size(); ++i) {
        GpuInstance& instance = m_instances[i];

        if (const auto* mesh = dynamic_cast<const TriangleMesh*>(m_geometries[i].get())) {
            instance.vertexOffset = static_cast<uint32_t>(vertices.size());
            instance.indexOffset = static_cast<uint32_t>(indices.size());
            instance.geometryKind = 0;

            vertices.insert(vertices.end(), mesh->data().vertices.begin(), mesh->data().vertices.end());

            const auto& localIndices = mesh->data().indices;
            const size_t vertCount = mesh->data().vertices.size();
            const size_t indexCount = localIndices.size();
            instance.indexCount = static_cast<uint32_t>(indexCount);

            for (uint32_t index : localIndices) {
                indices.push_back(index + instance.vertexOffset);
            }

            constexpr size_t kMaxVerts = 64;
            constexpr size_t kMaxTris = 124;
            const size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, kMaxVerts, kMaxTris);

            std::vector<meshopt_Meshlet> rawMeshlets(maxMeshlets);
            std::vector<uint32_t> rawVerts(maxMeshlets * kMaxVerts);
            std::vector<uint8_t> rawTris(maxMeshlets * kMaxTris * 3);

            std::vector<float> positions;
            positions.reserve(vertCount * 3);
            for (const auto& v : mesh->data().vertices) {
                positions.push_back(v.position.x);
                positions.push_back(v.position.y);
                positions.push_back(v.position.z);
            }

            const size_t meshletCount = meshopt_buildMeshlets(rawMeshlets.data(),
                                                              rawVerts.data(),
                                                              rawTris.data(),
                                                              localIndices.data(),
                                                              indexCount,
                                                              positions.data(),
                                                              vertCount,
                                                              sizeof(float) * 3,
                                                              kMaxVerts,
                                                              kMaxTris,
                                                              0.0f);

            instance.meshletOffset = static_cast<uint32_t>(gpuMeshlets.size());
            instance.meshletCount = static_cast<uint32_t>(meshletCount);

            for (size_t m = 0; m < meshletCount; ++m) {
                const meshopt_Meshlet& ml = rawMeshlets[m];
                const meshopt_Bounds bounds = meshopt_computeMeshletBounds(rawVerts.data() + ml.vertex_offset,
                                                                           rawTris.data() + ml.triangle_offset,
                                                                           ml.triangle_count,
                                                                           positions.data(),
                                                                           vertCount,
                                                                           sizeof(float) * 3);

                const uint32_t gpuVertBase = static_cast<uint32_t>(meshletVertices.size());
                for (uint32_t v = 0; v < ml.vertex_count; ++v) {
                    meshletVertices.push_back(instance.vertexOffset + rawVerts[ml.vertex_offset + v]);
                }

                const uint32_t gpuTriBase = static_cast<uint32_t>(meshletTriangles.size());
                const uint8_t* triBytes = rawTris.data() + ml.triangle_offset;
                const uint32_t triByteCount = ml.triangle_count * 3;
                const uint32_t triPadded = (triByteCount + 3) & ~3u;
                for (uint32_t b = 0; b < triPadded; b += 4) {
                    uint32_t packed = 0;
                    for (uint32_t k = 0; k < 4; ++k) {
                        if (b + k < triByteCount) {
                            packed |= static_cast<uint32_t>(triBytes[b + k]) << (k * 8);
                        }
                    }
                    meshletTriangles.push_back(packed);
                }

                gpuMeshlets.push_back(GpuMeshlet{
                    .vertexOffset = gpuVertBase,
                    .triangleOffset = gpuTriBase,
                    .vertexCount = ml.vertex_count,
                    .triangleCount = ml.triangle_count,
                    .centerX = bounds.center[0],
                    .centerY = bounds.center[1],
                    .centerZ = bounds.center[2],
                    .radius = bounds.radius,
                });
            }

            // Bounding sphere: centroid method — average vertex position + max-distance radius.
            // World-space (vertices are already in world space).
            {
                const auto& mverts = mesh->data().vertices;
                const float invN = 1.0f / std::max<float>(static_cast<float>(mverts.size()), 1.0f);
                float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                for (const auto& v : mverts) {
                    cx += v.position.x;
                    cy += v.position.y;
                    cz += v.position.z;
                }
                cx *= invN; cy *= invN; cz *= invN;
                float bs = 0.0f;
                for (const auto& v : mverts) {
                    const float dx = v.position.x - cx;
                    const float dy = v.position.y - cy;
                    const float dz = v.position.z - cz;
                    bs = std::max(bs, std::sqrt(dx*dx + dy*dy + dz*dz));
                }
                m_instanceBounds.push_back(GpuInstanceBounds{cx, cy, cz, bs});
            }
        } else if (const auto* sphere = dynamic_cast<const Sphere*>(m_geometries[i].get())) {
            instance.vertexOffset = static_cast<uint32_t>(vertices.size());
            instance.indexOffset = 0;
            instance.indexCount = 0;
            instance.meshletOffset = 0;
            instance.meshletCount = 0;
            instance.geometryKind = 1;
            vertices.push_back(GpuVertex{
                .position = sphere->center(),
                .tangentX = 0.0f,
                .normal = sm::float3{0.0f, 0.0f, 0.0f},
                .tangentY = 0.0f,
                .uv = sm::float2{0.0f, 0.0f},
                .tangentZ = 0.0f,
                .bitangentSign = 1.0f,
            });
            // Sphere primitives: bounds come directly from the analytical sphere params.
            m_instanceBounds.push_back(GpuInstanceBounds{
                sphere->center().x, sphere->center().y, sphere->center().z, sphere->radius()
            });
        }
    }

    if (vertices.empty()) {
        vertices.push_back(GpuVertex{});
    }
    if (indices.empty()) {
        indices.push_back(0);
    }
    // Record the true meshlet count before any sentinel padding below. The visibility
    // buffer is sized from this; the padding meshlet (index gpuMeshlets.size()) is never
    // referenced by any instance's [meshletOffset, meshletOffset+meshletCount) range.
    m_meshletCount = static_cast<uint32_t>(gpuMeshlets.size());
    if (gpuMeshlets.empty()) {
        gpuMeshlets.push_back(GpuMeshlet{});
    }
    if (meshletVertices.empty()) {
        meshletVertices.push_back(0);
    }
    if (meshletTriangles.empty()) {
        meshletTriangles.push_back(0);
    }

    auto instanceBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const GpuInstance>(m_instances.data(), m_instances.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.instances");
    if (!instanceBuf) {
        return instanceBuf.error();
    }

    // Ensure m_instanceBounds is the same size as m_instances (guard against missing push_back).
    m_instanceBounds.resize(m_instances.size());
    auto instanceBoundsBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const GpuInstanceBounds>(m_instanceBounds.data(), m_instanceBounds.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       "scene.instanceBounds");
    if (!instanceBoundsBuf) {
        return instanceBoundsBuf.error();
    }

    auto materialBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const GpuMaterial>(gpuMaterials.data(), gpuMaterials.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.materials");
    if (!materialBuf) {
        return materialBuf.error();
    }

    auto vertexBuf = Buffer::upload(ctx,
                                    pool,
                                    std::as_bytes(std::span<const GpuVertex>(vertices.data(), vertices.size())),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    "scene.vertices");
    if (!vertexBuf) {
        return vertexBuf.error();
    }

    auto indexBuf = Buffer::upload(ctx,
                                   pool,
                                   std::as_bytes(std::span<const uint32_t>(indices.data(), indices.size())),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   "scene.indices");
    if (!indexBuf) {
        return indexBuf.error();
    }

    auto meshletBuf = Buffer::upload(ctx,
                                     pool,
                                     std::as_bytes(std::span<const GpuMeshlet>(gpuMeshlets.data(), gpuMeshlets.size())),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                     "scene.meshlets");
    if (!meshletBuf) {
        return meshletBuf.error();
    }

    auto meshletVertexBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const uint32_t>(meshletVertices.data(), meshletVertices.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.meshletVertices");
    if (!meshletVertexBuf) {
        return meshletVertexBuf.error();
    }

    auto meshletTriangleBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const uint32_t>(meshletTriangles.data(), meshletTriangles.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.meshletTriangles");
    if (!meshletTriangleBuf) {
        return meshletTriangleBuf.error();
    }

    m_instanceBuffer = std::move(*instanceBuf);
    m_instanceBoundsBuffer = std::move(*instanceBoundsBuf);
    m_materialBuffer = std::move(*materialBuf);
    m_vertexBuffer = std::move(*vertexBuf);
    m_indexBuffer = std::move(*indexBuf);
    m_meshletBuffer = std::move(*meshletBuf);
    m_meshletVertexBuffer = std::move(*meshletVertexBuf);
    m_meshletTriangleBuffer = std::move(*meshletTriangleBuf);
    Logger::info("Scene built: {} meshlets", gpuMeshlets.size());

    // Per-instance world transforms.  In the current static-scene model, vertex
    // positions are baked into world space at load time (ObjImporter::flushMesh
    // applies the scene-file TRS to every vertex), so all geometry xforms are
    // identity.  These buffers are the A1b infrastructure for future dynamic
    // objects: when an instance's transform changes, callers update
    // m_instanceTransformBuffer and shift the old value into
    // m_prevInstanceTransformBuffer before each frame.
    const size_t instCount = std::max<size_t>(m_geometries.size(), 1);
    std::vector<sm::float4x4> instanceTransforms(instCount, sm::float4x4(1.0f));
    for (size_t i = 0; i < m_geometries.size(); ++i) {
        instanceTransforms[i] = m_geometries[i]->xform.matrix();
    }

    auto transformBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const sm::float4x4>(instanceTransforms.data(), instanceTransforms.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.instanceTransforms");
    if (!transformBuf) {
        return transformBuf.error();
    }
    // Previous-frame transforms start equal to current (first-frame / static default).
    auto prevTransformBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const sm::float4x4>(instanceTransforms.data(), instanceTransforms.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.prevInstanceTransforms");
    if (!prevTransformBuf) {
        return prevTransformBuf.error();
    }

    // Stable per-object IDs: objectId[i] == i.  Persistent across frames as long as
    // the scene is not rebuilt.  Suitable for temporal reservoir keys (ReSTIR DI, A-SVGF).
    std::vector<uint32_t> objectIds(instCount, 0u);
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_geometries.size()); ++i) {
        objectIds[i] = i;
    }
    auto objectIdBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const uint32_t>(objectIds.data(), objectIds.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.objectIds");
    if (!objectIdBuf) {
        return objectIdBuf.error();
    }

    m_instanceTransformBuffer = std::move(*transformBuf);
    m_prevInstanceTransformBuffer = std::move(*prevTransformBuf);
    m_objectIdBuffer = std::move(*objectIdBuf);

    // Light buffer — always upload at least one sentinel entry so the binding is valid.
    std::vector<GpuLight> gpuLights;
    gpuLights.reserve(std::max<size_t>(m_lights.size(), 1));
    for (const auto& light : m_lights) {
        gpuLights.push_back(light->toGpu());
    }

    // Auto-synthesize rect lights from emissive mesh instances.
    // In path tracing (Hyperion), emissive surfaces are sampled via NEE.
    // In real-time rendering (Theia), the forward shader needs explicit GpuLight entries.
    // When no lights were added explicitly, derive one RectLight per emissive mesh.
    if (m_lights.empty()) {
        for (size_t i = 0; i < m_geometries.size(); ++i) {
            const GpuInstance& inst = m_instances[i];
            if (inst.materialIndex >= m_materials.size() || !m_materials[inst.materialIndex].emissiveAsLightSource())
                continue;
            const GpuMaterial& gpuMat = gpuMaterials[inst.materialIndex];
            const float luminance = gpuMat.emissionColorLum.w;
            if (luminance <= 0.0f)
                continue;

            constexpr float kPiLight = 3.14159265358979323846F;

            // Emissive sphere primitive → omnidirectional point light at its centre.
            // The projected disk area (π r²) carries the radiance so the metal/diffuse
            // response matches a path-traced uniform spherical emitter.
            if (inst.geometryKind == 1U) {
                const auto* sph = dynamic_cast<const Sphere*>(m_geometries[i].get());
                if (!sph)
                    continue;
                const sm::float3 sc = static_cast<sm::float3>(m_geometries[i]->xform.matrix() * sm::float4(sph->center(), 1.0F));
                const float sr = std::max(sph->radius(), 0.01F);
                GpuLight gl{};
                gl.position = sc;
                gl.type = std::bit_cast<float>(static_cast<uint32_t>(LightType::Point));
                gl.direction = sm::float3(0.0F, -1.0F, 0.0F);
                gl.range = 0.0F;
                gl.color = static_cast<sm::float3>(gpuMat.emissionColorLum);
                gl.intensity = luminance * kPiLight * sr * sr;
                gl.halfWidth = sr;
                gl.halfHeight = sr;
                gpuLights.push_back(gl);
                Logger::info("Scene: emissive sphere {} → PointLight at ({:.1f},{:.1f},{:.1f}) lum={:.0f} r={:.2f}",
                             i,
                             sc.x,
                             sc.y,
                             sc.z,
                             luminance,
                             sr);
                continue;
            }
            if (inst.geometryKind != 0U)
                continue;

            const auto* mesh = dynamic_cast<const TriangleMesh*>(m_geometries[i].get());
            if (!mesh)
                continue;
            const auto& verts = mesh->data().vertices;
            const auto& idxBuf = mesh->data().indices;
            if (verts.empty() || idxBuf.empty())
                continue;

            const sm::float4x4 xform = m_geometries[i]->xform.matrix();

            // Compute world-space vertices and centroid.
            std::vector<sm::float3> wverts;
            wverts.reserve(verts.size());
            sm::float3 centroid{0.0f};
            for (const auto& v : verts) {
                wverts.push_back(static_cast<sm::float3>(xform * sm::float4(v.position, 1.0F)));
                centroid += wverts.back();
            }
            centroid /= static_cast<float>(wverts.size());

            // Compute area-weighted average normal.  normalSum accumulates 2·area·n̂
            // per triangle; its magnitude relative to total area measures planarity.
            sm::float3 normalSum{0.0f};
            float totalArea = 0.0f;
            const uint32_t triCount = static_cast<uint32_t>(idxBuf.size() / 3);
            for (uint32_t t = 0; t < triCount; ++t) {
                const sm::float3 e1 = wverts[idxBuf[t * 3 + 1]] - wverts[idxBuf[t * 3 + 0]];
                const sm::float3 e2 = wverts[idxBuf[t * 3 + 2]] - wverts[idxBuf[t * 3 + 0]];
                const sm::float3 c = sm::cross(e1, e2);
                const float area = 0.5F * sm::length(c);
                if (area > 1e-6F) {
                    normalSum += c;
                    totalArea += area;
                }
            }
            if (totalArea <= 0.0f)
                continue;

            // Planarity ∈ [0,1]: 1 for a flat quad (all face normals aligned), ~0 for a
            // closed/curved emitter (sphere) whose face normals cancel.  A degenerate
            // average normal cannot form a meaningful rect frame, so those emitters are
            // synthesized as omnidirectional point lights instead.
            const float planarity = sm::length(normalSum) / (2.0F * totalArea);

            constexpr float kPi = 3.14159265358979323846F;
            if (planarity < 0.5F) {
                // Bounding radius from the centroid approximates a uniform spherical
                // emitter.  Its projected disk area (π r²) carries the radiance into the
                // point-light intensity so Li = color · (L·π r²) / d² matches the rect form.
                float radius = 0.0F;
                for (const auto& wv : wverts) {
                    radius = std::max(radius, sm::length(wv - centroid));
                }
                radius = std::max(radius, 0.01F);

                GpuLight gl{};
                gl.position = centroid;
                gl.type = std::bit_cast<float>(static_cast<uint32_t>(LightType::Point));
                gl.direction = sm::float3(0.0F, -1.0F, 0.0F); // unused for point lights
                gl.range = 0.0F;
                gl.color = static_cast<sm::float3>(gpuMat.emissionColorLum);
                gl.intensity = luminance * kPi * radius * radius;
                gl.halfWidth = radius;
                gl.halfHeight = radius;
                gpuLights.push_back(gl);
                Logger::info("Scene: emissive mesh {} → PointLight at ({:.1f},{:.1f},{:.1f}) lum={:.0f} r={:.2f}",
                             i,
                             centroid.x,
                             centroid.y,
                             centroid.z,
                             luminance,
                             radius);
                continue;
            }

            const sm::float3 avgNormal = sm::normalize(normalSum);

            // Build tangent frame and project vertices to find half-extents.
            const sm::float3 up = (std::abs(avgNormal.y) < 0.99F) ? sm::float3{0.0f, 1.0f, 0.0f} : sm::float3{1.0f, 0.0f, 0.0f};
            const sm::float3 tangX = sm::normalize(sm::cross(up, avgNormal));
            const sm::float3 tangY = sm::cross(avgNormal, tangX);
            float minX = std::numeric_limits<float>::max(), maxX = std::numeric_limits<float>::lowest();
            float minY = std::numeric_limits<float>::max(), maxY = std::numeric_limits<float>::lowest();
            for (const auto& wv : wverts) {
                const sm::float3 rel = wv - centroid;
                minX = std::min(minX, sm::dot(rel, tangX));
                maxX = std::max(maxX, sm::dot(rel, tangX));
                minY = std::min(minY, sm::dot(rel, tangY));
                maxY = std::max(maxY, sm::dot(rel, tangY));
            }
            const float halfW = std::max((maxX - minX) * 0.5F, 0.01F);
            const float halfH = std::max((maxY - minY) * 0.5F, 0.01F);

            GpuLight gl{};
            gl.position = centroid;
            gl.type = std::bit_cast<float>(static_cast<uint32_t>(LightType::Rect));
            gl.direction = avgNormal; // emission direction (toward scene)
            gl.range = 0.0F;
            gl.color = static_cast<sm::float3>(gpuMat.emissionColorLum);
            // Raw emissive radiance — matches Hyperion's emissive-mesh NEE, which
            // illuminates with the unmodified Le = emissionColor * luminance
            // (closesthit.slang:233,258, no /683). The /683 luminous-efficacy divide is
            // reserved for explicit Light primitives (Light::toGpu), not meshes synthesized
            // into area lights here. Using raw Le keeps direct lighting equal to Hyperion's.
            gl.intensity = luminance;
            gl.halfWidth = halfW;
            gl.halfHeight = halfH;
            gpuLights.push_back(gl);
            Logger::info("Scene: emissive mesh {} → RectLight at ({:.1f},{:.1f},{:.1f}) lum={:.0f} hw={:.1f} hh={:.1f}",
                         i,
                         centroid.x,
                         centroid.y,
                         centroid.z,
                         luminance,
                         halfW,
                         halfH);
        }
    }

    // Record the real light count (auto-synthesized rect lights included) before
    // padding the buffer with a dummy entry when there are no lights at all.
    m_lightCount = static_cast<uint32_t>(gpuLights.size());

    if (gpuLights.empty()) {
        gpuLights.push_back(GpuLight{});
    }
    auto lightBuf = Buffer::upload(ctx,
                                   pool,
                                   std::as_bytes(std::span<const GpuLight>(gpuLights.data(), gpuLights.size())),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   "scene.lights");
    if (!lightBuf) {
        return lightBuf.error();
    }
    m_lightBuffer = std::move(*lightBuf);

    // Build emissive triangle buffer via the shared harmonia utility.
    std::vector<harmonia::EmissiveInstanceInfo> emissiveInstances;
    emissiveInstances.reserve(m_instances.size());
    for (const GpuInstance& inst : m_instances) {
        emissiveInstances.push_back({inst.geometryKind, inst.materialIndex});
    }
    harmonia::EmissiveData emissiveData =
        harmonia::buildEmissiveData(m_geometries, emissiveInstances, m_materials, gpuMaterials);

    m_emissiveTriangleCount = static_cast<uint32_t>(emissiveData.triangles.size());
    if (emissiveData.triangles.empty()) {
        emissiveData.triangles.push_back(GpuEmissiveTriangle{}); // sentinel — keeps the binding valid
    }
    auto emissiveBuf = Buffer::upload(
        ctx,
        pool,
        std::as_bytes(std::span<const GpuEmissiveTriangle>(emissiveData.triangles.data(), emissiveData.triangles.size())),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        "scene.emissiveTriangles");
    if (!emissiveBuf) {
        return emissiveBuf.error();
    }
    m_emissiveTriangleBuffer = std::move(*emissiveBuf);

    // Power-proportional selection CDF: cdf[i] = (Σ_{j≤i} power_j) / totalPower, so cdf[N-1] == 1.
    // Falls back to a uniform CDF when all emitters have zero power (degenerate/black emitters).
    const std::vector<float>& emissivePower = emissiveData.power;
    std::vector<float> emissiveCdf;
    emissiveCdf.reserve(emissivePower.size());
    double totalPower = 0.0;
    for (const float p : emissivePower) {
        totalPower += static_cast<double>(p);
    }
    if (totalPower > 0.0) {
        double running = 0.0;
        for (const float p : emissivePower) {
            running += static_cast<double>(p);
            emissiveCdf.push_back(static_cast<float>(running / totalPower));
        }
        if (!emissiveCdf.empty()) {
            emissiveCdf.back() = 1.0F; // guard against rounding leaving cdf[N-1] < 1
        }
    } else {
        const auto count = static_cast<uint32_t>(emissivePower.size());
        for (uint32_t i = 0; i < count; ++i) {
            emissiveCdf.push_back(static_cast<float>(i + 1) / static_cast<float>(count));
        }
    }
    if (emissiveCdf.empty()) {
        emissiveCdf.push_back(1.0F); // sentinel — keeps the binding valid when no emitters
    }
    auto emissiveCdfBuf = Buffer::upload(
        ctx,
        pool,
        std::as_bytes(std::span<const float>(emissiveCdf.data(), emissiveCdf.size())),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        "scene.emissiveCdf");
    if (!emissiveCdfBuf) {
        return emissiveCdfBuf.error();
    }
    m_emissiveCdfBuffer = std::move(*emissiveCdfBuf);

    return VK_SUCCESS;
}

VkResult Scene::buildTlas(const DeviceContext& ctx, const CommandPool& pool) {
    std::vector<VkAccelerationStructureInstanceKHR> instances(m_geometries.size());
    for (size_t i = 0; i < m_geometries.size(); ++i) {
        instances[i] = m_geometries[i]->makeInstance(static_cast<uint32_t>(i));
        // Instance masks:
        //   0x01 = emissive geometry
        //   0x02 = transparent non-emissive geometry
        //   0x04 = opaque non-emissive geometry
        // Shadow rays use non-emissive mask (0x06). 26c transmission rays can target
        // opaque-only geometry (0x04) before stacked transparent traversal (26e).
        const uint32_t matIdx = m_instances[i].materialIndex;
        const bool isEmissive = matIdx < m_materials.size() && m_materials[matIdx].emissiveAsLightSource() &&
                                m_materials[matIdx].gpu().emissionColorLum.w > 0.0F;
        bool isTransparent = false;
        if (!isEmissive && matIdx < m_materials.size()) {
            const auto gpuMat = m_materials[matIdx].gpu();
            const float opacity = std::clamp(gpuMat.opacityFlagsPad.x, 0.0f, 1.0f);
            const float transmissionWeight = std::max(gpuMat.transmissionColorWeight.w, 0.0f);
            isTransparent = (opacity < 0.9999f) || (transmissionWeight > 0.0f);
        }
        instances[i].mask = isEmissive ? 0x01u : (isTransparent ? 0x02u : 0x04u);
    }

    auto instanceUpload = Buffer::upload(
        ctx,
        pool,
        std::as_bytes(std::span<const VkAccelerationStructureInstanceKHR>(instances.data(), instances.size())),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        "scene.tlas.instances");
    if (!instanceUpload) {
        return instanceUpload.error();
    }

    const VkAccelerationStructureGeometryInstancesDataKHR instancesData{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .pNext = nullptr,
        .arrayOfPointers = VK_FALSE,
        .data = VkDeviceOrHostAddressConstKHR{instanceUpload->deviceAddress()},
    };
    const VkAccelerationStructureGeometryKHR geometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .pNext = nullptr,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry = VkAccelerationStructureGeometryDataKHR{.instances = instancesData},
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    const uint32_t primitiveCount = static_cast<uint32_t>(instances.size());
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .pNext = nullptr,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = VK_NULL_HANDLE,
        .dstAccelerationStructure = VK_NULL_HANDLE,
        .geometryCount = 1,
        .pGeometries = &geometry,
        .ppGeometries = nullptr,
        .scratchData = VkDeviceOrHostAddressKHR{},
    };
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

    auto tlasAS = AccelerationStructure::create(
        ctx, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizeInfo.accelerationStructureSize, "scene.tlas");
    if (!tlasAS) {
        return tlasAS.error();
    }

    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
    asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &asProps;
    vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &props);

    auto scratch = Buffer::create(
        ctx,
        std::max<VkDeviceSize>(sizeInfo.buildScratchSize + asProps.minAccelerationStructureScratchOffsetAlignment, 16),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        "scene.tlasScratch");
    if (!scratch) {
        return scratch.error();
    }

    buildInfo.dstAccelerationStructure = tlasAS->handle();
    buildInfo.scratchData.deviceAddress =
        bufferAlignUp(scratch->deviceAddress(), asProps.minAccelerationStructureScratchOffsetAlignment);
    const VkAccelerationStructureBuildRangeInfoKHR rangeInfo{
        .primitiveCount = primitiveCount,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };
    const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = &rangeInfo;

    auto cmd = pool.beginOneShot();
    if (!cmd) {
        return cmd.error();
    }
    vkCmdBuildAccelerationStructuresKHR(*cmd, 1, &buildInfo, &rangePtr);
    if (const VkResult result = pool.endOneShot(*cmd); result != VK_SUCCESS) {
        return result;
    }

    m_tlas = std::move(*tlasAS);
    m_tlasAddress = m_tlas.deviceAddress();
    return VK_SUCCESS;
}
