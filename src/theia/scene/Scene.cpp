#include "theia/scene/Scene.hpp"

#include <volk/volk.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <meshoptimizer.h>
#include <slang-math/slang-math.hpp>
#include <span>
#include <utility>
#include <vector>
#include <vma/vk_mem_alloc.h>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/renderer/TlasBuilder.hpp"
#include "harmonia/scene/EmissiveBuilder.hpp"
#include "harmonia/scene/Geometry.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"

namespace {

/// World-space instance bounding sphere from a mesh's object-space bounds + the
/// instance transform (centre transformed by the matrix, radius scaled by the
/// largest axis scale — conservative under non-uniform scale).
GpuInstanceBounds worldBounds(const Scene::MeshGpu& mesh, const Xform& xform) {
    const sm::float4x4 m = xform.matrix();
    const sm::float3 c =
        static_cast<sm::float3>(m * sm::float4(mesh.boundsCenterX, mesh.boundsCenterY, mesh.boundsCenterZ, 1.0f));
    const float maxScale = std::max({std::abs(xform.scale.x), std::abs(xform.scale.y), std::abs(xform.scale.z)});
    return GpuInstanceBounds{c.x, c.y, c.z, mesh.boundsRadius * maxScale};
}

} // namespace

uint32_t Scene::addSphereMesh(const DeviceContext& ctx, const CommandPool& pool, float radius, std::string_view name) {
    // Theia is a rasterizer (mesh-shader pipeline), so an analytic sphere cannot
    // be drawn directly — it is tessellated into a triangle mesh (object space,
    // centred at the origin) and placed by the instance transform.
    MeshData mesh = ProceduralGeometry::makeSphere(sm::float3{0.0f, 0.0f, 0.0f}, radius);
    return addMesh(ctx, pool, std::move(mesh), name.empty() ? "sphere" : name);
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

    // Lay out the global vertex/index/meshlet buffers from the unique meshes (object
    // space), recording each mesh's ranges + object-space bounding sphere.
    std::vector<GpuVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<GpuMeshlet> gpuMeshlets;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint32_t> meshletTriangles;
    m_meshGpu.assign(m_meshes.size(), MeshGpu{});

    constexpr size_t kMaxVerts = 64;
    constexpr size_t kMaxTris = 124;

    for (size_t mi = 0; mi < m_meshes.size(); ++mi) {
        const auto* mesh = dynamic_cast<const TriangleMesh*>(m_meshes[mi].get());
        if (mesh == nullptr) {
            continue; // Theia tessellates everything into triangle meshes
        }
        MeshGpu& gpu = m_meshGpu[mi];
        gpu.vertexOffset = static_cast<uint32_t>(vertices.size());
        gpu.indexOffset = static_cast<uint32_t>(indices.size());
        gpu.geometryKind = 0;

        const auto& meshVerts = mesh->data().vertices;
        const auto& localIndices = mesh->data().indices;
        const size_t vertCount = meshVerts.size();
        const size_t indexCount = localIndices.size();
        gpu.indexCount = static_cast<uint32_t>(indexCount);

        vertices.insert(vertices.end(), meshVerts.begin(), meshVerts.end());
        for (uint32_t index : localIndices) {
            indices.push_back(index + gpu.vertexOffset);
        }

        // Meshlets (object space — meshopt bounds are object-space too).
        const size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, kMaxVerts, kMaxTris);
        std::vector<meshopt_Meshlet> rawMeshlets(maxMeshlets);
        std::vector<uint32_t> rawVerts(maxMeshlets * kMaxVerts);
        std::vector<uint8_t> rawTris(maxMeshlets * kMaxTris * 3);

        std::vector<float> positions;
        positions.reserve(vertCount * 3);
        for (const auto& v : meshVerts) {
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

        gpu.meshletOffset = static_cast<uint32_t>(gpuMeshlets.size());
        gpu.meshletCount = static_cast<uint32_t>(meshletCount);

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
                meshletVertices.push_back(gpu.vertexOffset + rawVerts[ml.vertex_offset + v]);
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

        // Object-space bounding sphere (centroid method).
        const float invN = 1.0f / std::max<float>(static_cast<float>(vertCount), 1.0f);
        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        for (const auto& v : meshVerts) {
            cx += v.position.x;
            cy += v.position.y;
            cz += v.position.z;
        }
        cx *= invN;
        cy *= invN;
        cz *= invN;
        float bs = 0.0f;
        for (const auto& v : meshVerts) {
            const float dx = v.position.x - cx;
            const float dy = v.position.y - cy;
            const float dz = v.position.z - cz;
            bs = std::max(bs, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        gpu.boundsCenterX = cx;
        gpu.boundsCenterY = cy;
        gpu.boundsCenterZ = cz;
        gpu.boundsRadius = bs;
    }

    if (vertices.empty()) {
        vertices.push_back(GpuVertex{});
    }
    if (indices.empty()) {
        indices.push_back(0);
    }
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

    // Per-instance GPU rows + world-space bounds from the instance list.
    m_gpuInstances.clear();
    m_gpuInstances.reserve(m_instances.size());
    m_instanceBounds.clear();
    m_instanceBounds.reserve(m_instances.size());
    for (const InstanceRecord& inst : m_instances) {
        const MeshGpu& gpu = m_meshGpu[inst.meshIndex];
        m_gpuInstances.push_back(GpuInstance{
            .meshIndex = inst.meshIndex,
            .materialIndex = inst.materialIndex,
            .vertexOffset = gpu.vertexOffset,
            .indexOffset = gpu.indexOffset,
            .indexCount = gpu.indexCount,
            .meshletOffset = gpu.meshletOffset,
            .meshletCount = gpu.meshletCount,
            .geometryKind = gpu.geometryKind,
            .sphereRadius = gpu.sphereRadius,
        });
        m_instanceBounds.push_back(worldBounds(gpu, inst.xform));
    }
    m_instanceBounds.resize(m_instances.size(), GpuInstanceBounds{});

    auto instanceBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const GpuInstance>(m_gpuInstances.data(), m_gpuInstances.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.instances");
    if (!instanceBuf) {
        return instanceBuf.error();
    }
    auto instanceBoundsBuf = Buffer::upload(
        ctx,
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
    Logger::info(
        "Scene built: {} meshes, {} instances, {} meshlets", m_meshes.size(), m_instances.size(), gpuMeshlets.size());

    // Per-instance object→world transforms (current + previous). Static scenes upload
    // once at build; the previous buffer starts equal to the current.
    const size_t instCount = std::max<size_t>(m_instances.size(), 1);
    std::vector<sm::float4x4> instanceTransforms(instCount, sm::float4x4(1.0f));
    for (size_t i = 0; i < m_instances.size(); ++i) {
        instanceTransforms[i] = m_instances[i].xform.matrix();
    }
    auto transformBuf = Buffer::upload(
        ctx,
        pool,
        std::as_bytes(std::span<const sm::float4x4>(instanceTransforms.data(), instanceTransforms.size())),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        "scene.instanceTransforms");
    if (!transformBuf) {
        return transformBuf.error();
    }
    auto prevTransformBuf = Buffer::upload(
        ctx,
        pool,
        std::as_bytes(std::span<const sm::float4x4>(instanceTransforms.data(), instanceTransforms.size())),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        "scene.prevInstanceTransforms");
    if (!prevTransformBuf) {
        return prevTransformBuf.error();
    }
    std::vector<uint32_t> objectIds(instCount, 0u);
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_instances.size()); ++i) {
        objectIds[i] = i;
    }
    auto objectIdBuf = Buffer::upload(ctx,
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

    // Auto-synthesize lights from emissive mesh instances when none were added explicitly.
    if (m_lights.empty()) {
        for (size_t i = 0; i < m_instances.size(); ++i) {
            const InstanceRecord& inst = m_instances[i];
            if (inst.materialIndex >= m_materials.size() || !m_materials[inst.materialIndex].emissiveAsLightSource())
                continue;
            const GpuMaterial& gpuMat = gpuMaterials[inst.materialIndex];
            const float luminance = gpuMat.emissionColorLum.w;
            if (luminance <= 0.0f)
                continue;

            const auto* mesh = dynamic_cast<const TriangleMesh*>(m_meshes[inst.meshIndex].get());
            if (!mesh)
                continue;
            const auto& verts = mesh->data().vertices;
            const auto& idxBuf = mesh->data().indices;
            if (verts.empty() || idxBuf.empty())
                continue;

            constexpr float kPi = 3.14159265358979323846F;
            const sm::float4x4 xform = inst.xform.matrix();

            std::vector<sm::float3> wverts;
            wverts.reserve(verts.size());
            sm::float3 centroid{0.0f};
            for (const auto& v : verts) {
                wverts.push_back(static_cast<sm::float3>(xform * sm::float4(v.position, 1.0f)));
                centroid += wverts.back();
            }
            centroid /= static_cast<float>(wverts.size());

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

            const float planarity = sm::length(normalSum) / (2.0F * totalArea);

            if (planarity < 0.5F) {
                float radius = 0.0F;
                for (const auto& wv : wverts) {
                    radius = std::max(radius, sm::length(wv - centroid));
                }
                radius = std::max(radius, 0.01F);

                GpuLight gl{};
                gl.position = centroid;
                gl.type = std::bit_cast<float>(static_cast<uint32_t>(LightType::Point));
                gl.direction = sm::float3(0.0F, -1.0F, 0.0F);
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
            const sm::float3 up =
                (std::abs(avgNormal.y) < 0.99F) ? sm::float3{0.0f, 1.0f, 0.0f} : sm::float3{1.0f, 0.0f, 0.0f};
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
            gl.direction = avgNormal;
            gl.range = 0.0F;
            gl.color = static_cast<sm::float3>(gpuMat.emissionColorLum);
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

    // Emissive triangles via the shared harmonia utility (iterates instances).
    harmonia::EmissiveData emissiveData = harmonia::buildEmissiveData(m_meshes, m_instances, m_materials, gpuMaterials);

    m_emissiveTriangleCount = static_cast<uint32_t>(emissiveData.triangles.size());
    if (emissiveData.triangles.empty()) {
        emissiveData.triangles.push_back(GpuEmissiveTriangle{});
    }
    auto emissiveBuf = Buffer::upload(ctx,
                                      pool,
                                      std::as_bytes(std::span<const GpuEmissiveTriangle>(
                                          emissiveData.triangles.data(), emissiveData.triangles.size())),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      "scene.emissiveTriangles");
    if (!emissiveBuf) {
        return emissiveBuf.error();
    }
    m_emissiveTriangleBuffer = std::move(*emissiveBuf);

    // Power-proportional selection CDF for emissive-triangle NEE (shared helper —
    // identical across renderers, so the NEE sampling CDF cannot drift).
    const std::vector<float> emissiveCdf = harmonia::buildEmissiveCdf(emissiveData.power);
    auto emissiveCdfBuf = Buffer::upload(ctx,
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
    std::vector<VkAccelerationStructureInstanceKHR> instances(m_instances.size());
    for (size_t i = 0; i < m_instances.size(); ++i) {
        const InstanceRecord& inst = m_instances[i];
        instances[i] = m_meshes[inst.meshIndex]->makeInstance(static_cast<uint32_t>(i), inst.xform);
        // Instance masks (Theia-specific visibility channels for the raster/RT-GI passes):
        //   0x01 = emissive geometry
        //   0x02 = transparent non-emissive geometry
        //   0x04 = opaque non-emissive geometry
        const uint32_t matIdx = inst.materialIndex;
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
    // The TLAS-build mechanics are shared (harmonia::buildTlas); only the masks above diverge.
    return harmonia::buildTlas(ctx, pool, instances, m_tlas, m_tlasAddress);
}
