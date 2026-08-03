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

namespace {} // namespace

std::uint32_t Scene::addSphereMesh(const harmonia::DeviceContext& ctx,
                                   const harmonia::CommandPool& pool,
                                   float radius,
                                   std::string_view name) {
    // Theia is a rasterizer (mesh-shader pipeline), so an analytic sphere cannot
    // be drawn directly — it is tessellated into a triangle mesh (object space,
    // centred at the origin) and placed by the instance transform.
    harmonia::MeshData mesh = harmonia::ProceduralGeometry::makeSphere(sm::float3{0.0f, 0.0f, 0.0f}, radius);
    return addMesh(ctx, pool, std::move(mesh), name.empty() ? "sphere" : name);
}

VkResult Scene::buildSceneBuffers(const harmonia::DeviceContext& ctx, const harmonia::CommandPool& pool) {
    std::vector<harmonia::GpuMaterial> gpuMaterials;
    gpuMaterials.reserve(std::max<std::size_t>(m_materials.size(), 1));
    for (const harmonia::Material& material : m_materials) {
        gpuMaterials.push_back(material.gpu());
    }
    if (gpuMaterials.empty()) {
        gpuMaterials.push_back(harmonia::GpuMaterial{});
    }

    std::vector<harmonia::GpuVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<GpuMeshlet> gpuMeshlets;
    std::vector<std::uint32_t> meshletVertices;
    std::vector<std::uint32_t> meshletTriangles;
    buildMeshlets(vertices, indices, gpuMeshlets, meshletVertices, meshletTriangles);

    buildGpuInstances();

    if (VkResult r = uploadSceneBuffers(
            ctx, pool, gpuMaterials, vertices, indices, gpuMeshlets, meshletVertices, meshletTriangles)) {
        return r;
    }

    std::vector<harmonia::GpuLight> gpuLights;
    gpuLights.reserve(std::max<std::size_t>(m_lights.size(), 1));
    for (const auto& light : m_lights) {
        gpuLights.push_back(light->toGpu());
    }

    synthesizeEmissiveLights(gpuMaterials, gpuLights);

    m_lightCount = static_cast<std::uint32_t>(gpuLights.size());
    if (gpuLights.empty()) {
        gpuLights.push_back(harmonia::GpuLight{});
    }
    if (VkResult r =
            uploadBuffer(ctx,
                         pool,
                         std::as_bytes(std::span<const harmonia::GpuLight>(gpuLights.data(), gpuLights.size())),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         "scene.lights",
                         m_lightBuffer)) {
        return r;
    }

    harmonia::EmissiveData emissiveData = harmonia::buildEmissiveData(m_meshes, m_instances, m_materials, gpuMaterials);

    m_emissiveTriangleCount = static_cast<std::uint32_t>(emissiveData.triangles.size());
    if (emissiveData.triangles.empty()) {
        emissiveData.triangles.push_back(harmonia::GpuEmissiveTriangle{});
    }
    if (VkResult r = uploadBuffer(ctx,
                                  pool,
                                  std::as_bytes(std::span<const harmonia::GpuEmissiveTriangle>(
                                      emissiveData.triangles.data(), emissiveData.triangles.size())),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  "scene.emissiveTriangles",
                                  m_emissiveTriangleBuffer)) {
        return r;
    }

    const std::vector<float> emissiveCdf = harmonia::buildEmissiveCdf(emissiveData.power);
    if (VkResult r = uploadBuffer(ctx,
                                  pool,
                                  std::as_bytes(std::span<const float>(emissiveCdf.data(), emissiveCdf.size())),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  "scene.emissiveCdf",
                                  m_emissiveCdfBuffer)) {
        return r;
    }

    return VK_SUCCESS;
}

void Scene::buildMeshlets(std::vector<harmonia::GpuVertex>& vertices,
                          std::vector<std::uint32_t>& indices,
                          std::vector<GpuMeshlet>& gpuMeshlets,
                          std::vector<std::uint32_t>& meshletVertices,
                          std::vector<std::uint32_t>& meshletTriangles) {
    m_meshGpu.assign(m_meshes.size(), MeshGpu{});

    constexpr std::size_t kMaxVerts = 64;
    constexpr std::size_t kMaxTris = 124;

    for (std::size_t mi = 0; mi < m_meshes.size(); ++mi) {
        const auto* mesh = dynamic_cast<const harmonia::TriangleMesh*>(m_meshes[mi].get());
        if (mesh == nullptr) {
            continue;
        }
        MeshGpu& gpu = m_meshGpu[mi];
        gpu.vertexOffset = static_cast<std::uint32_t>(vertices.size());
        gpu.indexOffset = static_cast<std::uint32_t>(indices.size());
        gpu.geometryKind = 0;

        const auto& meshVerts = mesh->data().vertices;
        const auto& localIndices = mesh->data().indices;
        const std::size_t vertCount = meshVerts.size();
        const std::size_t indexCount = localIndices.size();
        gpu.indexCount = static_cast<std::uint32_t>(indexCount);

        vertices.insert(vertices.end(), meshVerts.begin(), meshVerts.end());
        for (std::uint32_t index : localIndices) {
            indices.push_back(index + gpu.vertexOffset);
        }

        const std::size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, kMaxVerts, kMaxTris);
        std::vector<meshopt_Meshlet> rawMeshlets(maxMeshlets);
        std::vector<std::uint32_t> rawVerts(maxMeshlets * kMaxVerts);
        std::vector<std::uint8_t> rawTris(maxMeshlets * kMaxTris * 3);

        std::vector<float> positions;
        positions.reserve(vertCount * 3);
        for (const auto& v : meshVerts) {
            positions.push_back(v.position.x);
            positions.push_back(v.position.y);
            positions.push_back(v.position.z);
        }

        const std::size_t meshletCount = meshopt_buildMeshlets(rawMeshlets.data(),
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

        gpu.meshletOffset = static_cast<std::uint32_t>(gpuMeshlets.size());
        gpu.meshletCount = static_cast<std::uint32_t>(meshletCount);

        for (std::size_t m = 0; m < meshletCount; ++m) {
            const meshopt_Meshlet& ml = rawMeshlets[m];
            const meshopt_Bounds bounds = meshopt_computeMeshletBounds(rawVerts.data() + ml.vertex_offset,
                                                                       rawTris.data() + ml.triangle_offset,
                                                                       ml.triangle_count,
                                                                       positions.data(),
                                                                       vertCount,
                                                                       sizeof(float) * 3);

            const std::uint32_t gpuVertBase = static_cast<std::uint32_t>(meshletVertices.size());
            for (std::uint32_t v = 0; v < ml.vertex_count; ++v) {
                meshletVertices.push_back(gpu.vertexOffset + rawVerts[ml.vertex_offset + v]);
            }

            const std::uint32_t gpuTriBase = static_cast<std::uint32_t>(meshletTriangles.size());
            const std::uint8_t* triBytes = rawTris.data() + ml.triangle_offset;
            const std::uint32_t triByteCount = ml.triangle_count * 3;
            const std::uint32_t triPadded = (triByteCount + 3) & ~3u;
            for (std::uint32_t b = 0; b < triPadded; b += 4) {
                std::uint32_t packed = 0;
                for (std::uint32_t k = 0; k < 4; ++k) {
                    if (b + k < triByteCount) {
                        packed |= static_cast<std::uint32_t>(triBytes[b + k]) << (k * 8);
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
    }

    if (vertices.empty()) {
        vertices.push_back(harmonia::GpuVertex{});
    }
    if (indices.empty()) {
        indices.push_back(0);
    }
    m_meshletCount = static_cast<std::uint32_t>(gpuMeshlets.size());
    if (gpuMeshlets.empty()) {
        gpuMeshlets.push_back(GpuMeshlet{});
    }
    if (meshletVertices.empty()) {
        meshletVertices.push_back(0);
    }
    if (meshletTriangles.empty()) {
        meshletTriangles.push_back(0);
    }
}

void Scene::buildGpuInstances() {
    m_gpuInstances.clear();
    m_gpuInstances.reserve(m_instances.size());
    m_instanceBounds.clear();
    m_instanceBounds.reserve(m_instances.size());
    for (const harmonia::InstanceRecord& inst : m_instances) {
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
        m_instanceBounds.push_back([&] {
            const harmonia::Aabb object = m_meshes[inst.meshIndex]->objectAabb();
            const harmonia::Aabb world = harmonia::worldAabbFromInstance(object, inst.xform);
            return GpuInstanceBounds{
                .minX = world.min.x,
                .minY = world.min.y,
                .minZ = world.min.z,
                .maxX = world.max.x,
                .maxY = world.max.y,
                .maxZ = world.max.z,
            };
        }());
    }
    m_instanceBounds.resize(m_instances.size(), GpuInstanceBounds{});
}

VkResult Scene::uploadBuffer(const harmonia::DeviceContext& ctx,
                             const harmonia::CommandPool& pool,
                             std::span<const std::byte> data,
                             VkBufferUsageFlags usage,
                             const char* name,
                             harmonia::Buffer& out) {
    auto buf = uploadStorageBuffer(ctx, pool, data, name, usage);
    if (!buf) {
        return buf.error();
    }
    out = std::move(*buf);
    return VK_SUCCESS;
}

VkResult Scene::uploadSceneBuffers(const harmonia::DeviceContext& ctx,
                                   const harmonia::CommandPool& pool,
                                   const std::vector<harmonia::GpuMaterial>& gpuMaterials,
                                   const std::vector<harmonia::GpuVertex>& vertices,
                                   const std::vector<std::uint32_t>& indices,
                                   const std::vector<GpuMeshlet>& gpuMeshlets,
                                   const std::vector<std::uint32_t>& meshletVertices,
                                   const std::vector<std::uint32_t>& meshletTriangles) {
    constexpr VkBufferUsageFlags kStorageAddr =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    if (VkResult r =
            uploadBuffer(ctx,
                         pool,
                         std::as_bytes(std::span<const GpuInstance>(m_gpuInstances.data(), m_gpuInstances.size())),
                         kStorageAddr,
                         "scene.instances",
                         m_instanceBuffer)) {
        return r;
    }
    if (VkResult r = uploadBuffer(
            ctx,
            pool,
            std::as_bytes(std::span<const GpuInstanceBounds>(m_instanceBounds.data(), m_instanceBounds.size())),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            "scene.instanceBounds",
            m_instanceBoundsBuffer)) {
        return r;
    }
    if (VkResult r = uploadBuffer(
            ctx,
            pool,
            std::as_bytes(std::span<const harmonia::GpuMaterial>(gpuMaterials.data(), gpuMaterials.size())),
            kStorageAddr,
            "scene.materials",
            m_materialBuffer)) {
        return r;
    }
    if (VkResult r = uploadBuffer(ctx,
                                  pool,
                                  std::as_bytes(std::span<const harmonia::GpuVertex>(vertices.data(), vertices.size())),
                                  kStorageAddr,
                                  "scene.vertices",
                                  m_vertexBuffer)) {
        return r;
    }
    if (VkResult r = uploadBuffer(ctx,
                                  pool,
                                  std::as_bytes(std::span<const std::uint32_t>(indices.data(), indices.size())),
                                  kStorageAddr,
                                  "scene.indices",
                                  m_indexBuffer)) {
        return r;
    }
    if (VkResult r = uploadBuffer(ctx,
                                  pool,
                                  std::as_bytes(std::span<const GpuMeshlet>(gpuMeshlets.data(), gpuMeshlets.size())),
                                  kStorageAddr,
                                  "scene.meshlets",
                                  m_meshletBuffer)) {
        return r;
    }
    if (VkResult r =
            uploadBuffer(ctx,
                         pool,
                         std::as_bytes(std::span<const std::uint32_t>(meshletVertices.data(), meshletVertices.size())),
                         kStorageAddr,
                         "scene.meshletVertices",
                         m_meshletVertexBuffer)) {
        return r;
    }
    if (VkResult r = uploadBuffer(
            ctx,
            pool,
            std::as_bytes(std::span<const std::uint32_t>(meshletTriangles.data(), meshletTriangles.size())),
            kStorageAddr,
            "scene.meshletTriangles",
            m_meshletTriangleBuffer)) {
        return r;
    }

    harmonia::Logger::info(
        "Scene built: {} meshes, {} instances, {} meshlets", m_meshes.size(), m_instances.size(), gpuMeshlets.size());

    const std::size_t instCount = std::max<std::size_t>(m_instances.size(), 1);
    std::vector<sm::float4x4> instanceTransforms(instCount, sm::float4x4(1.0f));
    for (std::size_t i = 0; i < m_instances.size(); ++i) {
        instanceTransforms[i] = m_instances[i].xform.matrix();
    }
    if (VkResult r = uploadBuffer(
            ctx,
            pool,
            std::as_bytes(std::span<const sm::float4x4>(instanceTransforms.data(), instanceTransforms.size())),
            kStorageAddr,
            "scene.instanceTransforms",
            m_instanceTransformBuffer)) {
        return r;
    }
    if (VkResult r = uploadBuffer(
            ctx,
            pool,
            std::as_bytes(std::span<const sm::float4x4>(instanceTransforms.data(), instanceTransforms.size())),
            kStorageAddr,
            "scene.prevInstanceTransforms",
            m_prevInstanceTransformBuffer)) {
        return r;
    }
    std::vector<std::uint32_t> objectIds(instCount, 0u);
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(m_instances.size()); ++i) {
        objectIds[i] = i;
    }
    if (VkResult r = uploadBuffer(ctx,
                                  pool,
                                  std::as_bytes(std::span<const std::uint32_t>(objectIds.data(), objectIds.size())),
                                  kStorageAddr,
                                  "scene.objectIds",
                                  m_objectIdBuffer)) {
        return r;
    }
    return VK_SUCCESS;
}

void Scene::synthesizeEmissiveLights(const std::vector<harmonia::GpuMaterial>& gpuMaterials,
                                     std::vector<harmonia::GpuLight>& gpuLights) {
    if (!m_lights.empty()) {
        return;
    }
    for (std::size_t i = 0; i < m_instances.size(); ++i) {
        const harmonia::InstanceRecord& inst = m_instances[i];
        if (inst.materialIndex >= m_materials.size() || !m_materials[inst.materialIndex].emissiveAsLightSource())
            continue;
        const harmonia::GpuMaterial& gpuMat = gpuMaterials[inst.materialIndex];
        const float luminance = gpuMat.emissionColorLum.w;
        if (luminance <= 0.0f)
            continue;

        const auto* mesh = dynamic_cast<const harmonia::TriangleMesh*>(m_meshes[inst.meshIndex].get());
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
        const std::uint32_t triCount = static_cast<std::uint32_t>(idxBuf.size() / 3);
        for (std::uint32_t t = 0; t < triCount; ++t) {
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

            harmonia::GpuLight gl{};
            gl.position = centroid;
            gl.type = std::bit_cast<float>(static_cast<std::uint32_t>(harmonia::LightType::Point));
            gl.direction = sm::float3(0.0F, -1.0F, 0.0F);
            gl.range = 0.0F;
            gl.color = static_cast<sm::float3>(gpuMat.emissionColorLum);
            gl.intensity = luminance * kPi * radius * radius;
            gl.halfWidth = radius;
            gl.halfHeight = radius;
            gpuLights.push_back(gl);
            harmonia::Logger::info(
                "Scene: emissive mesh {} → harmonia::PointLight at ({:.1f},{:.1f},{:.1f}) lum={:.0f} r={:.2f}",
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

        harmonia::GpuLight gl{};
        gl.position = centroid;
        gl.type = std::bit_cast<float>(static_cast<std::uint32_t>(harmonia::LightType::Rect));
        gl.direction = avgNormal;
        gl.range = 0.0F;
        gl.color = static_cast<sm::float3>(gpuMat.emissionColorLum);
        gl.intensity = luminance;
        gl.halfWidth = halfW;
        gl.halfHeight = halfH;
        gpuLights.push_back(gl);
        harmonia::Logger::info(
            "Scene: emissive mesh {} → harmonia::RectLight at ({:.1f},{:.1f},{:.1f}) lum={:.0f} hw={:.1f} hh={:.1f}",
            i,
            centroid.x,
            centroid.y,
            centroid.z,
            luminance,
            halfW,
            halfH);
    }
}

std::uint32_t Scene::instanceMask(std::size_t instanceIndex) const {
    const harmonia::InstanceRecord& inst = m_instances[instanceIndex];
    const std::uint32_t matIdx = inst.materialIndex;
    const bool isEmissive = matIdx < m_materials.size() && m_materials[matIdx].emissiveAsLightSource() &&
                            m_materials[matIdx].gpu().emissionColorLum.w > 0.0F;
    bool isTransparent = false;
    if (!isEmissive && matIdx < m_materials.size()) {
        const auto gpuMat = m_materials[matIdx].gpu();
        const float opacity = std::clamp(gpuMat.opacityFlagsPad.x, 0.0f, 1.0f);
        const float transmissionWeight = std::max(gpuMat.transmissionColorWeight.w, 0.0f);
        isTransparent = (opacity < 0.9999f) || (transmissionWeight > 0.0f);
    }
    return isEmissive ? 0x01u : (isTransparent ? 0x02u : 0x04u);
}
