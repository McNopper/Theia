#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "demo/importers/ObjImporter.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "demo/importers/MaterialLibrary.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/utils/Hash.hpp"

namespace {

struct ObjIndex {
    int position = 0;
    int texcoord = 0;
    int normal = 0;
};

struct PendingMesh {
    std::string name;
    std::vector<ObjIndex> indices;
};

// ── Text helpers ─────────────────────────────────────────────────────────

[[nodiscard]] std::string trim(std::string_view sv) {
    const auto first = sv.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return {};
    return std::string(sv.substr(first, sv.find_last_not_of(" \t\r\n") - first + 1));
}

[[nodiscard]] bool parseVec3(std::istringstream& s, glm::vec3& v) {
    return static_cast<bool>(s >> v.x >> v.y >> v.z);
}

[[nodiscard]] bool parseVec2(std::istringstream& s, glm::vec2& v) {
    return static_cast<bool>(s >> v.x >> v.y);
}

[[nodiscard]] int resolveIndex(int idx, size_t count) noexcept {
    if (idx > 0)
        return idx - 1;
    if (idx < 0)
        return static_cast<int>(count) + idx;
    return -1;
}

[[nodiscard]] ObjIndex parseFaceVertex(std::string_view token) {
    ObjIndex idx{};
    const size_t s0 = token.find('/');
    if (s0 == std::string_view::npos) {
        idx.position = std::stoi(std::string(token));
        return idx;
    }
    const size_t s1 = token.find('/', s0 + 1);
    idx.position = std::stoi(std::string(token.substr(0, s0)));
    if (s1 == std::string_view::npos) {
        if (s0 + 1 < token.size())
            idx.texcoord = std::stoi(std::string(token.substr(s0 + 1)));
        return idx;
    }
    if (s1 > s0 + 1)
        idx.texcoord = std::stoi(std::string(token.substr(s0 + 1, s1 - s0 - 1)));
    if (s1 + 1 < token.size())
        idx.normal = std::stoi(std::string(token.substr(s1 + 1)));
    return idx;
}

// ── Material resolution ───────────────────────────────────────────────────
//
// OBJ files are treated as pure geometry.  The scene file (via SceneLoader)
// is the sole authority for material assignment; the OBJ's own mtllib/usemtl
// directives are intentionally NOT consulted (material import is disabled by
// design).
//
// Priority order (highest first):
//   1. options.groupMaterials[meshName]  — per-group override from the scene library
//   2. options.overrideMaterial          — whole-object override from the scene library
//   3. fallback: diffuse gray

[[nodiscard]] Material resolveMaterial(const std::string& meshName, const ImportOptions& options) {
    if (options.library != nullptr) {
        // Per-group override.
        if (const auto it = options.groupMaterials.find(meshName); it != options.groupMaterials.end()) {
            if (auto m = options.library->get(it->second))
                return *m;
        }
        // Whole-object override.
        if (!options.overrideMaterial.empty()) {
            if (auto m = options.library->get(options.overrideMaterial))
                return *m;
        }
    }
    return Material::diffuse(glm::vec3(0.8f));
}

// ── Mesh flush ────────────────────────────────────────────────────────────

bool flushMesh(PendingMesh& pending,
               Scene& scene,
               const DeviceContext& ctx,
               const CommandPool& pool,
               const std::vector<glm::vec3>& positions,
               const std::vector<glm::vec3>& normals,
               const std::vector<glm::vec2>& texcoords,
               const ImportOptions& options) {
    if (pending.indices.empty())
        return true;

    const glm::mat4& T = options.worldTransform;
    const glm::mat4 Tn = glm::transpose(glm::inverse(T));

    std::unordered_map<GpuVertex, uint32_t, VertexHash, VertexEqual> unique;
    MeshData mesh;
    mesh.vertices.reserve(pending.indices.size());
    mesh.indices.reserve(pending.indices.size());

    for (const ObjIndex& oi : pending.indices) {
        const int p = resolveIndex(oi.position, positions.size());
        const int n = resolveIndex(oi.normal, normals.size());
        const int t = resolveIndex(oi.texcoord, texcoords.size());

        if (p < 0 || p >= static_cast<int>(positions.size())) {
            Logger::error("OBJ '{}': invalid position index", pending.name);
            return false;
        }

        const glm::vec3 pos = glm::vec3(T * glm::vec4(positions[static_cast<size_t>(p)], 1.0f));
        const glm::vec3 nrm = glm::normalize(
            glm::vec3(Tn * glm::vec4((n >= 0 && n < static_cast<int>(normals.size())) ? normals[static_cast<size_t>(n)]
                                                                                      : glm::vec3(0.0f, 1.0f, 0.0f),
                                     0.0f)));
        const glm::vec2 uv =
            (t >= 0 && t < static_cast<int>(texcoords.size())) ? texcoords[static_cast<size_t>(t)] : glm::vec2(0.0f);

        GpuVertex v{.position = pos,
                    .tangentX = 0.0f,
                    .normal = nrm,
                    .tangentY = 0.0f,
                    .uv = uv,
                    .tangentZ = 0.0f,
                    .bitangentSign = 1.0f};

        const auto [it, inserted] = unique.emplace(v, static_cast<uint32_t>(mesh.vertices.size()));
        if (inserted)
            mesh.vertices.push_back(v);
        mesh.indices.push_back(it->second);
    }

    const Material mat = resolveMaterial(pending.name, options);
    const uint32_t mi = scene.addMaterial(mat);
    if (scene.addMesh(ctx, pool, std::move(mesh), mi, pending.name) == std::numeric_limits<uint32_t>::max()) {
        Logger::error("OBJ: failed to upload mesh '{}'", pending.name);
        return false;
    }

    pending.indices.clear();
    return true;
}

} // namespace

// ── ObjImporter::import ───────────────────────────────────────────────────

bool ObjImporter::import(const std::filesystem::path& path,
                         Scene& scene,
                         const DeviceContext& ctx,
                         const CommandPool& pool,
                         const ImportOptions& options) {
    std::ifstream file(path);
    if (!file) {
        Logger::error("ObjImporter: cannot open '{}'", path.string());
        return false;
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    PendingMesh pending{.name = path.stem().string(), .indices = {}};

    std::string line;
    while (std::getline(file, line)) {
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#')
            continue;

        std::istringstream ss(trimmed);
        std::string kw;
        ss >> kw;

        if (kw == "v") {
            glm::vec3 p{};
            if (parseVec3(ss, p))
                positions.push_back(p);
        } else if (kw == "vn") {
            glm::vec3 n{};
            if (parseVec3(ss, n))
                normals.push_back(glm::normalize(n));
        } else if (kw == "vt") {
            glm::vec2 uv{};
            if (parseVec2(ss, uv))
                texcoords.push_back(uv);
        } else if (kw == "usemtl") {
            // OBJ material directives are ignored by design — materials are assigned
            // exclusively by the scene file.  usemtl still acts as a mesh boundary so
            // multi-material OBJs split into separately-assignable sub-meshes.
            if (!flushMesh(pending, scene, ctx, pool, positions, normals, texcoords, options))
                return false;
        } else if (kw == "o" || kw == "g") {
            if (!flushMesh(pending, scene, ctx, pool, positions, normals, texcoords, options))
                return false;
            std::string name;
            std::getline(ss, name);
            pending.name = trim(name);
            if (pending.name.empty())
                pending.name = path.stem().string();
        } else if (kw == "f") {
            // Only triangles are supported; non-triangle faces are skipped
            // with a warning. Export with "Triangulate Mesh" in Blender.
            std::string t0, t1, t2, extra;
            if (!(ss >> t0 >> t1 >> t2))
                continue;
            if (ss >> extra) {
                Logger::warn("OBJ '{}': skipping non-triangle face", path.stem().string());
                continue;
            }
            pending.indices.push_back(parseFaceVertex(t0));
            pending.indices.push_back(parseFaceVertex(t1));
            pending.indices.push_back(parseFaceVertex(t2));
        }
    }

    return flushMesh(pending, scene, ctx, pool, positions, normals, texcoords, options);
}
