#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "demo/SceneLoader.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "demo/importers/ISceneImporter.hpp"
#include "demo/importers/MaterialLibrary.hpp"
#include "demo/importers/ObjImporter.hpp"
#include "hyperion/core/Logger.hpp"
#include "hyperion/scene/ProceduralGeometry.hpp"
#include "hyperion/scene/Scene.hpp"
#include "hyperion/scene/Texture.hpp"

namespace {

// ── String helpers ────────────────────────────────────────────────────────────

[[nodiscard]] std::string_view trimSv(std::string_view sv) noexcept {
    const auto first = sv.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return {};
    return sv.substr(first, sv.find_last_not_of(" \t\r\n") - first + 1);
}

[[nodiscard]] std::string_view stripComment(std::string_view line) noexcept {
    const auto pos = line.find('#');
    return trimSv(pos == std::string_view::npos ? line : line.substr(0, pos));
}

// ── Value parsers ─────────────────────────────────────────────────────────────

bool parseVec3(std::string_view text, glm::vec3& out) {
    std::istringstream ss{std::string(text)};
    return static_cast<bool>(ss >> out.x >> out.y >> out.z);
}

bool parseFloat(std::string_view text, float& out) {
    std::istringstream ss{std::string(text)};
    return static_cast<bool>(ss >> out);
}

bool parseUint(std::string_view text, uint32_t& out) {
    std::istringstream ss{std::string(text)};
    return static_cast<bool>(ss >> out);
}

// ── Pending geometry block ────────────────────────────────────────────────────

struct Block {
    enum class Kind { None, Object, Sphere, Box } kind = Kind::None;

    // Object
    std::string objPath;

    // Sphere — radius; centre set via translate
    float sphereRadius = 0.0f;

    // Box — half-extents; positioned/oriented via TRS
    glm::vec3 boxHalf{0.0f};

    // Shared TRS  (glTF T × R × S convention)
    std::string materialName;
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // GLM stores (w, x, y, z)
    glm::vec3 scale{1.0f};

    // Per-group material assignments for Object blocks.
    // Populated by "material GroupName MatName" scene keywords.
    // Key = OBJ group/object name; value = material name looked up in library.
    std::unordered_map<std::string, std::string> groupMaterials;
};

// ── Pending camera block ──────────────────────────────────────────────────────

struct CameraBlock {
    glm::vec3 position{0.0f};
    bool hasTranslate = false;

    // Orientation: rotate and look_at are mutually exclusive; last one wins.
    std::optional<glm::quat> rotation; // from rotate / rotate_y
    std::optional<glm::vec3> lookAt;   // from look_at

    glm::vec3 up{0.0f, 1.0f, 0.0f};
    std::optional<float> vfov;
};

// Build a 4×4 TRS matrix from the block's T/R/S fields.
[[nodiscard]] glm::mat4 trsMatrix(const Block& b) {
    return glm::translate(glm::mat4(1.0f), b.translation) * glm::mat4_cast(b.rotation) *
           glm::scale(glm::mat4(1.0f), b.scale);
}

// ── Block flusher ─────────────────────────────────────────────────────────────

[[nodiscard]] bool flushBlock(Block& blk,
                              Scene& scene,
                              const DeviceContext& ctx,
                              const CommandPool& pool,
                              MaterialLibrary& lib,
                              const std::filesystem::path& assetsDir,
                              std::unordered_map<std::string, uint32_t>& texCache) {

    // Pre-load textures for all materials referenced in this block.
    // Must run before ObjImporter/addMaterial so that patched texture indices
    // are visible when lib.getOrDefault() is called.
    auto loadMatTextures = [&](const std::string& matName) {
        if (matName.empty())
            return;
        const auto refs = lib.textureRefs(matName);
        if (!refs)
            return;

        // Slot order matches GpuMaterial textures: [0-3] base_color, normal, ORM, emission;
        // [4-6] coat_normal, tangent, coat_tangent (textureIndices2).
        const std::array<std::pair<uint32_t, const MaterialLibrary::MaterialTextureRef*>, 7> slots{{
            {0u, &refs->base_color},
            {1u, &refs->normal},
            {2u, &refs->orm},
            {3u, &refs->emission},
            {4u, &refs->coat_normal},
            {5u, &refs->tangent},
            {6u, &refs->coat_tangent},
        }};

        for (const auto& [slot, ref] : slots) {
            if (ref->empty())
                continue;

            const std::string& relPath = ref->path;
            if (const auto it = texCache.find(relPath); it != texCache.end()) {
                // Already uploaded — just patch the index.
                lib.patchTextureIndex(matName, slot, it->second);
                continue;
            }

            auto result = Texture::loadFromFile(ctx, pool, assetsDir / relPath, ref->colorSpace, matName);
            if (!result) {
                Logger::warn("SceneLoader: failed to load texture '{}' for material '{}'", relPath, matName);
                continue;
            }

            const uint32_t idx = scene.addTexture(std::move(*result));
            texCache.emplace(relPath, idx);
            lib.patchTextureIndex(matName, slot, idx);
            Logger::info("SceneLoader: loaded texture '{}' (slot {}) → index {}", relPath, slot, idx);
        }
    };

    loadMatTextures(blk.materialName);
    for (const auto& [groupName, matName] : blk.groupMaterials)
        loadMatTextures(matName);

    switch (blk.kind) {
    case Block::Kind::None:
        return true;

    case Block::Kind::Object: {
        if (blk.objPath.empty()) {
            Logger::warn("SceneLoader: empty object path — skipping");
            return true;
        }
        ObjImporter importer;
        return importer.import(assetsDir / blk.objPath,
                               scene,
                               ctx,
                               pool,
                               ImportOptions{
                                   .worldTransform = trsMatrix(blk),
                                   .library = &lib,
                                   .overrideMaterial = blk.materialName,
                                   .groupMaterials = blk.groupMaterials,
                               });
    }

    case Block::Kind::Sphere: {
        if (blk.sphereRadius <= 0.0f) {
            Logger::warn("SceneLoader: sphere radius ≤ 0 — skipping");
            return true;
        }
        // Tessellate into a UV sphere mesh so the forward rasterizer can render it.
        // scale.x is used as a uniform radius multiplier (spheres are symmetric).
        const float radius = blk.sphereRadius * blk.scale.x;
        const uint32_t mat = scene.addMaterial(lib.getOrDefault(blk.materialName));
        MeshData mesh = ProceduralGeometry::makeSphere(blk.translation, radius);
        return scene.addMesh(ctx, pool, std::move(mesh), mat, "sphere") != std::numeric_limits<uint32_t>::max();
    }

    case Block::Kind::Box: {
        if (blk.boxHalf == glm::vec3(0.0f)) {
            Logger::warn("SceneLoader: box half-extents are zero — skipping");
            return true;
        }
        const uint32_t mat = scene.addMaterial(lib.getOrDefault(blk.materialName));
        MeshData mesh = ProceduralGeometry::makeBox(blk.boxHalf, trsMatrix(blk));
        return scene.addMesh(ctx, pool, std::move(mesh), mat, "box") != std::numeric_limits<uint32_t>::max();
    }
    }
    return true; // unreachable
}

} // namespace

// ── SceneLoader::load ─────────────────────────────────────────────────────────

std::optional<SceneLoader::SceneConfig> SceneLoader::load(const std::filesystem::path& sceneFile,
                                                          const std::filesystem::path& assetsDir,
                                                          Scene& scene,
                                                          const DeviceContext& ctx,
                                                          const CommandPool& pool) {
    std::ifstream file(sceneFile);
    if (!file) {
        Logger::error("SceneLoader: cannot open '{}'", sceneFile.string());
        return std::nullopt;
    }

    SceneConfig cfg{};
    MaterialLibrary lib;
    std::unordered_map<std::string, uint32_t> texCache; // relPath → texture index

    // Active block state.
    enum class ActiveBlock { None, Geometry, Camera };
    ActiveBlock activeBlock = ActiveBlock::None;
    Block blk{};
    CameraBlock camBlk{};

    // Flush pending geometry block.
    auto flushGeometry = [&]() -> bool { return flushBlock(blk, scene, ctx, pool, lib, assetsDir, texCache); };

    // Flush pending camera block into cfg.
    auto flushCamera = [&]() {
        if (camBlk.hasTranslate)
            cfg.cameraPos = camBlk.position;

        if (camBlk.rotation) {
            // Derive look-at target and up from quaternion.
            // Camera default forward is -Z; target = position + forward.
            const glm::vec3 pos = camBlk.hasTranslate ? camBlk.position : glm::vec3(278.0f, 273.0f, -800.0f);
            const glm::vec3 forward = *camBlk.rotation * glm::vec3(0.0f, 0.0f, -1.0f);
            cfg.cameraAt = pos + forward;
            cfg.cameraUp = *camBlk.rotation * glm::vec3(0.0f, 1.0f, 0.0f);
        } else if (camBlk.lookAt) {
            cfg.cameraAt = *camBlk.lookAt;
            cfg.cameraUp = camBlk.up;
        }

        if (camBlk.vfov)
            cfg.cameraVfov = *camBlk.vfov;
    };

    // Flush whichever block type is currently active.
    auto flushActive = [&]() -> bool {
        if (activeBlock == ActiveBlock::Camera) {
            flushCamera();
            return true;
        }
        return flushGeometry();
    };

    std::string line;
    while (std::getline(file, line)) {
        const std::string_view sv = stripComment(line);
        if (sv.empty())
            continue;

        std::istringstream ss{std::string(sv)};
        std::string kw;
        ss >> kw;
        std::string rest;
        std::getline(ss, rest);
        const std::string_view rv = trimSv(rest);

        // ── Block starters ────────────────────────────────────────────────────
        if (kw == "camera") {
            if (!flushActive())
                return std::nullopt;
            camBlk = {};
            activeBlock = ActiveBlock::Camera;

        } else if (kw == "instance") {
            // Instantiate a Wavefront OBJ as a scene object (geometry only — the
            // OBJ's own materials are never imported; assign via usemtl/material).
            if (!flushActive())
                return std::nullopt;
            blk = {};
            blk.kind = Block::Kind::Object;
            blk.objPath = std::string(rv);
            activeBlock = ActiveBlock::Geometry;

        } else if (kw == "sphere") {
            if (!flushActive())
                return std::nullopt;
            blk = {};
            blk.kind = Block::Kind::Sphere;
            parseFloat(rv, blk.sphereRadius);
            activeBlock = ActiveBlock::Geometry;

        } else if (kw == "box") {
            if (!flushActive())
                return std::nullopt;
            blk = {};
            blk.kind = Block::Kind::Box;
            std::istringstream vs{std::string(rv)};
            vs >> blk.boxHalf.x >> blk.boxHalf.y >> blk.boxHalf.z;
            activeBlock = ActiveBlock::Geometry;

        }
        // ── Shared modifiers: translate / rotate / rotate_y ───────────────────
        else if (kw == "translate") {
            if (activeBlock == ActiveBlock::Camera) {
                parseVec3(rv, camBlk.position);
                camBlk.hasTranslate = true;
            } else {
                parseVec3(rv, blk.translation);
            }

        } else if (kw == "rotate") {
            // glTF convention on disk: qx qy qz qw
            // GLM quat constructor:    quat(w, x, y, z)
            float qx = 0.0f, qy = 0.0f, qz = 0.0f, qw = 1.0f;
            std::istringstream vs{std::string(rv)};
            vs >> qx >> qy >> qz >> qw;
            const glm::quat q(qw, qx, qy, qz);
            if (activeBlock == ActiveBlock::Camera) {
                camBlk.rotation = q;
                camBlk.lookAt = std::nullopt; // rotate wins over look_at
            } else {
                blk.rotation = q;
            }

        } else if (kw == "rotate_y") {
            float deg = 0.0f;
            parseFloat(rv, deg);
            const glm::quat q = glm::angleAxis(glm::radians(deg), glm::vec3(0.0f, 1.0f, 0.0f));
            if (activeBlock == ActiveBlock::Camera) {
                camBlk.rotation = q;
                camBlk.lookAt = std::nullopt;
            } else {
                blk.rotation = q;
            }

        }
        // ── Geometry-only modifiers ───────────────────────────────────────────
        else if (kw == "usemtl") {
            blk.materialName = std::string(rv);

        } else if (kw == "material" && activeBlock == ActiveBlock::Geometry) {
            // "material GroupName MatName" — per-group assignment for Object blocks.
            std::istringstream ms{std::string(rv)};
            std::string groupName, matName;
            if (ms >> groupName >> matName)
                blk.groupMaterials[groupName] = matName;

        } else if (kw == "scale") {
            std::istringstream vs{std::string(rv)};
            float sx = 1.0f, sy = 1.0f, sz = 1.0f;
            vs >> sx;
            if (!(vs >> sy >> sz)) {
                sy = sx;
                sz = sx; // single value → uniform scale
            }
            blk.scale = {sx, sy, sz};

        }
        // ── Camera-only modifiers ─────────────────────────────────────────────
        else if (kw == "look_at") {
            glm::vec3 v;
            if (parseVec3(rv, v)) {
                camBlk.lookAt = v;
                camBlk.rotation = std::nullopt; // look_at wins over rotate
            }

        } else if (kw == "up") {
            parseVec3(rv, camBlk.up);

        } else if (kw == "vfov") {
            float v;
            if (parseFloat(rv, v))
                camBlk.vfov = v;

        }
        // ── Material libraries ────────────────────────────────────────────────
        else if (kw == "mtllib") {
            lib.load(assetsDir / std::string(rv));

        }
        // ── Render settings ───────────────────────────────────────────────────
        else if (kw == "spp") {
            uint32_t v;
            if (parseUint(rv, v))
                cfg.spp = v;
        } else if (kw == "max_depth") {
            uint32_t v;
            if (parseUint(rv, v))
                cfg.maxDepth = v;
        } else if (kw == "env_unit_nits") {
            float v;
            if (parseFloat(rv, v))
                cfg.envUnitNits = v;
        } else if (kw == "ev100") {
            float v;
            if (parseFloat(rv, v))
                cfg.cameraEv100 = v;
        } else if (kw == "env_map") {
            if (!rv.empty())
                cfg.envMapFile = std::filesystem::path(std::string(rv));
        } else if (kw == "tonemapper") {
            const std::string name(rv);
            if (name == "aces")
                cfg.tonemapper = 0u;
            else if (name == "agx")
                cfg.tonemapper = 1u;
            else if (name == "reinhard")
                cfg.tonemapper = 2u;
            else if (name == "hable")
                cfg.tonemapper = 3u;
            else
                Logger::warn("SceneLoader: unknown tonemapper '{}' — using default (aces)", name);
        }
        // Unknown keywords are silently ignored, consistent with OBJ/MTL.
    }

    // Flush the last pending block.
    if (!flushActive())
        return std::nullopt;

    Logger::info("SceneLoader: loaded '{}'", sceneFile.filename().string());
    return cfg;
}
