#pragma once

#include <glm/glm.hpp>

#include <filesystem>
#include <string>
#include <unordered_map>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/scene/Scene.hpp"

class MaterialLibrary;

/// Options passed to ISceneImporter::import().
struct ImportOptions {
    /// Transform applied to every vertex position and normal.
    glm::mat4 worldTransform = glm::mat4(1.0f);

    /// If non-null, used to resolve material names from overrides below.
    const MaterialLibrary* library = nullptr;

    /// Override material for every group/object in the file.
    /// When non-empty, all sub-meshes use this material name (looked up in
    /// library).  Per-group overrides in groupMaterials still take priority.
    std::string overrideMaterial;

    /// Per-group material overrides: OBJ group/object name → material name
    /// (looked up in library).  Takes priority over overrideMaterial and the
    /// file's own usemtl directives.
    std::unordered_map<std::string, std::string> groupMaterials;
};

class ISceneImporter {
  public:
    virtual ~ISceneImporter() = default;

    virtual bool import(const std::filesystem::path& path,
                        Scene& scene,
                        const DeviceContext& ctx,
                        const CommandPool& pool,
                        const ImportOptions& options = {}) = 0;
};
