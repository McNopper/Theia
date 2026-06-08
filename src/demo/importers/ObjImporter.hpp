#pragma once

#include <filesystem>

#include "demo/importers/ISceneImporter.hpp"

class ObjImporter final : public ISceneImporter {
  public:
    bool import(const std::filesystem::path& path,
                Scene& scene,
                const DeviceContext& ctx,
                const CommandPool& pool,
                const ImportOptions& options = {}) override;
};
