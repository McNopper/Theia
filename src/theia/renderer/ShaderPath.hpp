#ifndef THEIA_RENDERER_SHADERPATH_HPP
#define THEIA_RENDERER_SHADERPATH_HPP

#include <filesystem>

namespace theia {

/// Resolves a bare SPIR-V filename (e.g. "forward_render.mesh.spv") against the
/// build-time shader output directory, mirroring how Harmonia and Hyperion load
/// their shaders. Never depends on the current working directory.
[[nodiscard]] inline std::filesystem::path shaderPath(const char* filename) {
    return std::filesystem::path(THEIA_SHADER_DIR) / filename;
}

} // namespace theia
#endif // THEIA_RENDERER_SHADERPATH_HPP
