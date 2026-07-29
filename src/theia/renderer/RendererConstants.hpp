#ifndef THEIA_RENDERER_RENDERERCONSTANTS_HPP
#define THEIA_RENDERER_RENDERERCONSTANTS_HPP

#include <volk/volk.h>

#include <cstdint>

namespace theia {

inline constexpr std::uint32_t kMaxBindlessTextures = 256;

inline constexpr std::uint32_t kMeshTaskIndirectCmdSize =
    static_cast<std::uint32_t>(sizeof(VkDrawMeshTasksIndirectCommandEXT));

} // namespace theia

#endif // THEIA_RENDERER_RENDERERCONSTANTS_HPP
