// Wrapper that compiles Dear ImGui's Vulkan backend with volk loaded first.
// VK_NO_PROTOTYPES is defined globally; volk.h provides the function pointer variables.
// We define IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING so the backend supports our pipeline.

#define IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
#include <volk/volk.h>
// Suppress all warnings from third-party backend
#ifdef _MSC_VER
#pragma warning(push, 0)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#endif
#include "imgui_impl_vulkan.cpp" // NOLINT — intentional .cpp include to compile with our preamble
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif
