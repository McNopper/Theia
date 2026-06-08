// Wrapper that compiles Dear ImGui's SDL3 backend.
// Suppress all warnings from third-party backend.
#ifdef _MSC_VER
#pragma warning(push, 0)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#endif
#include "imgui_impl_sdl3.cpp" // NOLINT — intentional .cpp include
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif
