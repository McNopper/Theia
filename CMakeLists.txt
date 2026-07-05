cmake_minimum_required(VERSION 3.28)

project(theia VERSION 0.6.8 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(FetchContent)
include(Platform)
include(VulkanSDK)
# compile_slang_shaders() is provided by Harmonia (cmake/CompileShaders.cmake),
# loaded when Harmonia is added via FetchContent below.

enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(VK_NO_PROTOTYPES)

# ── Shared libraries: Aether (file format) + Harmonia (common Vulkan) ─────────
# Consumed via FetchContent, pinned to published release tags.  For local
# development against working copies, override FETCHCONTENT_SOURCE_DIR_AETHER /
# FETCHCONTENT_SOURCE_DIR_HARMONIA.  Aether is pinned v0.5.0; Harmonia is pinned v0.6.4.
FetchContent_Declare(
    aether
    GIT_REPOSITORY https://github.com/McNopper/Aether.git
    GIT_TAG v0.6.8
)
FetchContent_Declare(
    harmonia
    GIT_REPOSITORY https://github.com/McNopper/Harmonia.git
    GIT_TAG v0.6.8
)
FetchContent_MakeAvailable(aether harmonia)

set(THEIA_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")

# Forward rendering shaders for real-time rendering
set(THEIA_ENTRY_SHADERS
    forward_render.task.slang
    forward_render.mesh.slang
    forward_render.frag.slang
    sky.vert.slang
    sky.frag.slang
    ibl_brdf_lut.comp.slang
    ibl_sheen_lut.comp.slang
    ibl_diffuse.comp.slang
    ibl_specular.comp.slang
    light_cull.comp.slang
    gi.comp.slang
    motion_vector.comp.slang
    hiz_build.comp.slang
    taa.comp.slang
)

# Theia-specific sources only. Common Vulkan code now lives in Harmonia and the
# file-format parsers in Aether, both linked via harmonia::harmonia.
set(THEIA_LIBRARY_SOURCES
    # Theia forward rendering
    src/theia/renderer/ForwardRenderer.cpp
    src/theia/renderer/IblPrecompute.cpp
    src/theia/renderer/LightCuller.cpp
    src/theia/renderer/GiPass.cpp
    src/theia/renderer/MotionVectorPass.cpp
    src/theia/renderer/TaaPass.cpp
    src/theia/renderer/HiZPass.cpp
    # Theia GPU scene (rasterizer/meshlet layout — renderer-specific)
    src/theia/scene/Scene.cpp
)

set(THEIA_DEMO_SOURCES
    src/demo/Application.cpp
)

set(THEIA_APP_SOURCES
    src/demo/main.cpp
)

# Volk is provided by Harmonia (created during FetchContent above); only define
# it ourselves if Harmonia did not (e.g. when building Theia in isolation).
if(NOT TARGET volk)
    add_library(volk STATIC "${VULKANSDK_VOLK_SOURCE}")
    target_include_directories(volk SYSTEM PUBLIC "${VULKANSDK_INCLUDE_DIR}")
    set_target_properties(volk PROPERTIES LINKER_LANGUAGE C)
endif()

add_library(theia_project_options INTERFACE)
theia_configure_target(theia_project_options)

add_library(theia_vulkan_sdk INTERFACE)
target_include_directories(theia_vulkan_sdk SYSTEM INTERFACE "${VULKANSDK_INCLUDE_DIR}")
target_link_libraries(theia_vulkan_sdk INTERFACE ${VULKANSDK_LIBRARIES})

if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(googletest)
        # gtest v1.15.2 has an implicit char8_t→char32_t conversion in C++23; suppress.
        foreach(_gt_tgt IN ITEMS gtest gtest_main gmock gmock_main)
            if(TARGET ${_gt_tgt})
                target_compile_options(${_gt_tgt} PRIVATE
                    $<$<CXX_COMPILER_ID:Clang,MSVC>:-Wno-character-conversion>
                )
            endif()
        endforeach()
    endif()

    if(TARGET gtest)
        set_property(TARGET gtest PROPERTY SYSTEM ON)
    endif()
    if(TARGET gtest_main)
        set_property(TARGET gtest_main PROPERTY SYSTEM ON)
    endif()
    if(NOT TARGET GTest::gtest AND TARGET gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(NOT TARGET GTest::gtest_main AND TARGET gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
endif()

find_package(meshoptimizer CONFIG REQUIRED)

# Compile Slang shaders at build time into THEIA_SHADER_OUTPUT_DIR; loaded at
# runtime via the THEIA_SHADER_DIR compile definition (same scheme as Harmonia
# and Hyperion).
# Harmonia's shared Slang modules (env_sample, math, bsdf, …) are the source
# of truth; add their directory so slangc resolves imports from there.
set(COMPILE_SLANG_EXTRA_INCLUDE_DIRS "${HARMONIA_SHADER_SOURCE_DIR}")
compile_slang_shaders(theia_shaders "${THEIA_SHADER_OUTPUT_DIR}" ${THEIA_ENTRY_SHADERS})
unset(COMPILE_SLANG_EXTRA_INCLUDE_DIRS)

add_library(theia_lib STATIC ${THEIA_LIBRARY_SOURCES})
add_dependencies(theia_lib theia_shaders)

target_include_directories(theia_lib
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# harmonia::harmonia brings the common Vulkan library plus Aether, Vulkan SDK,
# volk, stb and OpenEXR (incl. HARMONIA_HAS_OPENEXR) transitively.
target_link_libraries(theia_lib
    PUBLIC theia_project_options harmonia::harmonia meshoptimizer::meshoptimizer
)

target_compile_definitions(theia_lib
    PUBLIC THEIA_SHADER_DIR=\"${THEIA_SHADER_OUTPUT_DIR}\"
           THEIA_ASSETS_DIR=\"${AETHER_ASSETS_DIR}\"
)

add_library(theia_demo STATIC ${THEIA_DEMO_SOURCES})
target_link_libraries(theia_demo PUBLIC theia_lib)

add_executable(theia ${THEIA_APP_SOURCES})
target_link_libraries(theia PRIVATE theia_demo)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
