cmake_minimum_required(VERSION 3.28)

project(theia VERSION 0.1.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(FetchContent)
include(Platform)
include(VulkanSDK)
include(CompileShaders)

enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_compile_definitions(VK_NO_PROTOTYPES)

set(THEIA_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")

# Forward rendering shaders for real-time rendering
set(THEIA_ENTRY_SHADERS
    forward_render.vert.slang
    forward_render.frag.slang
)

# Theia library sources — same structure as Hyperion
set(THEIA_LIBRARY_SOURCES
    # Core Vulkan utilities
    src/hyperion/core/Buffer.cpp
    src/hyperion/core/CommandPool.cpp
    src/hyperion/core/Image.cpp
    src/hyperion/core/Logger.cpp
    src/hyperion/core/Queue.cpp
    src/hyperion/core/Timer.cpp
    # Utilities
    src/hyperion/utils/ColorSpace.cpp
    # Renderer utilities (deferred/forward pipeline foundation)
    src/hyperion/renderer/Camera.cpp
    src/hyperion/renderer/Descriptors.cpp
    src/hyperion/renderer/Pipeline.cpp
    src/hyperion/renderer/AccelerationStructure.cpp
    # Scene representation
    src/hyperion/scene/Geometry.cpp
    src/hyperion/scene/IblProbe.cpp
    src/hyperion/scene/Light.cpp
    src/hyperion/scene/Mesh.cpp
    src/hyperion/scene/ProceduralGeometry.cpp
    src/hyperion/scene/Scene.cpp
    src/hyperion/scene/Texture.cpp
    # Implementation details
    src/vma_impl.cpp
    src/stb_image_impl.cpp
    # CPU-only importers (no Vulkan deps)
    src/demo/importers/ObjImporter.cpp
    src/demo/importers/MaterialLibrary.cpp
    # Theia forward rendering
    src/theia/renderer/ForwardRenderer.cpp
    src/theia/renderer/IblPrecompute.cpp
    src/theia/renderer/LightCuller.cpp
    src/theia/renderer/SSRPass.cpp
)

set(THEIA_DEMO_SOURCES
    src/demo/vulkan_init/Context.cpp
    src/demo/vulkan_init/PhysicalDevice.cpp
    src/demo/vulkan_init/DebugUtils.cpp
    src/demo/presentation/Surface.cpp
    src/demo/presentation/Swapchain.cpp
    src/demo/presentation/ToneMapper.cpp
    src/demo/SceneLoader.cpp
    src/demo/Application.cpp
    src/demo/ImGuiLayer.cpp
    # Dear ImGui Vulkan + SDL3 backends compiled with volk preamble to suppress VK_NO_PROTOTYPES issues
    src/demo/imgui_vulkan_backend.cpp
    src/demo/imgui_sdl3_backend.cpp
)

set(THEIA_APP_SOURCES
    src/demo/main.cpp
)

# ── Dear ImGui (vendored in third_party/imgui) ──────────────────────────────
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui")

add_library(imgui_core STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
)
target_include_directories(imgui_core SYSTEM PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
if(MSVC)
    target_compile_options(imgui_core PRIVATE /W0)
else()
    target_compile_options(imgui_core PRIVATE -w)
endif()
set_property(TARGET imgui_core PROPERTY SYSTEM ON)
# ────────────────────────────────────────────────────────────────────────────

# Volk — compiled from the Vulkan SDK source
add_library(volk STATIC "${VULKANSDK_VOLK_SOURCE}")
target_include_directories(volk SYSTEM PUBLIC "${VULKANSDK_INCLUDE_DIR}")
set_target_properties(volk PROPERTIES LINKER_LANGUAGE C)

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

find_package(OpenEXR QUIET)
find_package(Stb REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)

# Compile SLANG shaders for forward rendering pipeline (TODO: create support libs)
# compile_slang_shaders(theia_shaders "${THEIA_SHADER_OUTPUT_DIR}" ${THEIA_ENTRY_SHADERS})

add_library(theia_lib STATIC ${THEIA_LIBRARY_SOURCES})

target_include_directories(theia_lib
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(theia_lib
    PUBLIC theia_project_options theia_vulkan_sdk volk meshoptimizer::meshoptimizer
)

target_include_directories(theia_lib
    SYSTEM PUBLIC "${Stb_INCLUDE_DIR}"
)

target_compile_definitions(theia_lib
    PUBLIC THEIA_SHADER_DIR=\"${THEIA_SHADER_OUTPUT_DIR}\"
)

if(OpenEXR_FOUND)
    if(TARGET OpenEXR::OpenEXR)
        set_property(TARGET OpenEXR::OpenEXR PROPERTY SYSTEM ON)
        target_link_libraries(theia_lib PUBLIC OpenEXR::OpenEXR)
    elseif(OPENEXR_LIBRARIES)
        target_link_libraries(theia_lib PUBLIC ${OPENEXR_LIBRARIES})
        if(OPENEXR_INCLUDE_DIRS)
            target_include_directories(theia_lib SYSTEM PUBLIC ${OPENEXR_INCLUDE_DIRS})
        endif()
    endif()
    target_compile_definitions(theia_lib PUBLIC THEIA_HAS_OPENEXR=1)
endif()

add_library(theia_demo STATIC ${THEIA_DEMO_SOURCES})
target_link_libraries(theia_demo PUBLIC theia_lib imgui_core)
# Backend wrappers and ImGuiLayer need imgui headers
target_include_directories(theia_demo SYSTEM PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
# Backend wrapper files are third-party; silence warnings
if(MSVC)
    set_source_files_properties(
        src/demo/imgui_vulkan_backend.cpp
        src/demo/imgui_sdl3_backend.cpp
        PROPERTIES COMPILE_FLAGS "/W0"
    )
else()
    set_source_files_properties(
        src/demo/imgui_vulkan_backend.cpp
        src/demo/imgui_sdl3_backend.cpp
        PROPERTIES COMPILE_FLAGS "-w"
    )
endif()

add_executable(theia ${THEIA_APP_SOURCES})
target_link_libraries(theia PRIVATE theia_demo)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
