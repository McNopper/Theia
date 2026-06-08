cmake_minimum_required(VERSION 3.28)

project(hyperion VERSION 0.1.0 LANGUAGES C CXX)

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

add_compile_definitions(VK_NO_PROTOTYPES)

set(HYPERION_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")

set(HYPERION_ENTRY_SHADERS
    raygen.slang
    closesthit.slang
    intersection.slang
    miss.slang
    shadow_miss.slang
    tonemap_vert.slang
    tonemap.slang
)

set(HYPERION_LIBRARY_SOURCES
    src/hyperion/core/Buffer.cpp
    src/hyperion/core/CommandPool.cpp
    src/hyperion/core/Image.cpp
    src/hyperion/core/Logger.cpp
    src/hyperion/core/Queue.cpp
    src/hyperion/core/Timer.cpp
    src/hyperion/renderer/AccelerationStructure.cpp
    src/hyperion/renderer/Camera.cpp
    src/hyperion/renderer/Descriptors.cpp
    src/hyperion/renderer/PathTracer.cpp
    src/hyperion/renderer/Pipeline.cpp
    src/hyperion/renderer/ShaderBindingTable.cpp
    src/hyperion/scene/Geometry.cpp
    src/hyperion/scene/IblProbe.cpp
    src/hyperion/scene/Light.cpp
    src/hyperion/scene/Mesh.cpp
    src/hyperion/scene/ProceduralGeometry.cpp
    src/hyperion/scene/Scene.cpp
    src/hyperion/scene/Texture.cpp
    src/hyperion/utils/ColorSpace.cpp
    src/vma_impl.cpp
    src/stb_image_impl.cpp
    # Pure-CPU importers — no Vulkan deps; live in hyperion_lib so unit tests
    # can link them without pulling in the full demo (SDL3, Vulkan context).
    src/demo/importers/ObjImporter.cpp
    src/demo/importers/MaterialLibrary.cpp
)

set(HYPERION_DEMO_SOURCES
    src/demo/vulkan_init/Context.cpp
    src/demo/vulkan_init/PhysicalDevice.cpp
    src/demo/vulkan_init/DebugUtils.cpp
    src/demo/presentation/Surface.cpp
    src/demo/presentation/Swapchain.cpp
    src/demo/presentation/ToneMapper.cpp
    src/demo/SceneLoader.cpp
    src/demo/Application.cpp
)

set(HYPERION_APP_SOURCES
    src/demo/main.cpp
)

# Volk — compiled from the Vulkan SDK source
add_library(volk STATIC "${VULKANSDK_VOLK_SOURCE}")
target_include_directories(volk SYSTEM PUBLIC "${VULKANSDK_INCLUDE_DIR}")
set_target_properties(volk PROPERTIES LINKER_LANGUAGE C)

add_library(hyperion_project_options INTERFACE)
hyperion_configure_target(hyperion_project_options)

add_library(hyperion_vulkan_sdk INTERFACE)
target_include_directories(hyperion_vulkan_sdk SYSTEM INTERFACE "${VULKANSDK_INCLUDE_DIR}")
target_link_libraries(hyperion_vulkan_sdk INTERFACE ${VULKANSDK_LIBRARIES})

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

compile_slang_shaders(hyperion_shaders "${HYPERION_SHADER_OUTPUT_DIR}" ${HYPERION_ENTRY_SHADERS})

add_library(hyperion_lib STATIC ${HYPERION_LIBRARY_SOURCES})
add_dependencies(hyperion_lib hyperion_shaders)

target_include_directories(hyperion_lib
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(hyperion_lib
    PUBLIC hyperion_project_options hyperion_vulkan_sdk volk
)

target_include_directories(hyperion_lib
    SYSTEM PUBLIC "${Stb_INCLUDE_DIR}"
)

target_compile_definitions(hyperion_lib
    PUBLIC HYPERION_SHADER_DIR=\"${HYPERION_SHADER_OUTPUT_DIR}\"
)

if(OpenEXR_FOUND)
    if(TARGET OpenEXR::OpenEXR)
        set_property(TARGET OpenEXR::OpenEXR PROPERTY SYSTEM ON)
        target_link_libraries(hyperion_lib PUBLIC OpenEXR::OpenEXR)
    elseif(OPENEXR_LIBRARIES)
        target_link_libraries(hyperion_lib PUBLIC ${OPENEXR_LIBRARIES})
        if(OPENEXR_INCLUDE_DIRS)
            target_include_directories(hyperion_lib SYSTEM PUBLIC ${OPENEXR_INCLUDE_DIRS})
        endif()
    endif()
    target_compile_definitions(hyperion_lib PUBLIC HYPERION_HAS_OPENEXR=1)
endif()

add_library(hyperion_demo STATIC ${HYPERION_DEMO_SOURCES})
target_link_libraries(hyperion_demo PUBLIC hyperion_lib)

add_executable(hyperion ${HYPERION_APP_SOURCES})
add_dependencies(hyperion hyperion_shaders)
target_link_libraries(hyperion PRIVATE hyperion_demo)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Standalone diagnostic executables — no GTest, no Hyperion libs.
# Only SDL3 + Vulkan + volk. Used to isolate driver/presentation issues.
add_executable(hyperion_diag tests/diag/test_sdl_vulkan_init.cpp)
target_link_libraries(hyperion_diag PRIVATE volk hyperion_vulkan_sdk hyperion_project_options)

# Swapchain loop integration test: creates a real visible window, clears it to
# orange every frame for 5 s. Validates SDL + Vulkan + swapchain end-to-end.
add_executable(hyperion_window_test tests/diag/window_swapchain_loop.cpp)
target_link_libraries(hyperion_window_test PRIVATE volk hyperion_vulkan_sdk hyperion_project_options)

# EXR diagnostic tool — checks pixel values for inf/NaN (only built when OpenEXR is available).
if(OpenEXR_FOUND)
    add_executable(check_exr tools/check_exr.cpp)
    target_link_libraries(check_exr PRIVATE hyperion_project_options)
    if(TARGET OpenEXR::OpenEXR)
        target_link_libraries(check_exr PRIVATE OpenEXR::OpenEXR)
    else()
        target_link_libraries(check_exr PRIVATE ${OPENEXR_LIBRARIES})
        if(OPENEXR_INCLUDE_DIRS)
            target_include_directories(check_exr SYSTEM PRIVATE ${OPENEXR_INCLUDE_DIRS})
        endif()
    endif()
endif()
