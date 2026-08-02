cmake_minimum_required(VERSION 3.28)

project(theia VERSION 0.7.5 LANGUAGES C CXX)

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

# ============================================================
# AI-first static analysis (clang-tidy / cppcheck / clang-format)
#
# Canonical setup shared by every repo in the pipeline, ported from the
# Hephaestus cpp/ template. Analysis runs only when the generator emits a
# compile database (Ninja/Makefiles) and the compiler is Clang/GNU-
# compatible; otherwise it is skipped and the project configures/builds
# normally. The analysis *targets* are opt-in; in-build analysis is OFF by
# default so the build is never blocked by findings.
#
# Guarded on PROJECT_IS_TOP_LEVEL so the targets are only defined when this
# project is built standalone: several of these repos are consumed via
# FetchContent by downstream pipeline repos, where redefining the targets
# would clash with the consumer's own.
# ============================================================
if(PROJECT_IS_TOP_LEVEL)
    set(PROJECT_REPORTS_DIR ${CMAKE_BINARY_DIR}/reports)
    file(MAKE_DIRECTORY ${PROJECT_REPORTS_DIR})

    set(PROJECT_ANALYSIS_AVAILABLE FALSE)
    if(CMAKE_EXPORT_COMPILE_COMMANDS
       AND (CMAKE_GENERATOR MATCHES "Ninja" OR CMAKE_GENERATOR MATCHES "Makefiles")
       AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
        set(PROJECT_ANALYSIS_AVAILABLE TRUE)
        message(STATUS "AI-first static analysis enabled (${CMAKE_GENERATOR}, ${CMAKE_CXX_COMPILER_ID})")
    else()
        message(STATUS "AI-first static analysis skipped for this toolchain "
            "(${CMAKE_GENERATOR}, ${CMAKE_CXX_COMPILER_ID}); "
            "use Ninja/Makefiles with a Clang- or GNU-compatible compiler to enable it")
    endif()

    file(GENERATE
        OUTPUT ${PROJECT_REPORTS_DIR}/analysis-status.txt
        CONTENT "analysis=$<BOOL:${PROJECT_ANALYSIS_AVAILABLE}>\ngenerator=${CMAKE_GENERATOR}\ncompiler=${CMAKE_CXX_COMPILER_ID}\n"
    )

    option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" ON)
    option(ENABLE_CLANG_TIDY_IN_BUILD "Run clang-tidy during compilation" OFF)
    option(ENABLE_CLANG_FORMAT "Enable clang-format targets" ON)
    option(ENABLE_CPPCHECK "Enable cppcheck static analysis target" ON)
    option(ENABLE_CPPCHECK_IN_BUILD "Run cppcheck alongside the compiler" OFF)

    # All first-party C/C++ sources (clang-format); compiled translation units
    # only (clang-tidy) — headers are analysed transitively via HeaderFilterRegex.
    file(GLOB_RECURSE PROJECT_CXX_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hxx"
    )
    file(GLOB_RECURSE PROJECT_CXX_TUS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cxx"
    )

    # ----- clang-tidy -----
    if(ENABLE_CLANG_TIDY AND PROJECT_ANALYSIS_AVAILABLE)
        find_program(CLANG_TIDY_EXE NAMES clang-tidy)
        if(CLANG_TIDY_EXE)
            if(ENABLE_CLANG_TIDY_IN_BUILD)
                set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
                message(STATUS "clang-tidy enabled during build: ${CLANG_TIDY_EXE}")
            else()
                message(STATUS "clang-tidy target enabled: ${CLANG_TIDY_EXE}")
            endif()

            set(CLANG_TIDY_REPORT_DIR ${PROJECT_REPORTS_DIR}/clang-tidy)
            file(MAKE_DIRECTORY ${CLANG_TIDY_REPORT_DIR})

            add_custom_target(tidy
                COMMAND ${CLANG_TIDY_EXE}
                    -p ${CMAKE_BINARY_DIR}
                    --export-fixes=${CLANG_TIDY_REPORT_DIR}/fixes.yaml
                    ${PROJECT_CXX_TUS}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Running clang-tidy; fixes -> ${CLANG_TIDY_REPORT_DIR}/fixes.yaml"
                VERBATIM
            )
        else()
            message(STATUS "clang-tidy not found")
        endif()
    endif()

    # ----- cppcheck -----
    if(ENABLE_CPPCHECK AND PROJECT_ANALYSIS_AVAILABLE)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        include(cppcheck)
    endif()

    # ----- clang-format -----
    if(ENABLE_CLANG_FORMAT)
        find_program(CLANG_FORMAT_EXE NAMES clang-format)
        if(CLANG_FORMAT_EXE)
            add_custom_target(format
                COMMAND ${CLANG_FORMAT_EXE}
                    -i
                    --style=file
                    ${PROJECT_CXX_SOURCES}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Formatting source files with clang-format"
                VERBATIM
            )

            add_custom_target(format-check
                COMMAND ${CLANG_FORMAT_EXE}
                    --dry-run
                    --Werror
                    --style=file
                    ${PROJECT_CXX_SOURCES}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Checking source formatting with clang-format"
                VERBATIM
            )

            message(STATUS "clang-format enabled: ${CLANG_FORMAT_EXE}")
        else()
            message(STATUS "clang-format not found")
        endif()
    endif()

    # ----- Aggregate verification targets -----
    # verify:      build + test (fast default).
    # verify-full: verify + format-check + tidy + cppcheck (opt-in strict).
    add_custom_target(verify
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG>
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --build-config $<CONFIG>
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Fast verification: build + test"
        VERBATIM
    )

    add_custom_target(verify-full
        COMMENT "Full verification: verify + format-check + tidy + cppcheck"
    )
    add_dependencies(verify-full verify)
    if(TARGET format-check)
        add_dependencies(verify-full format-check)
    endif()
    if(TARGET cppcheck)
        add_dependencies(verify-full cppcheck)
    endif()
    if(TARGET tidy)
        add_dependencies(verify-full tidy)
    endif()
endif()

# ── Shared libraries: Aether (file format) + Harmonia (common Vulkan) ─────────
# Consumed via FetchContent, pinned to published release tags.  For local
# development against working copies, override FETCHCONTENT_SOURCE_DIR_AETHER /
# FETCHCONTENT_SOURCE_DIR_HARMONIA.  Aether is pinned v0.7.2; Harmonia is pinned
# to the latest release.
FetchContent_Declare(
    aether
    GIT_REPOSITORY https://github.com/McNopper/Aether.git
    GIT_TAG v0.7.2
)
FetchContent_Declare(
    harmonia
    GIT_REPOSITORY https://github.com/McNopper/Harmonia.git
    GIT_TAG v0.7.6
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
    light_cull.comp.slang
    forward_cull.comp.slang
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
    src/theia/renderer/GpuCullPass.cpp
    src/theia/renderer/LightCuller.cpp
    src/theia/renderer/GiPass.cpp
    src/theia/renderer/MotionVectorPass.cpp
    src/theia/renderer/TaaPass.cpp
    src/theia/renderer/HiZPass.cpp
    src/theia/renderer/GpuDrivenState.cpp
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

