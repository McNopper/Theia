cmake_minimum_required(VERSION 3.28)

project(theia VERSION 0.7.17 LANGUAGES C CXX)

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

    # Sanitizer lane (Clang/GCC only):
    #   cmake -DTHEIA_SANITIZER=address|undefined|thread ...
    set(THEIA_SANITIZER "" CACHE STRING "Sanitizer: address, undefined, or thread (Clang/GCC only)")
    if(THEIA_SANITIZER)
        if(MSVC)
            message(FATAL_ERROR "THEIA_SANITIZER requires Clang/GCC (use a Ninja+clang or WSL configure)")
        endif()
        add_compile_options("-fsanitize=${THEIA_SANITIZER}" -fno-omit-frame-pointer)
        add_link_options("-fsanitize=${THEIA_SANITIZER}")
    endif()
    # Coverage lane (Clang/GCC only):
    #   cmake -DTHEIA_COVERAGE=ON -B build/coverage ...   then: python tools/coverage.py
    set(THEIA_COVERAGE OFF CACHE BOOL "Instrument for source coverage (Clang/GCC only)")
    if(THEIA_COVERAGE)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            message(FATAL_ERROR "THEIA_COVERAGE requires Clang/GCC (use a Ninja+clang or WSL configure)")
        endif()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
            if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
                # clang-cl + MSVC linker: the profile runtime only links when
                # lld-link drives the link step
                add_link_options(-fprofile-instr-generate -fuse-ld=lld)
            else()
                add_link_options(-fprofile-instr-generate)
            endif()
        else()
            add_compile_options(--coverage)
            add_link_options(--coverage)
        endif()
    endif()


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
# FETCHCONTENT_SOURCE_DIR_HARMONIA.
FetchContent_Declare(
    aether
    GIT_REPOSITORY https://github.com/McNopper/Aether.git
    GIT_TAG v0.8.0
)
FetchContent_Declare(
    harmonia
    GIT_REPOSITORY https://github.com/McNopper/Harmonia.git
    GIT_TAG v0.7.16
)
# slang-math arrives transitively: Aether and Harmonia are its direct consumers
# and pin it in their own CMakeLists; Harmonia is the renderer-tier vendor
# (Theia and Hyperion take the version Harmonia fetches).  Never declare
# slang-math here - a direct declare would fork the renderer tier's math
# version.
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

# Per-band sources (AR2), bottom-up. Common Vulkan code lives in Harmonia and
# the file-format parsers in Aether; the bands below carry the Theia-side
# module DAG of ARCHITECTURE.md (theia/scene + theia/camera leaves -> depth ->
# cull -> renderer -> gi/postfx -> demo).
set(THEIA_scene_SOURCES
    # Theia GPU scene (rasterizer/meshlet layout — renderer-specific)
    src/theia/scene/Scene.cpp
    src/theia/scene/Scene.hpp
)
# theia_camera (CameraController.hpp, CameraJitter.hpp) and
# theia_renderer_core (theia/renderer/RendererConstants.hpp + ShaderPath.hpp)
# are header-only bands — no source lists, INTERFACE targets below.
set(THEIA_depth_SOURCES
    src/theia/depth/HiZPass.cpp
    src/theia/depth/HiZPass.hpp
)
set(THEIA_cull_SOURCES
    src/theia/cull/GpuCullPass.cpp
    src/theia/cull/GpuCullPass.hpp
    src/theia/cull/GpuDrivenState.cpp
    src/theia/cull/GpuDrivenState.hpp
    src/theia/cull/LightCuller.cpp
    src/theia/cull/LightCuller.hpp
)
set(THEIA_renderer_SOURCES
    # One C++ class, TU-split across the theia/renderer and theia/pipelines
    # arch modules (see the theia_renderer target comment below): the TUs
    # reference each other's symbols, so they ship in one static library.
    src/theia/renderer/ForwardRenderer.cpp
    src/theia/renderer/ForwardRenderer.hpp
    src/theia/pipelines/ForwardRendererPipelines.cpp
)
set(THEIA_gi_SOURCES
    src/theia/gi/GiPass.cpp
    src/theia/gi/GiPass.hpp
    src/theia/gi/PairingTextures.cpp
    src/theia/gi/PairingTextures.hpp
)
set(THEIA_postfx_SOURCES
    src/theia/postfx/MotionVectorPass.cpp
    src/theia/postfx/MotionVectorPass.hpp
    src/theia/postfx/TaaPass.cpp
    src/theia/postfx/TaaPass.hpp
)

set(THEIA_DEMO_SOURCES
    src/demo/Application.cpp
    src/demo/Application.hpp
    src/demo/ApplicationCompute.cpp
    src/demo/EnvImportanceResources.hpp
    src/demo/ViewAccumulation.hpp
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
        # gtest v1.15.2 has an implicit char8_t->char32_t conversion in C++23;
        # suppress it - but only where the compiler actually KNOWS
        # -Wno-character-conversion (clang added that group after the runner
        # images' bundled clang; unknown -Wno-* options are silent until
        # -Werror, where they hard-fail - observed in CI). The check runs with
        # -Werror so it fails exactly like the real build would; a compiler
        # whose check fails also cannot emit the warning, so flag and warning
        # stay coupled (clang-cl 22 warns AND knows the flag).
        include(CheckCXXCompilerFlag)
        set(_theia_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror")
        check_cxx_compiler_flag("-Wno-character-conversion" THEIA_HAVE_WNO_CHARACTER_CONVERSION)
        set(CMAKE_REQUIRED_FLAGS "${_theia_saved_required_flags}")
        if(THEIA_HAVE_WNO_CHARACTER_CONVERSION)
            foreach(_gt_tgt IN ITEMS gtest gtest_main gmock gmock_main)
                if(TARGET ${_gt_tgt})
                    target_compile_options(${_gt_tgt} PRIVATE
                        $<$<CXX_COMPILER_ID:Clang,MSVC>:-Wno-character-conversion>
                    )
                endif()
            endforeach()
        endif()
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

# ── Layered targets (AR2, Harmonia pattern) ──────────────────────────────────
# The include-DAG of ARCHITECTURE.md expressed as target_link_libraries — an
# upward include of an out-of-band symbol is now a LINK error. Bottom-up;
# shared leaves sit lowest. Only the leaves publish src/ (every band reaches
# one transitively).
#
#   theia_scene   theia_camera   theia_renderer_core      (layer leaves)
#         \____________|______________/  |
#          theia_depth -> renderer_core |
#          theia_cull -> scene, depth, renderer_core
#          theia_renderer -> cull, depth, scene, camera, renderer_core
#          theia_gi -> scene, renderer_core     theia_postfx -> renderer_core
#          theia_demo -> renderer, cull, gi, postfx, scene, camera
#          theia (exe) -> demo
#
# theia_renderer carries BOTH ForwardRenderer TUs (the theia/renderer and
# theia/pipelines arch modules): the class is TU-split with mutual symbol
# references (ForwardRenderer.cpp calls prepareAttachments/beginSceneRendering/
# bindMeshSets/updateSceneDescriptors, ForwardRendererPipelines.cpp calls
# loadShaderModule), so two targets could not flow one way. The file layout and
# the include graph are unchanged; this is a build-graph grouping only.
# harmonia::harmonia brings the common Vulkan library plus Aether, Vulkan SDK,
# volk, stb and OpenEXR (incl. HARMONIA_HAS_OPENEXR) transitively.
add_library(theia_scene STATIC ${THEIA_scene_SOURCES})
target_include_directories(theia_scene
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(theia_scene
    PUBLIC theia_project_options theia_vulkan_sdk harmonia::harmonia
           meshoptimizer::meshoptimizer
)

# theia_camera and theia_renderer_core are header-only (no .cpp, no symbols
# of their own): INTERFACE targets — no archive, include dirs/definitions
# propagate to their consumers through the link interface.
add_library(theia_camera INTERFACE)
target_include_directories(theia_camera
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(theia_camera
    INTERFACE theia_project_options theia_vulkan_sdk harmonia::harmonia
)

add_library(theia_renderer_core INTERFACE)
target_include_directories(theia_renderer_core
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(theia_renderer_core
    INTERFACE theia_project_options theia_vulkan_sdk harmonia::harmonia
)
target_compile_definitions(theia_renderer_core
    INTERFACE THEIA_SHADER_DIR=\"${THEIA_SHADER_OUTPUT_DIR}\"
)

add_library(theia_depth STATIC ${THEIA_depth_SOURCES})
target_link_libraries(theia_depth
    PUBLIC theia_project_options theia_vulkan_sdk harmonia::harmonia
           theia_renderer_core
)

add_library(theia_cull STATIC ${THEIA_cull_SOURCES})
target_link_libraries(theia_cull
    PUBLIC theia_project_options theia_vulkan_sdk harmonia::harmonia
           theia_scene theia_depth theia_renderer_core
)

add_library(theia_renderer STATIC ${THEIA_renderer_SOURCES})
target_link_libraries(theia_renderer
    PUBLIC theia_project_options theia_vulkan_sdk harmonia::harmonia
           theia_cull theia_depth theia_scene theia_camera theia_renderer_core
)

add_library(theia_gi STATIC ${THEIA_gi_SOURCES})
target_link_libraries(theia_gi
    PUBLIC theia_project_options theia_vulkan_sdk harmonia::harmonia
           theia_scene theia_renderer_core
)

add_library(theia_postfx STATIC ${THEIA_postfx_SOURCES})
target_link_libraries(theia_postfx
    PUBLIC theia_project_options theia_vulkan_sdk harmonia::harmonia
           theia_renderer_core
)

add_library(theia_demo STATIC ${THEIA_DEMO_SOURCES})
add_dependencies(theia_demo theia_shaders)
target_link_libraries(theia_demo
    PUBLIC theia_project_options theia_vulkan_sdk harmonia::harmonia
           theia_renderer theia_cull theia_gi theia_postfx theia_scene theia_camera
)
target_compile_definitions(theia_demo
    PUBLIC THEIA_ASSETS_DIR=\"${AETHER_ASSETS_DIR}\"
)

add_executable(theia ${THEIA_APP_SOURCES})
target_link_libraries(theia PRIVATE theia_demo)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Static-analysis ctest lane: parallel clang-tidy over
# build/compile_commands.json, classified per the .clang-tidy WarningsAsErrors
# contract. Skips (77) when clang-tidy / Python3 / the database is unavailable.
if(PROJECT_IS_TOP_LEVEL)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME test_tidy
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/check_tidy.py)
        set_tests_properties(test_tidy PROPERTIES SKIP_RETURN_CODE 77 LABELS analysis RUN_SERIAL TRUE)
        # The architecture gate (LOCAL ONLY - not a CI gate): the engine and
        # the ARCHITECTURE.md contract are vendored in the repo, but the gate
        # runs from a per-user engine install (the paths below). Machines
        # without that install - e.g. CI runners - skip registration.
        set(CHECK_ARCH_ENGINE "")
        foreach(_cand "$ENV{USERPROFILE}/.config/opencode/skills/check-arch/scripts/check_arch.py"
                      "$ENV{HOME}/.config/opencode/skills/check-arch/scripts/check_arch.py")
          if(EXISTS "${_cand}")
            set(CHECK_ARCH_ENGINE "${_cand}")
            break()
          endif()
        endforeach()
        if(CHECK_ARCH_ENGINE)
            add_test(NAME test_arch
                     COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/check_arch.py
                             --repo "${CMAKE_SOURCE_DIR}" --doc "${CMAKE_SOURCE_DIR}/ARCHITECTURE.md")
            set_tests_properties(test_arch PROPERTIES SKIP_RETURN_CODE 77 LABELS analysis RUN_SERIAL TRUE)
        endif()
    endif()
endif()

