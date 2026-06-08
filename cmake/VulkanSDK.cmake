include_guard(GLOBAL)

if(NOT DEFINED ENV{VULKAN_SDK} OR "$ENV{VULKAN_SDK}" STREQUAL "")
    message(FATAL_ERROR "VULKAN_SDK environment variable is not set.")
endif()

file(TO_CMAKE_PATH "$ENV{VULKAN_SDK}" _vulkan_sdk_root)

if(WIN32)
    set(_vulkansdk_include_dir "${_vulkan_sdk_root}/Include")
    set(_vulkansdk_lib_dir "${_vulkan_sdk_root}/Lib")
    set(_vulkansdk_bin_dir "${_vulkan_sdk_root}/Bin")
    # Note: vulkan-1.lib is intentionally NOT linked. Volk loads vulkan-1.dll
    # at runtime via LoadLibraryA, so the import library is not needed. Linking
    # it would cause a symbol conflict: vulkan-1.lib provides Vulkan functions
    # as DLL import thunks (read-only code), while volk defines the same names
    # as writable global function pointer variables. When vulkan-1.lib is linked
    # first, volkInitialize() crashes writing to a non-writable code page.
    set(_sdl3_library "${_vulkansdk_lib_dir}/SDL3.lib")
    set(_slangc_name slangc.exe)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_vulkansdk_include_dir "${_vulkan_sdk_root}/include")
    set(_vulkansdk_lib_dir "${_vulkan_sdk_root}/lib")
    set(_vulkansdk_bin_dir "${_vulkan_sdk_root}/bin")
    # Same reasoning as Windows: volk uses dlopen("libvulkan.so.1") at runtime.
    set(_sdl3_library "${_vulkansdk_lib_dir}/libSDL3.so")
    set(_slangc_name slangc)
else()
    message(FATAL_ERROR "Unsupported platform for Vulkan SDK detection: ${CMAKE_SYSTEM_NAME}")
endif()

foreach(_required_dir IN ITEMS "${_vulkansdk_include_dir}" "${_vulkansdk_lib_dir}" "${_vulkansdk_bin_dir}")
    if(NOT EXISTS "${_required_dir}")
        message(FATAL_ERROR "Required Vulkan SDK path does not exist: ${_required_dir}")
    endif()
endforeach()

foreach(_required_header IN ITEMS
    "${_vulkansdk_include_dir}/glm/glm.hpp"
    "${_vulkansdk_include_dir}/vma/vk_mem_alloc.h"
    "${_vulkansdk_include_dir}/SDL3/SDL.h"
)
    if(NOT EXISTS "${_required_header}")
        message(FATAL_ERROR "Required Vulkan SDK header is missing: ${_required_header}")
    endif()
endforeach()

if(NOT EXISTS "${_sdl3_library}")
    message(FATAL_ERROR "Required Vulkan SDK library is missing: ${_sdl3_library}")
endif()

find_program(SLANGC_EXECUTABLE
    NAMES ${_slangc_name}
    PATHS "${_vulkansdk_bin_dir}"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT EXISTS "${SLANGC_EXECUTABLE}")
    message(FATAL_ERROR "slangc executable was not found in ${_vulkansdk_bin_dir}")
endif()

set(VULKANSDK_INCLUDE_DIR  "${_vulkansdk_include_dir}"          CACHE PATH     "Vulkan SDK include directory" FORCE)
set(VULKANSDK_BIN_DIR      "${_vulkansdk_bin_dir}"              CACHE PATH     "Vulkan SDK bin directory"     FORCE)
set(VULKANSDK_LIBRARIES    "${_sdl3_library}"                   CACHE STRING   "Vulkan SDK libraries"         FORCE)
set(SLANGC_EXECUTABLE      "${SLANGC_EXECUTABLE}"                CACHE FILEPATH "slangc executable"            FORCE)
set(VULKANSDK_VOLK_SOURCE  "${_vulkansdk_include_dir}/volk/volk.c" CACHE FILEPATH "Volk implementation source" FORCE)

if(NOT EXISTS "${VULKANSDK_VOLK_SOURCE}")
    message(FATAL_ERROR "volk.c not found in Vulkan SDK: ${VULKANSDK_VOLK_SOURCE}")
endif()

mark_as_advanced(VULKANSDK_INCLUDE_DIR VULKANSDK_LIBRARIES SLANGC_EXECUTABLE VULKANSDK_VOLK_SOURCE)

message(STATUS "Using Vulkan SDK from: ${_vulkan_sdk_root}")
message(STATUS "Using slangc: ${SLANGC_EXECUTABLE}")
