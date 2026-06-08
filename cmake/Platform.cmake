include_guard(GLOBAL)

if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Theia supports only Windows and Linux.")
endif()

function(theia_configure_target target)
    target_compile_features(${target} INTERFACE cxx_std_23)

    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    if(MSVC)
        target_compile_options(${target} INTERFACE
            /W4
            /WX
            /permissive-
            /Zc:__cplusplus
            # /Zc:preprocessor is MSVC-only; clang-cl (COMPILER_ID=Clang, simulates MSVC)
            # does not support it and will error with -Wunused-command-line-argument.
            $<$<NOT:$<CXX_COMPILER_ID:Clang>>:/Zc:preprocessor>
            /wd4324
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} INTERFACE
            -Wall
            -Wextra
            -Werror
            -Wpedantic
            -Wno-unused-parameter
        )
    else()
        message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()

    if(WIN32)
        target_compile_definitions(${target} INTERFACE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            THEIA_PLATFORM_WINDOWS=1
        )
    else()
        target_compile_definitions(${target} INTERFACE THEIA_PLATFORM_LINUX=1)
    endif()
endfunction()
