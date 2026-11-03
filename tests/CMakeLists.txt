include(GoogleTest)

# ---------------------------------------------------------------------------
# Module tests — GPU scene build workflows (TLAS + meshlet/light buffers)
# ---------------------------------------------------------------------------
set(THEIA_MODULE_TEST_SOURCES
    module/main.cpp
    module/test_scene_build.cpp
    module/test_rng_contract.cpp
    module/test_camera_jitter_contract.cpp
    module/test_env_sampling_contract.cpp
    module/test_bsdf_sampling_contract.cpp
    module/test_view_accumulation_contract.cpp
    module/test_cli_flags.cpp
    module/test_pass_timer.cpp
    module/test_sms_scene.cpp
)

add_executable(theia_module_tests ${THEIA_MODULE_TEST_SOURCES})
add_dependencies(theia_module_tests theia_shaders)
set(COMPILE_SLANG_SHADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(COMPILE_SLANG_EXTRA_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/shaders" "${HARMONIA_SHADER_SOURCE_DIR}")
compile_slang_shaders(theia_sms_test_shaders "${CMAKE_CURRENT_BINARY_DIR}/shaders" sms_scene_test.slang)
unset(COMPILE_SLANG_SHADER_ROOT)
unset(COMPILE_SLANG_EXTRA_INCLUDE_DIRS)
add_dependencies(theia_module_tests theia_sms_test_shaders)
target_compile_definitions(theia_module_tests PRIVATE THEIA_TEST_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
target_link_libraries(theia_module_tests
    PRIVATE
        theia_demo
        GTest::gtest
)
target_include_directories(theia_module_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${harmonia_SOURCE_DIR}/tests"
)
gtest_discover_tests(theia_module_tests PROPERTIES RUN_SERIAL TRUE)
