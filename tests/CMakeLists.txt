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
)

add_executable(theia_module_tests ${THEIA_MODULE_TEST_SOURCES})
add_dependencies(theia_module_tests theia_shaders)
target_link_libraries(theia_module_tests
    PRIVATE
        theia_demo
        GTest::gtest
)
target_include_directories(theia_module_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${harmonia_SOURCE_DIR}/tests"
)
gtest_discover_tests(theia_module_tests)
