include(GoogleTest)

# ---------------------------------------------------------------------------
# Module tests — GPU scene build workflows (TLAS + meshlet/light buffers)
# ---------------------------------------------------------------------------
set(THEIA_MODULE_TEST_SOURCES
    module/main.cpp
    module/test_scene_build.cpp
    module/test_gpu_cull.cpp
    module/test_scene_anim.cpp
    module/test_gi_tlas_rebuild.cpp
    module/test_gi_stale_path_state.cpp
    module/test_rng_contract.cpp
    module/test_camera_jitter_contract.cpp
    module/test_env_sampling_contract.cpp
    module/test_bsdf_sampling_contract.cpp
    module/test_view_accumulation_contract.cpp
    module/test_cli_flags.cpp
    module/test_pass_timer.cpp
    module/test_sms_scene.cpp
    module/test_sms_gi_frame.cpp
    module/test_sms_resampling.cpp
    module/test_sms_shift.cpp
    module/test_forward_tangent.cpp
    module/test_haze.cpp
    module/test_gi_haze.cpp
    module/test_push_constants.cpp
)

add_executable(theia_module_tests ${THEIA_MODULE_TEST_SOURCES})
add_dependencies(theia_module_tests theia_shaders)
set(COMPILE_SLANG_SHADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(COMPILE_SLANG_EXTRA_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/shaders" "${HARMONIA_SHADER_SOURCE_DIR}")
set(COMPILE_SLANG_EXTRA_DEPENDS "${PROJECT_SOURCE_DIR}/shaders/sms_resampling.slang"
    "${PROJECT_SOURCE_DIR}/shaders/sms_shift.slang"
    # The shader contracts compare the shipped modules with Harmonia's: a change to
    # either side recompiles the tests.
    "${PROJECT_SOURCE_DIR}/shaders/forward_tangent.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/surface.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/haze.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/math.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/bsdf.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/bsdf_shared.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/common.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/gpu_types.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/rng.slang")
compile_slang_shaders(theia_sms_test_shaders "${CMAKE_CURRENT_BINARY_DIR}/shaders" sms_scene_test.slang
    sms_resampling_test.slang sms_shift_test.slang forward_tangent_test.slang haze_test.slang)
unset(COMPILE_SLANG_EXTRA_DEPENDS)
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
