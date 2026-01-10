cmake_minimum_required(VERSION 3.22)

project(AudioPluginTest)

# Signals to CMake that we want to run tests from this directory.
enable_testing()

# Creates the test console application.
set(SOURCE_FILES source/AudioProcessorTest.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Sets the necessary include directories of googletest.
target_include_directories(${PROJECT_NAME} PRIVATE ${GOOGLETEST_SOURCE_DIR}/googletest/include)

# Thanks to the fact that we link against the gtest_main library, we don't have to write the main function ourselves.
target_link_libraries(${PROJECT_NAME} PRIVATE AudioPlugin GTest::gtest_main)

# Enables strict C++ warnings and treats warnings as errors.
# This needs to be set up only for your projects, not 3rd party
set_source_files_properties(${SOURCE_FILES} PROPERTIES COMPILE_OPTIONS "${PROJECT_WARNINGS_CXX}")

# Adds googletest-specific CMake commands at our disposal.
include(GoogleTest)

# On Windows, copy ONNX Runtime DLL to test executable directory
if(WIN32)
  set(_ORT_LOCAL_DIR "${CMAKE_SOURCE_DIR}/libs/onnxruntime")
  find_file(ONNXRUNTIME_TEST_DLL
    NAMES onnxruntime.dll
    PATHS "${_ORT_LOCAL_DIR}/lib"
    NO_DEFAULT_PATH
  )
  if(ONNXRUNTIME_TEST_DLL)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_TEST_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
      COMMENT "Bundling ONNX Runtime DLL for tests"
    )
  endif()
endif()

# Create symlinks/copies to model files so tests can load the ONNX model.
# The test binary looks for the model at ${CMAKE_BINARY_DIR}/Resources/model.onnx
# (relative to the test executable's grandparent directory).
set(MODEL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/model")
set(MODEL_DEST_DIR "${CMAKE_BINARY_DIR}/Resources")

if(EXISTS "${MODEL_SOURCE_DIR}/model.onnx")
  # Create Resources directory and copy/symlink after build
  # Windows doesn't support symlinks without admin privileges, so use copy there
  if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${MODEL_DEST_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MODEL_SOURCE_DIR}/model.onnx" 
        "${MODEL_DEST_DIR}/model.onnx"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MODEL_SOURCE_DIR}/model.onnx.data" 
        "${MODEL_DEST_DIR}/model.onnx.data"
      COMMENT "Copying ONNX model for tests"
    )
  else()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${MODEL_DEST_DIR}"
      COMMAND ${CMAKE_COMMAND} -E rm -f "${MODEL_DEST_DIR}/model.onnx"
      COMMAND ${CMAKE_COMMAND} -E create_symlink 
        "${MODEL_SOURCE_DIR}/model.onnx" 
        "${MODEL_DEST_DIR}/model.onnx"
      COMMAND ${CMAKE_COMMAND} -E rm -f "${MODEL_DEST_DIR}/model.onnx.data"
      COMMAND ${CMAKE_COMMAND} -E create_symlink 
        "${MODEL_SOURCE_DIR}/model.onnx.data" 
        "${MODEL_DEST_DIR}/model.onnx.data"
      COMMENT "Creating symlinks to ONNX model for tests"
    )
  endif()
endif()

# Add all tests defined with googletest to the CMake metadata
# so that these tests are run upon a call to ctest in the test
# projects' binary directory.
if(CMAKE_GENERATOR STREQUAL Xcode)
  # On macOS arm64, all binaries have to be signed before running.
  # In local development, the linker adds an ad-hoc placeholder signature.
  # In Xcode however, the ad-hoc signature is delayed until after the “Run Script” build phase,
  # so the POST_BUILD command added by gtest_discover_tests cannot run.
  # Thus, we need to delay test discovery until run time.
  # Source: https://discourse.cmake.org/t/googletest-crash-when-using-cmake-xcode-arm64/5766/8
  gtest_discover_tests(${PROJECT_NAME} DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(${PROJECT_NAME})
endif()
