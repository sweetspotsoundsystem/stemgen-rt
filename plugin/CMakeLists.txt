cmake_minimum_required(VERSION 3.22)

# Version is needed by JUCE.
project(AudioPlugin VERSION 0.3.0)

# Adding a directory with the library/application name as a subfolder of the
# include folder is a good practice. It helps avoid name clashes later on.
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/StemgenRT")

# JUCE's macOS module-info helper can remain blocked in dyld while loading the
# freshly linked bundle (before any plugin code runs). The VST3 manifest is
# optional, so keep macOS builds deterministic; Windows retains the normal
# auto-generated manifest.
set(STEMGENRT_VST3_AUTO_MANIFEST TRUE)
if(APPLE)
  set(STEMGENRT_VST3_AUTO_MANIFEST FALSE)
endif()

# Adds a plugin target (that's basically what the Projucer does).
juce_add_plugin(
  ${PROJECT_NAME}
  COMPANY_NAME
  "Sweet Spot"
  IS_SYNTH
  FALSE
  NEEDS_MIDI_INPUT
  FALSE
  NEEDS_MIDI_OUTPUT
  FALSE
  VST3_AUTO_MANIFEST
  ${STEMGENRT_VST3_AUTO_MANIFEST}
  PLUGIN_MANUFACTURER_CODE
  Swee
  PLUGIN_CODE
  Stem
  FORMATS
  Standalone
  VST3
  AU
  PRODUCT_NAME
  "StemgenRT"
  BUNDLE_ID
  "com.sweetspot.StemgenRT"
)

# Binary data for logo asset
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
if(EXISTS "${ASSETS_DIR}/logo.png")
  juce_add_binary_data(PluginAssets SOURCES "${ASSETS_DIR}/logo.png")
  set(HAS_LOGO_ASSET TRUE)
else()
  message(STATUS "Logo not found at ${ASSETS_DIR}/logo.png - building without logo")
  set(HAS_LOGO_ASSET FALSE)
endif()

# Sets the source files of the plugin project.
set(SOURCE_FILES
  source/PluginEditor.cpp
  source/PluginProcessor.cpp
  source/OnnxRuntime.cpp
  source/InferenceQueue.cpp
  source/StreamingSampleRateAdapter.cpp
  source/OverlapAddProcessor.cpp
  source/OutputWriter.cpp
)
# Optional; includes header files in the project file tree in Visual Studio
set(HEADER_FILES
  ${INCLUDE_DIR}/PluginEditor.h
  ${INCLUDE_DIR}/PluginProcessor.h
  ${INCLUDE_DIR}/Constants.h
  ${INCLUDE_DIR}/OnnxRuntime.h
  ${INCLUDE_DIR}/InferenceQueue.h
  ${INCLUDE_DIR}/StreamingSampleRateAdapter.h
  ${INCLUDE_DIR}/ModelOutputScheduler.h
  ${INCLUDE_DIR}/OverlapAddProcessor.h
  ${INCLUDE_DIR}/OutputWriter.h
)
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES} ${HEADER_FILES})

# Sets the include directories of the plugin project.
target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STEMGENRT_GENERATED_INCLUDE_DIR})

# Links to all necessary dependencies. The present ones are recommended by JUCE.
# If you use one of the additional modules, like the DSP module, you need to specify it here.
target_link_libraries_system(
  ${PROJECT_NAME} PUBLIC juce::juce_audio_utils juce::juce_cryptography juce::juce_dsp
)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags
                         juce::juce_recommended_warning_flags
)

# These definitions are recommended by JUCE.
target_compile_definitions(${PROJECT_NAME} PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)

# Debug UI overlay (status/latency/underrun counters).
#
# Note: In this repo it's common to configure a dev build without setting
# CMAKE_BUILD_TYPE. In that case, JUCE's recommended config flags still define
# NDEBUG (because the config isn't "Debug"), so we key the debug UI off our own
# macro instead.
target_compile_definitions(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Debug>:STEMGENRT_DEBUG_UI=1>")
if(NOT CMAKE_CONFIGURATION_TYPES AND (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL ""))
  target_compile_definitions(${PROJECT_NAME} PRIVATE STEMGENRT_DEBUG_UI=1)
endif()

# Link logo binary data if available
if(HAS_LOGO_ASSET)
  target_link_libraries(${PROJECT_NAME} PRIVATE PluginAssets)
  target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_LOGO_ASSET=1)
endif()

# Optional: ONNX Runtime integration
# Prefers the official ONNX Runtime release (self-contained, no external dependencies)
# over Homebrew (which has dependencies like protobuf that break distribution).
#
# To set up: ./scripts/download-onnxruntime.sh

set(_ORT_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libs/onnxruntime")
set(_ORT_FOUND FALSE)
set(_ORT_LOCAL_VERSION "")

if(EXISTS "${_ORT_LOCAL_DIR}/VERSION_NUMBER")
  file(STRINGS "${_ORT_LOCAL_DIR}/VERSION_NUMBER" _ORT_LOCAL_VERSION LIMIT_COUNT 1)
  string(STRIP "${_ORT_LOCAL_VERSION}" _ORT_LOCAL_VERSION)
endif()

# Priority 1: Check for local official release (recommended for distribution)
if(EXISTS "${_ORT_LOCAL_DIR}/lib"
   AND EXISTS "${_ORT_LOCAL_DIR}/include/onnxruntime_c_api.h")
  message(STATUS "Using local ONNX Runtime from: ${_ORT_LOCAL_DIR}")

  # Find the library (platform-specific names)
  if(WIN32)
    find_library(ONNXRUNTIME_LIBRARY
      NAMES onnxruntime
      PATHS "${_ORT_LOCAL_DIR}/lib"
      NO_DEFAULT_PATH
    )
    # Also find the DLL for runtime
    find_file(ONNXRUNTIME_DLL
      NAMES onnxruntime.dll
      PATHS "${_ORT_LOCAL_DIR}/lib"
      NO_DEFAULT_PATH
    )
  else()
    find_library(ONNXRUNTIME_LIBRARY
      NAMES onnxruntime
      PATHS "${_ORT_LOCAL_DIR}/lib"
      NO_DEFAULT_PATH
    )
  endif()
  
  set(ONNXRUNTIME_INCLUDE_DIR "${_ORT_LOCAL_DIR}/include")

  # A Windows deployment needs the SDK import library as an integrity/layout
  # check as well as the sibling DLL that the plugin loads explicitly.
  if(ONNXRUNTIME_LIBRARY AND EXISTS "${ONNXRUNTIME_LIBRARY}"
     AND (NOT WIN32
          OR (ONNXRUNTIME_DLL AND EXISTS "${ONNXRUNTIME_DLL}")))
    set(_ORT_FOUND TRUE)
    set(_ORT_SOURCE "local")
    message(STATUS
      "ONNX Runtime (local ${_ORT_LOCAL_VERSION}): lib='${ONNXRUNTIME_LIBRARY}', include='${ONNXRUNTIME_INCLUDE_DIR}'")
    if(WIN32)
      message(STATUS "ONNX Runtime DLL: '${ONNXRUNTIME_DLL}'")
    endif()
  elseif(WIN32)
    message(WARNING
      "Incomplete local ONNX Runtime Windows SDK at ${_ORT_LOCAL_DIR}; both onnxruntime.lib and onnxruntime.dll are required")
  endif()
endif()

if(_ORT_FOUND AND _ORT_SOURCE STREQUAL "local"
   AND NOT _ORT_LOCAL_VERSION STREQUAL STEMGENRT_QUALIFIED_ORT_VERSION)
  if(STEMGENRT_REQUIRE_QUALIFIED_ORT)
    message(FATAL_ERROR
      "Qualified packaging requires ONNX Runtime ${STEMGENRT_QUALIFIED_ORT_VERSION}, but ${_ORT_LOCAL_DIR}/VERSION_NUMBER reports '${_ORT_LOCAL_VERSION}'")
  else()
    message(WARNING
      "Using unqualified ONNX Runtime '${_ORT_LOCAL_VERSION}'. Release/CI artifacts require ${STEMGENRT_QUALIFIED_ORT_VERSION} with -DSTEMGENRT_REQUIRE_QUALIFIED_ORT=ON")
  endif()
endif()

# Priority 2: Fall back to Homebrew/system (with warning about distribution)
if(NOT _ORT_FOUND)
  find_package(onnxruntime CONFIG QUIET
    HINTS
      /opt/homebrew/opt/onnxruntime
      /usr/local/opt/onnxruntime
      $ENV{ONNXRUNTIME_ROOT}
      $ENV{ONNXRUNTIME_DIR}
  )
  
  if(onnxruntime_FOUND)
    message(WARNING "Using Homebrew ONNX Runtime - plugin will NOT be distributable to other Macs!")
    message(WARNING "For distribution, run: ./scripts/download-onnxruntime.sh")
    target_link_libraries(${PROJECT_NAME} PUBLIC onnxruntime::onnxruntime)
    target_compile_definitions(${PROJECT_NAME} PUBLIC STEMGENRT_USE_ONNXRUNTIME=1)
    
    if(APPLE)
      get_target_property(_ort_imported_loc onnxruntime::onnxruntime LOCATION)
      if(_ort_imported_loc)
        get_filename_component(_ort_lib_dir "${_ort_imported_loc}" DIRECTORY)
        get_filename_component(_ort_basename "${_ort_imported_loc}" NAME)
        foreach(_fmt_target IN ITEMS ${PROJECT_NAME}_Standalone ${PROJECT_NAME}_VST3 ${PROJECT_NAME}_AU ${PROJECT_NAME}_AAX)
          if(TARGET ${_fmt_target})
            set_property(TARGET ${_fmt_target} APPEND PROPERTY BUILD_RPATH "${_ort_lib_dir}")
            set_property(TARGET ${_fmt_target} APPEND PROPERTY BUILD_RPATH "@loader_path/../Frameworks")
            add_custom_command(TARGET ${_fmt_target} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks"
              COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_ort_imported_loc}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks/${_ort_basename}"
              COMMAND install_name_tool -id "@rpath/${_ort_basename}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks/${_ort_basename}"
              COMMAND install_name_tool -change "${_ort_imported_loc}" "@rpath/${_ort_basename}" "$<TARGET_FILE:${_fmt_target}>"
            )
          endif()
        endforeach()
      endif()
    endif()
    set(_ORT_FOUND TRUE)
    set(_ORT_SOURCE "homebrew")
  endif()
endif()

if(STEMGENRT_REQUIRE_QUALIFIED_ORT)
  if(NOT _ORT_FOUND)
    message(FATAL_ERROR
      "Qualified packaging requires the official ONNX Runtime ${STEMGENRT_QUALIFIED_ORT_VERSION} SDK at ${_ORT_LOCAL_DIR}. Run the platform download script first.")
  endif()
  if(NOT _ORT_SOURCE STREQUAL "local")
    message(FATAL_ERROR
      "Qualified packaging requires the self-contained official ONNX Runtime SDK; system/Homebrew packages are not accepted")
  endif()
  if(NOT _ORT_LOCAL_VERSION STREQUAL STEMGENRT_QUALIFIED_ORT_VERSION)
    message(FATAL_ERROR
      "Qualified packaging requires ONNX Runtime ${STEMGENRT_QUALIFIED_ORT_VERSION}, got '${_ORT_LOCAL_VERSION}'")
  endif()
  if(WIN32 AND (NOT EXISTS "${ONNXRUNTIME_LIBRARY}"
                OR NOT EXISTS "${ONNXRUNTIME_DLL}"))
    message(FATAL_ERROR
      "Qualified Windows packaging requires both onnxruntime.lib and onnxruntime.dll from the official SDK")
  endif()
  if(APPLE)
    if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "14.0")
      message(FATAL_ERROR
        "Qualified StemgenRT macOS artifacts require CMAKE_OSX_DEPLOYMENT_TARGET >= 14.0 because the official ONNX Runtime ${STEMGENRT_QUALIFIED_ORT_VERSION} dylib requires macOS 14.0")
    endif()
    if(CMAKE_OSX_ARCHITECTURES)
      if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        message(FATAL_ERROR
          "The official ONNX Runtime ${STEMGENRT_QUALIFIED_ORT_VERSION} macOS archive is arm64-only; qualified artifacts must set CMAKE_OSX_ARCHITECTURES=arm64")
      endif()
    elseif(NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
      message(FATAL_ERROR
        "Qualified macOS artifacts require Apple Silicon or an explicit -DCMAKE_OSX_ARCHITECTURES=arm64 cross-build")
    endif()
  endif()
endif()

# Configure local ONNX Runtime if found
if(_ORT_FOUND AND _ORT_SOURCE STREQUAL "local")
  target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_compile_definitions(${PROJECT_NAME} PUBLIC STEMGENRT_USE_ONNXRUNTIME=1)
  if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_LIBRARY})
  endif()
  
  if(APPLE)
    # Always package one canonical filename. The official dylib's source
    # install ID is versioned, so the sealing step rewrites both that ID and
    # each bundle executable's matching load command before signing.
    get_filename_component(_ort_realpath "${ONNXRUNTIME_LIBRARY}" REALPATH)
    set(_ort_basename "libonnxruntime.dylib")
    
    foreach(_fmt_target IN ITEMS ${PROJECT_NAME}_Standalone ${PROJECT_NAME}_VST3 ${PROJECT_NAME}_AU ${PROJECT_NAME}_AAX)
      if(TARGET ${_fmt_target})
        # Set rpath to find bundled framework
        set_property(TARGET ${_fmt_target} APPEND PROPERTY BUILD_RPATH "@loader_path/../Frameworks")
        
        # Bundle the dylib and fix install names
        add_custom_command(TARGET ${_fmt_target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_ort_realpath}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks/${_ort_basename}"
          COMMENT "Bundling ONNX Runtime into ${_fmt_target}"
        )
      endif()
    endforeach()
  elseif(WIN32)
    # On Windows the wrapper resolves OrtGetApiBase directly from this exact
    # bundled DLL path, avoiding collisions with another ORT already loaded by
    # the host or a different plugin.
    if(ONNXRUNTIME_DLL)
      foreach(_fmt_target IN ITEMS ${PROJECT_NAME}_Standalone ${PROJECT_NAME}_VST3 ${PROJECT_NAME}_AAX)
        if(TARGET ${_fmt_target})
          add_custom_command(TARGET ${_fmt_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:${_fmt_target}>"
            COMMENT "Bundling ONNX Runtime DLL into ${_fmt_target}"
          )
        endif()
      endforeach()
    endif()
    
  endif()
endif()

if(NOT _ORT_FOUND)
  message(STATUS "ONNX Runtime not found. Building without ORT support.")
  message(STATUS "To enable: ./scripts/download-onnxruntime.sh")
endif()

# Bundle the immutable, single-file ONNX model into each plugin's Resources folder.
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model")
set(MODEL_ONNX "${MODEL_DIR}/model.onnx")

if(NOT EXISTS "${MODEL_ONNX}")
  message(FATAL_ERROR "Required ONNX model not found: ${MODEL_ONNX}")
endif()

# Re-run configuration if the model is replaced so the immutable artifact
# checks cannot be bypassed by an incremental build.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${MODEL_ONNX}")

file(SIZE "${MODEL_ONNX}" MODEL_ONNX_SIZE)
if(NOT "${MODEL_ONNX_SIZE}" STREQUAL
       "${STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE}")
  message(FATAL_ERROR
    "Unexpected ONNX model size at ${MODEL_ONNX}: expected ${STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE} bytes, got ${MODEL_ONNX_SIZE} bytes")
endif()

file(SHA256 "${MODEL_ONNX}" MODEL_ONNX_SHA256)
if(NOT "${MODEL_ONNX_SHA256}" STREQUAL
       "${STEMGENRT_QUALIFIED_MODEL_SHA256}")
  message(FATAL_ERROR
    "Unexpected ONNX model SHA-256 at ${MODEL_ONNX}: expected ${STEMGENRT_QUALIFIED_MODEL_SHA256}, got ${MODEL_ONNX_SHA256}")
endif()

message(STATUS
  "Bundling verified ONNX model: ${MODEL_ONNX} (${MODEL_ONNX_SIZE} bytes, SHA-256 ${MODEL_ONNX_SHA256})")

# A TARGET POST_BUILD command only runs when the binary is rebuilt. These
# per-format ALL targets also refresh the model after a model-only change.
if(APPLE)
  set(STEMGENRT_MACOS_SIGN_IDENTITY "-" CACHE STRING
      "codesign identity used after bundle resource mutation ('-' for ad-hoc)")
endif()

set(_stemgenrt_bundle_model_targets "")
set(_stemgenrt_apple_format_targets "")
foreach(_fmt_target IN ITEMS ${PROJECT_NAME}_Standalone ${PROJECT_NAME}_VST3 ${PROJECT_NAME}_AU ${PROJECT_NAME}_AAX)
  if(TARGET ${_fmt_target})
    set(_bundle_model_target "${_fmt_target}_BundleModel")
    if(APPLE)
      set(_bundle_ort_refresh_commands "")
      if(_ORT_FOUND AND _ORT_SOURCE STREQUAL "local")
        list(APPEND _bundle_ort_refresh_commands
          COMMAND ${CMAKE_COMMAND} -E make_directory
                  "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  "${_ort_realpath}"
                  "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks/libonnxruntime.dylib")
      endif()
      add_custom_target(${_bundle_model_target} ALL
        ${_bundle_ort_refresh_commands}
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources"
        COMMAND ${CMAKE_COMMAND} -E rm -f "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources/model.onnx.data"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MODEL_ONNX}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources/model.onnx"
        COMMAND ${CMAKE_COMMAND}
          "-DSTEMGENRT_BUNDLE_PATH=$<TARGET_BUNDLE_DIR:${_fmt_target}>"
          "-DSTEMGENRT_SIGN_IDENTITY=${STEMGENRT_MACOS_SIGN_IDENTITY}"
          -P "${CMAKE_SOURCE_DIR}/cmake/SignAndVerifyAppleBundle.cmake"
        COMMENT "Bundling the verified model and sealing ${_fmt_target}"
        VERBATIM
      )
      list(APPEND _stemgenrt_apple_format_targets ${_fmt_target})
    else()
      add_custom_target(${_bundle_model_target} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources"
        COMMAND ${CMAKE_COMMAND} -E rm -f "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources/model.onnx.data"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MODEL_ONNX}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources/model.onnx"
        COMMENT "Bundling verified ONNX model into ${_fmt_target}"
        VERBATIM
      )
    endif()
    add_dependencies(${_bundle_model_target} ${_fmt_target})
    list(APPEND _stemgenrt_bundle_model_targets ${_bundle_model_target})
  endif()
endforeach()

if(APPLE)
  add_custom_target(StemgenRT_VerifyMacBundles
    COMMENT "Strictly verifying packaged StemgenRT Apple bundles")
  foreach(_bundle_model_target IN LISTS _stemgenrt_bundle_model_targets)
    add_dependencies(StemgenRT_VerifyMacBundles ${_bundle_model_target})
  endforeach()
  foreach(_fmt_target IN LISTS _stemgenrt_apple_format_targets)
    add_custom_command(TARGET StemgenRT_VerifyMacBundles POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        "-DSTEMGENRT_BUNDLE_PATH=$<TARGET_BUNDLE_DIR:${_fmt_target}>"
        -DSTEMGENRT_VERIFY_ONLY=ON
        -P "${CMAKE_SOURCE_DIR}/cmake/SignAndVerifyAppleBundle.cmake"
      VERBATIM
    )
  endforeach()
endif()

# Enables strict C++ warnings and treats warnings as errors.
# This needs to be set up only for your projects, not 3rd party
set_source_files_properties(${SOURCE_FILES} PROPERTIES COMPILE_OPTIONS "${PROJECT_WARNINGS_CXX}")

# In Visual Studio this command provides a nice grouping of source files in "filters".
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/.. FILES ${SOURCE_FILES} ${HEADER_FILES})
