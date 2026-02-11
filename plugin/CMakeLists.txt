cmake_minimum_required(VERSION 3.22)

# Version is needed by JUCE.
project(AudioPlugin VERSION 0.1.0)

# Adding a directory with the library/application name as a subfolder of the
# include folder is a good practice. It helps avoid name clashes later on.
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/StemgenRT")

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
  source/Crossover.cpp
  source/VocalsGate.cpp
  source/InputNormalizer.cpp
  source/OnnxRuntime.cpp
  source/InferenceQueue.cpp
  source/OverlapAddProcessor.cpp
  source/OutputWriter.cpp
  source/StemPostProcessor.cpp
  source/LowBandStabilizer.cpp
)
# Optional; includes header files in the project file tree in Visual Studio
set(HEADER_FILES
  ${INCLUDE_DIR}/PluginEditor.h
  ${INCLUDE_DIR}/PluginProcessor.h
  ${INCLUDE_DIR}/Constants.h
  ${INCLUDE_DIR}/Crossover.h
  ${INCLUDE_DIR}/SoftGate.h
  ${INCLUDE_DIR}/VocalsGate.h
  ${INCLUDE_DIR}/InputNormalizer.h
  ${INCLUDE_DIR}/OnnxRuntime.h
  ${INCLUDE_DIR}/InferenceQueue.h
  ${INCLUDE_DIR}/OverlapAddProcessor.h
  ${INCLUDE_DIR}/OutputWriter.h
  ${INCLUDE_DIR}/StemPostProcessor.h
  ${INCLUDE_DIR}/LowBandStabilizer.h
)
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES} ${HEADER_FILES})

# Sets the include directories of the plugin project.
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Links to all necessary dependencies. The present ones are recommended by JUCE.
# If you use one of the additional modules, like the DSP module, you need to specify it here.
target_link_libraries_system(${PROJECT_NAME} PUBLIC juce::juce_audio_utils juce::juce_dsp)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags
                         juce::juce_recommended_warning_flags
)

# These definitions are recommended by JUCE.
target_compile_definitions(${PROJECT_NAME} PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)

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

# Priority 1: Check for local official release (recommended for distribution)
if(EXISTS "${_ORT_LOCAL_DIR}/lib" AND EXISTS "${_ORT_LOCAL_DIR}/include")
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
  
  if(ONNXRUNTIME_LIBRARY)
    set(_ORT_FOUND TRUE)
    set(_ORT_SOURCE "local")
    message(STATUS "ONNX Runtime (local): lib='${ONNXRUNTIME_LIBRARY}', include='${ONNXRUNTIME_INCLUDE_DIR}'")
    if(WIN32 AND ONNXRUNTIME_DLL)
      message(STATUS "ONNX Runtime DLL: '${ONNXRUNTIME_DLL}'")
    endif()
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
    target_compile_definitions(${PROJECT_NAME} PRIVATE STEMGENRT_USE_ONNXRUNTIME=1)
    
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

# Configure local ONNX Runtime if found
if(_ORT_FOUND AND _ORT_SOURCE STREQUAL "local")
  target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_LIBRARY})
  target_compile_definitions(${PROJECT_NAME} PRIVATE STEMGENRT_USE_ONNXRUNTIME=1)
  
  if(APPLE)
    get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    # Get the actual dylib filename (handles symlinks)
    get_filename_component(_ort_realpath "${ONNXRUNTIME_LIBRARY}" REALPATH)
    get_filename_component(_ort_basename "${_ort_realpath}" NAME)
    
    foreach(_fmt_target IN ITEMS ${PROJECT_NAME}_Standalone ${PROJECT_NAME}_VST3 ${PROJECT_NAME}_AU ${PROJECT_NAME}_AAX)
      if(TARGET ${_fmt_target})
        # Set rpath to find bundled framework
        set_property(TARGET ${_fmt_target} APPEND PROPERTY BUILD_RPATH "@loader_path/../Frameworks")
        
        # Bundle the dylib and fix install names
        add_custom_command(TARGET ${_fmt_target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_ort_realpath}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks/${_ort_basename}"
          COMMAND install_name_tool -id "@rpath/${_ort_basename}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Frameworks/${_ort_basename}"
          COMMAND install_name_tool -change "${ONNXRUNTIME_LIBRARY}" "@rpath/${_ort_basename}" "$<TARGET_FILE:${_fmt_target}>" || true
          COMMENT "Bundling ONNX Runtime into ${_fmt_target}"
        )
      endif()
    endforeach()
  elseif(WIN32)
    # On Windows, use delay loading so we can load the bundled DLL before the system one.
    # Without this, Windows finds onnxruntime.dll in System32 before our bundled version.
    # Delay loading defers DLL resolution until the first function call, giving our
    # ensureOrtDllLoaded() function a chance to load the correct DLL first.
    foreach(_fmt_target IN ITEMS ${PROJECT_NAME}_Standalone ${PROJECT_NAME}_VST3 ${PROJECT_NAME}_AAX)
      if(TARGET ${_fmt_target})
        target_link_options(${_fmt_target} PRIVATE "/DELAYLOAD:onnxruntime.dll")
        target_link_libraries(${_fmt_target} PRIVATE delayimp.lib)
      endif()
    endforeach()
    
    # Bundle the DLL next to the plugin executable
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
    
    # Check for GPU build (marker file created by download script)
    # If GPU build, also copy CUDA provider DLLs
    if(EXISTS "${_ORT_LOCAL_DIR}/.gpu_build")
      message(STATUS "ONNX Runtime GPU build detected - will bundle CUDA provider DLLs")
      
      # Find all provider-related DLLs (onnxruntime_providers_*.dll)
      file(GLOB ONNXRUNTIME_PROVIDER_DLLS "${_ORT_LOCAL_DIR}/lib/onnxruntime_providers_*.dll")
      
      foreach(_fmt_target IN ITEMS ${PROJECT_NAME}_Standalone ${PROJECT_NAME}_VST3 ${PROJECT_NAME}_AAX)
        if(TARGET ${_fmt_target})
          foreach(_provider_dll IN LISTS ONNXRUNTIME_PROVIDER_DLLS)
            get_filename_component(_dll_name "${_provider_dll}" NAME)
            add_custom_command(TARGET ${_fmt_target} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_provider_dll}" "$<TARGET_FILE_DIR:${_fmt_target}>"
              COMMENT "Bundling ${_dll_name} into ${_fmt_target}"
            )
          endforeach()
        endif()
      endforeach()
    endif()
  endif()
endif()

if(NOT _ORT_FOUND)
  message(STATUS "ONNX Runtime not found. Building without ORT support.")
  message(STATUS "To enable: ./scripts/download-onnxruntime.sh")
endif()

# Bundle ONNX model files into plugin Resources folder
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model")
set(MODEL_ONNX "${MODEL_DIR}/model.onnx")
set(MODEL_DATA "${MODEL_DIR}/model.onnx.data")

# Check if model files exist
if(EXISTS "${MODEL_ONNX}" AND EXISTS "${MODEL_DATA}")
  message(STATUS "Bundling ONNX model from: ${MODEL_DIR}")
  foreach(_fmt_target IN ITEMS ${PROJECT_NAME}_Standalone ${PROJECT_NAME}_VST3 ${PROJECT_NAME}_AU ${PROJECT_NAME}_AAX)
    if(TARGET ${_fmt_target})
      add_custom_command(TARGET ${_fmt_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MODEL_ONNX}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources/model.onnx"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MODEL_DATA}" "$<TARGET_FILE_DIR:${_fmt_target}>/../Resources/model.onnx.data"
        COMMENT "Bundling ONNX model into ${_fmt_target}"
      )
    endif()
  endforeach()
else()
  message(WARNING "ONNX model files not found at ${MODEL_DIR}. Plugin will not include bundled model.")
endif()

# Enables strict C++ warnings and treats warnings as errors.
# This needs to be set up only for your projects, not 3rd party
set_source_files_properties(${SOURCE_FILES} PROPERTIES COMPILE_OPTIONS "${PROJECT_WARNINGS_CXX}")

# In Visual Studio this command provides a nice grouping of source files in "filters".
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/..)
