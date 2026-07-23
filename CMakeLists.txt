# Mandatory line, sets the minimum version of CMake that should be used with this repository.
# I specified 3.22 because I trust it. However, currently I have 3.26 installed on my machine.
# To verify your version run
# $ cmake --version
cmake_minimum_required(VERSION 3.22)

# The official ONNX Runtime 1.26.0 arm64 dylib has LC_BUILD_VERSION minos 14.0.
# Keep the entire product at that baseline so the plugin never advertises a
# deployment target lower than one of its required embedded libraries. Also
# migrate build trees that cached the former 13.3 default.
if(APPLE AND
   (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR
    CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "" OR
    CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "14.0"))
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING
      "Minimum macOS version supported by StemgenRT" FORCE)
endif()

set(STEMGENRT_QUALIFIED_ORT_VERSION "1.26.0")
option(STEMGENRT_REQUIRE_QUALIFIED_ORT
       "Require the complete official ONNX Runtime ${STEMGENRT_QUALIFIED_ORT_VERSION} deployment bundle"
       OFF)

# Sets a few variables, like PROJECT_NAME
project(WolfSoundAudioPluginTemplate)

# Load the single qualified-model contract and generate the C++ view consumed
# by the runtime and tests. Packaging validation includes this same CMake file.
include(cmake/QualifiedModelContract.cmake)

foreach(_stemgenrt_contract_variable IN ITEMS
    STEMGENRT_QUALIFIED_CONTRACT_ID
    STEMGENRT_QUALIFIED_MODEL_SHA256
    STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE
    STEMGENRT_QUALIFIED_CHECKPOINT_SHA256
    STEMGENRT_QUALIFIED_SAMPLE_RATE
    STEMGENRT_QUALIFIED_CHANNELS
    STEMGENRT_QUALIFIED_STEMS
    STEMGENRT_QUALIFIED_HOP_SAMPLES
    STEMGENRT_QUALIFIED_ANALYSIS_WINDOW_SAMPLES
    STEMGENRT_QUALIFIED_FUSION_HIDDEN_LAYERS
    STEMGENRT_QUALIFIED_FUSION_HIDDEN_SIZE
    STEMGENRT_QUALIFIED_MODEL_OUTPUT_DELAY_CHUNKS
    STEMGENRT_QUALIFIED_INPUT_AUDIO_NAME
    STEMGENRT_QUALIFIED_INPUT_PAST_NAME
    STEMGENRT_QUALIFIED_INPUT_HIDDEN_NAME
    STEMGENRT_QUALIFIED_OUTPUT_SEPARATED_NAME
    STEMGENRT_QUALIFIED_OUTPUT_PAST_NAME
    STEMGENRT_QUALIFIED_OUTPUT_HIDDEN_NAME
    STEMGENRT_QUALIFIED_METADATA_INPUT_NAMES
    STEMGENRT_QUALIFIED_METADATA_OUTPUT_NAMES
    STEMGENRT_QUALIFIED_METADATA_EXPORT_MODE
    STEMGENRT_QUALIFIED_METADATA_MIXTURE_CONSISTENCY
    STEMGENRT_QUALIFIED_METADATA_EXPORT_EXTERNAL_DATA
    STEMGENRT_QUALIFIED_METADATA_CAUSAL_CURRENT_CHUNK
    STEMGENRT_QUALIFIED_METADATA_SOURCE_ORDER
    STEMGENRT_QUALIFIED_METADATA_RESIDUAL_SOURCE_NAME
    STEMGENRT_QUALIFIED_METADATA_EXTERNAL_OVERLAP_ADD
    STEMGENRT_QUALIFIED_METADATA_FUSION_HIDDEN_SHAPE
    STEMGENRT_QUALIFIED_METADATA_INITIAL_STATE
    STEMGENRT_QUALIFIED_METADATA_PREROLL
    STEMGENRT_QUALIFIED_METADATA_FLUSH
    STEMGENRT_QUALIFIED_METADATA_RESET
    STEMGENRT_QUALIFIED_METADATA_DEPLOYMENT_STATUS
    STEMGENRT_QUALIFIED_METADATA_QUALITY_GUARDS
    STEMGENRT_QUALIFIED_DRUMS_SOURCE_INDEX
    STEMGENRT_QUALIFIED_BASS_SOURCE_INDEX
    STEMGENRT_QUALIFIED_VOCALS_SOURCE_INDEX
    STEMGENRT_QUALIFIED_OTHER_SOURCE_INDEX
    STEMGENRT_QUALIFIED_RESIDUAL_SOURCE_INDEX
    STEMGENRT_QUALIFIED_METADATA_EXTERNAL_DATA
    STEMGENRT_QUALIFIED_OUTPUT_ALIGNMENT)
  if(NOT DEFINED ${_stemgenrt_contract_variable} OR
     "${${_stemgenrt_contract_variable}}" STREQUAL "")
    message(FATAL_ERROR
      "Qualified model contract is missing ${_stemgenrt_contract_variable}")
  endif()
endforeach()
if(NOT DEFINED STEMGENRT_QUALIFIED_METADATA_OVERLAP_ADD_BUFFER_SHAPE)
  message(FATAL_ERROR
    "Qualified model contract is missing STEMGENRT_QUALIFIED_METADATA_OVERLAP_ADD_BUFFER_SHAPE")
endif()
if(NOT "${STEMGENRT_QUALIFIED_METADATA_OVERLAP_ADD_BUFFER_SHAPE}" STREQUAL "")
  message(FATAL_ERROR
    "The c126 graph requires an explicitly empty overlap-add buffer shape")
endif()

string(LENGTH "${STEMGENRT_QUALIFIED_MODEL_SHA256}"
       _stemgenrt_model_sha_length)
string(LENGTH "${STEMGENRT_QUALIFIED_CHECKPOINT_SHA256}"
       _stemgenrt_checkpoint_sha_length)
if(NOT _stemgenrt_model_sha_length EQUAL 64 OR
   STEMGENRT_QUALIFIED_MODEL_SHA256 MATCHES "[^0-9a-f]")
  message(FATAL_ERROR "Qualified model SHA-256 must be 64 lowercase hex digits")
endif()
if(NOT _stemgenrt_checkpoint_sha_length EQUAL 64 OR
   STEMGENRT_QUALIFIED_CHECKPOINT_SHA256 MATCHES "[^0-9a-f]")
  message(FATAL_ERROR
    "Qualified checkpoint SHA-256 must be 64 lowercase hex digits")
endif()
if(NOT STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE MATCHES "^[0-9]+$" OR
   STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE LESS_EQUAL 0)
  message(FATAL_ERROR "Qualified model byte size must be positive")
endif()
foreach(_stemgenrt_positive_contract_integer IN ITEMS
    STEMGENRT_QUALIFIED_SAMPLE_RATE
    STEMGENRT_QUALIFIED_CHANNELS
    STEMGENRT_QUALIFIED_STEMS
    STEMGENRT_QUALIFIED_HOP_SAMPLES
    STEMGENRT_QUALIFIED_ANALYSIS_WINDOW_SAMPLES
    STEMGENRT_QUALIFIED_FUSION_HIDDEN_LAYERS
    STEMGENRT_QUALIFIED_FUSION_HIDDEN_SIZE)
  if(NOT ${_stemgenrt_positive_contract_integer} MATCHES "^[0-9]+$" OR
     ${_stemgenrt_positive_contract_integer} LESS_EQUAL 0)
    message(FATAL_ERROR
      "Qualified model contract requires a positive integer for ${_stemgenrt_positive_contract_integer}")
  endif()
endforeach()
if(NOT STEMGENRT_QUALIFIED_MODEL_OUTPUT_DELAY_CHUNKS MATCHES "^[0-9]+$")
  message(FATAL_ERROR
    "Qualified model output delay must be a non-negative integer")
endif()
foreach(_stemgenrt_source_index_variable IN ITEMS
    STEMGENRT_QUALIFIED_DRUMS_SOURCE_INDEX
    STEMGENRT_QUALIFIED_BASS_SOURCE_INDEX
    STEMGENRT_QUALIFIED_VOCALS_SOURCE_INDEX
    STEMGENRT_QUALIFIED_OTHER_SOURCE_INDEX
    STEMGENRT_QUALIFIED_RESIDUAL_SOURCE_INDEX)
  if(NOT ${_stemgenrt_source_index_variable} MATCHES "^[0-9]+$" OR
     ${_stemgenrt_source_index_variable} GREATER_EQUAL
       STEMGENRT_QUALIFIED_STEMS)
    message(FATAL_ERROR
      "${_stemgenrt_source_index_variable} must address an available stem")
  endif()
endforeach()
math(EXPR _stemgenrt_expected_analysis_window
     "2 * ${STEMGENRT_QUALIFIED_HOP_SAMPLES}")
if(NOT STEMGENRT_QUALIFIED_ANALYSIS_WINDOW_SAMPLES EQUAL
       _stemgenrt_expected_analysis_window)
  message(FATAL_ERROR
    "The current native state layout requires a two-hop analysis window")
endif()
if(NOT STEMGENRT_QUALIFIED_OUTPUT_ALIGNMENT STREQUAL
       "current_input_chunk")
  message(FATAL_ERROR
    "The c126 native scheduler requires current_input_chunk alignment")
endif()

set(STEMGENRT_GENERATED_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${STEMGENRT_GENERATED_INCLUDE_DIR}/StemgenRT")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/QualifiedModelContract.h.in"
  "${STEMGENRT_GENERATED_INCLUDE_DIR}/StemgenRT/QualifiedModelContract.h"
  @ONLY)

# Always use the newest C++ standard on green-field projects if possible.
# Currently, JUCE supports only C++ 20 cross-platform.
set(CMAKE_CXX_STANDARD 20)

# I like to download the dependencies to the same folder as the project.
# If you want to install them system wide, set CPM_SOURCE_CACHE with the path to the dependencies
# either as an environment variable or pass it to the cmake script with -DCPM_SOURCE_CACHE=<path>.
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)

# Downloads CPM if not already downloaded. CPM is an easy-to-use package manager nicely integrated with CMake.
include(cmake/cpm.cmake)

# This commands downloads AND configures JUCE. It sets up some variables, like JUCE_SOURCE_DIR.
cpmaddpackage(
  NAME
  JUCE
  GIT_TAG
  8.0.6
  VERSION
  8.0.6
  GITHUB_REPOSITORY
  juce-framework/JUCE
  SOURCE_DIR
  ${LIB_DIR}/juce
)

# Adds googletest.
cpmaddpackage(
  NAME
  GOOGLETEST
  GITHUB_REPOSITORY
  google/googletest
  VERSION
  1.16.0
  SOURCE_DIR
  ${LIB_DIR}/googletest
  OPTIONS
  "INSTALL_GTEST OFF"
  "gtest_force_shared_crt ON"
)

# Add compiler warning utilities
include(cmake/CompilerWarnings.cmake)
include(cmake/Util.cmake)

# Adds all the targets configured in the "plugin" folder.
add_subdirectory(plugin)

# This command allows running tests from the "build" folder (the one where CMake generates the project to).
enable_testing()

# Adds all the targets configured in the "test" folder.
add_subdirectory(test)
