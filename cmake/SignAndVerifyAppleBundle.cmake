cmake_minimum_required(VERSION 3.22)

# Bundle sealing consumes the same qualified identity used to generate the
# runtime C++ contract and to validate the source artifact at configure time.
include("${CMAKE_CURRENT_LIST_DIR}/QualifiedModelContract.cmake")

if(NOT DEFINED STEMGENRT_BUNDLE_PATH OR STEMGENRT_BUNDLE_PATH STREQUAL "")
  message(FATAL_ERROR "STEMGENRT_BUNDLE_PATH is required")
endif()

if(NOT EXISTS "${STEMGENRT_BUNDLE_PATH}")
  message(FATAL_ERROR "Apple bundle does not exist: ${STEMGENRT_BUNDLE_PATH}")
endif()

find_program(STEMGENRT_CODESIGN_EXECUTABLE codesign REQUIRED)
find_program(STEMGENRT_OTOOL_EXECUTABLE otool REQUIRED)

function(_stemgenrt_extract_ort_links linked_libraries output_variable)
  string(REGEX MATCHALL
    "[ \t][^ \t\r\n]*libonnxruntime[^ \t\r\n]*\\.dylib"
    _stemgenrt_raw_ort_links "${linked_libraries}")
  set(_stemgenrt_ort_links "")
  foreach(_stemgenrt_raw_link IN LISTS _stemgenrt_raw_ort_links)
    string(STRIP "${_stemgenrt_raw_link}" _stemgenrt_ort_link)
    list(APPEND _stemgenrt_ort_links "${_stemgenrt_ort_link}")
  endforeach()
  list(REMOVE_DUPLICATES _stemgenrt_ort_links)
  set(${output_variable} "${_stemgenrt_ort_links}" PARENT_SCOPE)
endfunction()

set(_stemgenrt_model_path
    "${STEMGENRT_BUNDLE_PATH}/Contents/Resources/model.onnx")
if(NOT EXISTS "${_stemgenrt_model_path}")
  message(FATAL_ERROR
    "Qualified model is missing from bundle: ${_stemgenrt_model_path}")
endif()
file(SIZE "${_stemgenrt_model_path}" _stemgenrt_model_size)
file(SHA256 "${_stemgenrt_model_path}" _stemgenrt_model_sha256)
if(NOT _stemgenrt_model_size EQUAL
       STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE OR
   NOT _stemgenrt_model_sha256 STREQUAL
       STEMGENRT_QUALIFIED_MODEL_SHA256)
  message(FATAL_ERROR
    "Qualified model identity mismatch in ${STEMGENRT_BUNDLE_PATH}: size=${_stemgenrt_model_size}, SHA-256=${_stemgenrt_model_sha256}")
endif()
if(EXISTS
   "${STEMGENRT_BUNDLE_PATH}/Contents/Resources/model.onnx.data")
  message(FATAL_ERROR
    "Obsolete model.onnx.data must not be packaged in ${STEMGENRT_BUNDLE_PATH}")
endif()

file(GLOB _stemgenrt_bundle_executables LIST_DIRECTORIES FALSE
     "${STEMGENRT_BUNDLE_PATH}/Contents/MacOS/*")
list(LENGTH _stemgenrt_bundle_executables _stemgenrt_executable_count)
if(NOT _stemgenrt_executable_count EQUAL 1)
  message(FATAL_ERROR
    "Expected exactly one bundle executable in ${STEMGENRT_BUNDLE_PATH}, found ${_stemgenrt_executable_count}")
endif()
list(GET _stemgenrt_bundle_executables 0 _stemgenrt_bundle_executable)

set(_stemgenrt_ort_name "libonnxruntime.dylib")
set(_stemgenrt_ort_install_name "@rpath/${_stemgenrt_ort_name}")
set(_stemgenrt_ort_dylib
    "${STEMGENRT_BUNDLE_PATH}/Contents/Frameworks/${_stemgenrt_ort_name}")

if(NOT STEMGENRT_VERIFY_ONLY)
  find_program(STEMGENRT_INSTALL_NAME_TOOL_EXECUTABLE install_name_tool REQUIRED)

  if(NOT EXISTS "${_stemgenrt_ort_dylib}")
    message(FATAL_ERROR
      "Canonical bundled ONNX Runtime is missing: ${_stemgenrt_ort_dylib}")
  endif()

  # SDK changes can leave an older versioned dylib in an incremental bundle.
  # Keep one canonical payload before inspecting or signing it.
  file(GLOB _stemgenrt_existing_ort_dylibs LIST_DIRECTORIES FALSE
       "${STEMGENRT_BUNDLE_PATH}/Contents/Frameworks/libonnxruntime*.dylib")
  foreach(_stemgenrt_existing_ort_dylib IN LISTS _stemgenrt_existing_ort_dylibs)
    if(NOT _stemgenrt_existing_ort_dylib STREQUAL _stemgenrt_ort_dylib)
      file(REMOVE "${_stemgenrt_existing_ort_dylib}")
    endif()
  endforeach()

  execute_process(
    COMMAND "${STEMGENRT_OTOOL_EXECUTABLE}" -L
            "${_stemgenrt_bundle_executable}"
    RESULT_VARIABLE _stemgenrt_otool_result
    OUTPUT_VARIABLE _stemgenrt_linked_libraries
    ERROR_VARIABLE _stemgenrt_otool_error)
  if(NOT _stemgenrt_otool_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to inspect bundle linkage for ${_stemgenrt_bundle_executable}:\n${_stemgenrt_otool_error}")
  endif()
  _stemgenrt_extract_ort_links(
    "${_stemgenrt_linked_libraries}" _stemgenrt_original_ort_links)
  list(LENGTH _stemgenrt_original_ort_links _stemgenrt_original_ort_link_count)
  if(NOT _stemgenrt_original_ort_link_count EQUAL 1)
    message(FATAL_ERROR
      "Expected exactly one ONNX Runtime load command in ${_stemgenrt_bundle_executable}, found ${_stemgenrt_original_ort_link_count}:\n${_stemgenrt_linked_libraries}")
  endif()
  list(GET _stemgenrt_original_ort_links 0 _stemgenrt_original_ort_link)

  if(NOT _stemgenrt_original_ort_link STREQUAL _stemgenrt_ort_install_name)
    execute_process(
      COMMAND "${STEMGENRT_INSTALL_NAME_TOOL_EXECUTABLE}" -change
              "${_stemgenrt_original_ort_link}"
              "${_stemgenrt_ort_install_name}"
              "${_stemgenrt_bundle_executable}"
      RESULT_VARIABLE _stemgenrt_install_name_result
      OUTPUT_VARIABLE _stemgenrt_install_name_output
      ERROR_VARIABLE _stemgenrt_install_name_error)
    if(NOT _stemgenrt_install_name_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to rewrite ONNX Runtime load command in ${_stemgenrt_bundle_executable}:\n${_stemgenrt_install_name_output}${_stemgenrt_install_name_error}")
    endif()
  endif()

  execute_process(
    COMMAND "${STEMGENRT_INSTALL_NAME_TOOL_EXECUTABLE}" -id
            "${_stemgenrt_ort_install_name}" "${_stemgenrt_ort_dylib}"
    RESULT_VARIABLE _stemgenrt_install_name_result
    OUTPUT_VARIABLE _stemgenrt_install_name_output
    ERROR_VARIABLE _stemgenrt_install_name_error)
  if(NOT _stemgenrt_install_name_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to set embedded ONNX Runtime install name for ${_stemgenrt_ort_dylib}:\n${_stemgenrt_install_name_output}${_stemgenrt_install_name_error}")
  endif()
endif()

file(GLOB _stemgenrt_ort_dylibs LIST_DIRECTORIES FALSE
     "${STEMGENRT_BUNDLE_PATH}/Contents/Frameworks/libonnxruntime*.dylib")
list(LENGTH _stemgenrt_ort_dylibs _stemgenrt_ort_dylib_count)
if(NOT _stemgenrt_ort_dylib_count EQUAL 1)
  message(FATAL_ERROR
    "Expected exactly one bundled ONNX Runtime dylib in ${STEMGENRT_BUNDLE_PATH}, found ${_stemgenrt_ort_dylib_count}")
endif()
list(GET _stemgenrt_ort_dylibs 0 _stemgenrt_found_ort_dylib)
if(NOT _stemgenrt_found_ort_dylib STREQUAL _stemgenrt_ort_dylib)
  message(FATAL_ERROR
    "Bundled ONNX Runtime must use the canonical path ${_stemgenrt_ort_dylib}; found ${_stemgenrt_found_ort_dylib}")
endif()

execute_process(
  COMMAND "${STEMGENRT_OTOOL_EXECUTABLE}" -D "${_stemgenrt_ort_dylib}"
  RESULT_VARIABLE _stemgenrt_otool_result
  OUTPUT_VARIABLE _stemgenrt_ort_id_output
  ERROR_VARIABLE _stemgenrt_otool_error)
if(NOT _stemgenrt_otool_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to inspect embedded ONNX Runtime install name: ${_stemgenrt_otool_error}")
endif()
string(STRIP "${_stemgenrt_ort_id_output}" _stemgenrt_ort_id_output)
string(REPLACE "\r\n" "\n" _stemgenrt_ort_id_output
       "${_stemgenrt_ort_id_output}")
string(REPLACE "\n" ";" _stemgenrt_ort_id_lines
       "${_stemgenrt_ort_id_output}")
list(GET _stemgenrt_ort_id_lines -1 _stemgenrt_ort_id)
if(NOT _stemgenrt_ort_id STREQUAL _stemgenrt_ort_install_name)
  message(FATAL_ERROR
    "Embedded ONNX Runtime install name must be ${_stemgenrt_ort_install_name}, found '${_stemgenrt_ort_id}'")
endif()

execute_process(
  COMMAND "${STEMGENRT_OTOOL_EXECUTABLE}" -L
          "${_stemgenrt_bundle_executable}"
  RESULT_VARIABLE _stemgenrt_otool_result
  OUTPUT_VARIABLE _stemgenrt_linked_libraries
  ERROR_VARIABLE _stemgenrt_otool_error)
if(NOT _stemgenrt_otool_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to inspect bundle linkage for ${_stemgenrt_bundle_executable}:\n${_stemgenrt_otool_error}")
endif()
_stemgenrt_extract_ort_links(
  "${_stemgenrt_linked_libraries}" _stemgenrt_sealed_ort_links)
list(LENGTH _stemgenrt_sealed_ort_links _stemgenrt_sealed_ort_link_count)
if(NOT _stemgenrt_sealed_ort_link_count EQUAL 1)
  message(FATAL_ERROR
    "Expected exactly one sealed ONNX Runtime load command in ${_stemgenrt_bundle_executable}, found ${_stemgenrt_sealed_ort_link_count}:\n${_stemgenrt_linked_libraries}")
endif()
list(GET _stemgenrt_sealed_ort_links 0 _stemgenrt_sealed_ort_link)
if(NOT _stemgenrt_sealed_ort_link STREQUAL _stemgenrt_ort_install_name)
  message(FATAL_ERROR
    "Bundle executable must link embedded ONNX Runtime as ${_stemgenrt_ort_install_name}, found '${_stemgenrt_sealed_ort_link}'")
endif()

execute_process(
  COMMAND "${STEMGENRT_OTOOL_EXECUTABLE}" -l
          "${_stemgenrt_bundle_executable}"
  RESULT_VARIABLE _stemgenrt_otool_result
  OUTPUT_VARIABLE _stemgenrt_load_commands
  ERROR_VARIABLE _stemgenrt_otool_error)
if(NOT _stemgenrt_otool_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to inspect bundle load commands for ${_stemgenrt_bundle_executable}:\n${_stemgenrt_otool_error}")
endif()
string(FIND "${_stemgenrt_load_commands}"
       "path @loader_path/../Frameworks" _stemgenrt_rpath_index)
if(_stemgenrt_rpath_index EQUAL -1)
  message(FATAL_ERROR
    "Bundle executable is missing @loader_path/../Frameworks rpath: ${_stemgenrt_bundle_executable}")
endif()

if(NOT STEMGENRT_VERIFY_ONLY)
  if(NOT DEFINED STEMGENRT_SIGN_IDENTITY OR STEMGENRT_SIGN_IDENTITY STREQUAL "")
    set(STEMGENRT_SIGN_IDENTITY "-")
  endif()

  # Resource copying and install_name_tool both invalidate signatures. Sign
  # embedded code first, then seal the outer bundle after every mutation.
  file(GLOB_RECURSE _stemgenrt_nested_dylibs LIST_DIRECTORIES FALSE
       "${STEMGENRT_BUNDLE_PATH}/Contents/Frameworks/*.dylib")
  foreach(_stemgenrt_dylib IN LISTS _stemgenrt_nested_dylibs)
    execute_process(
      COMMAND "${STEMGENRT_CODESIGN_EXECUTABLE}" --force --sign
              "${STEMGENRT_SIGN_IDENTITY}" "${_stemgenrt_dylib}"
      RESULT_VARIABLE _stemgenrt_sign_result
      OUTPUT_VARIABLE _stemgenrt_sign_output
      ERROR_VARIABLE _stemgenrt_sign_error)
    if(NOT _stemgenrt_sign_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to sign nested library ${_stemgenrt_dylib}:\n${_stemgenrt_sign_output}${_stemgenrt_sign_error}")
    endif()
  endforeach()

  execute_process(
    COMMAND "${STEMGENRT_CODESIGN_EXECUTABLE}" --force --sign
            "${STEMGENRT_SIGN_IDENTITY}" "${STEMGENRT_BUNDLE_PATH}"
    RESULT_VARIABLE _stemgenrt_sign_result
    OUTPUT_VARIABLE _stemgenrt_sign_output
    ERROR_VARIABLE _stemgenrt_sign_error)
  if(NOT _stemgenrt_sign_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to sign bundle ${STEMGENRT_BUNDLE_PATH}:\n${_stemgenrt_sign_output}${_stemgenrt_sign_error}")
  endif()
endif()

execute_process(
  COMMAND "${STEMGENRT_CODESIGN_EXECUTABLE}" --verify --deep --strict
          --verbose=2 "${STEMGENRT_BUNDLE_PATH}"
  RESULT_VARIABLE _stemgenrt_verify_result
  OUTPUT_VARIABLE _stemgenrt_verify_output
  ERROR_VARIABLE _stemgenrt_verify_error)
if(NOT _stemgenrt_verify_result EQUAL 0)
  message(FATAL_ERROR
    "Strict signature verification failed for ${STEMGENRT_BUNDLE_PATH}:\n${_stemgenrt_verify_output}${_stemgenrt_verify_error}")
endif()

message(STATUS "Strict signature verification passed: ${STEMGENRT_BUNDLE_PATH}")
