# Bind the independent reference bytes to the model selected by this checkout.
# Also usable without configuring JUCE:
# cmake -DSTEMGENRT_SOURCE_ROOT=/path/to/checkout -P VerifyParityFixture.cmake
function(stemgenrt_verify_parity_fixture source_root)
  include("${source_root}/cmake/QualifiedModelContract.cmake")
  set(fixture "${source_root}/test/fixtures/cropped1024-pytorch.bin")
  set(metadata "${source_root}/test/fixtures/cropped1024-pytorch.json")
  set(model "${source_root}/model/model.onnx")
  foreach(path IN ITEMS "${fixture}" "${metadata}" "${model}")
    if(NOT EXISTS "${path}")
      message(FATAL_ERROR "Missing parity input: ${path}")
    endif()
  endforeach()
  file(READ "${metadata}" reference)
  string(JSON expected_fixture ERROR_VARIABLE fixture_error
    GET "${reference}" fixture_sha256)
  string(JSON expected_graph ERROR_VARIABLE graph_error
    GET "${reference}" deployment_graph_sha256)
  if(fixture_error OR graph_error)
    message(FATAL_ERROR "Invalid parity fixture metadata: ${metadata}")
  endif()
  file(SHA256 "${fixture}" actual_fixture)
  file(SHA256 "${model}" actual_graph)
  file(SIZE "${model}" actual_size)
  if(NOT actual_fixture STREQUAL expected_fixture)
    message(FATAL_ERROR
      "Parity fixture SHA-256 mismatch at ${fixture}: expected ${expected_fixture}, got ${actual_fixture}")
  endif()
  if(NOT expected_graph STREQUAL STEMGENRT_QUALIFIED_MODEL_SHA256 OR
     NOT actual_graph STREQUAL STEMGENRT_QUALIFIED_MODEL_SHA256 OR
     NOT actual_size EQUAL STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE)
    message(FATAL_ERROR
      "Parity fixture/model identity mismatch. Contract: ${STEMGENRT_QUALIFIED_MODEL_SHA256}; fixture expects: ${expected_graph}; model: ${actual_graph} (${actual_size} bytes). Rebuild with matching model and independent fixture.")
  endif()
  message(STATUS "Verified independent parity fixture: ${actual_fixture}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
  if(NOT DEFINED STEMGENRT_SOURCE_ROOT)
    message(FATAL_ERROR "Set STEMGENRT_SOURCE_ROOT to the checkout being diagnosed")
  endif()
  stemgenrt_verify_parity_fixture("${STEMGENRT_SOURCE_ROOT}")
endif()
