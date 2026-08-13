cmake_minimum_required(VERSION 3.22)

# With no arguments this validates the live fail-closed draft. A substitution
# tool may instead pass an absolute C236_CONTRACT_FILE and
# C236_CONTRACT_PHASE=final. C236_MODEL_FILE and C236_EXPORT_RECEIPT_FILE bind
# disposable staged artifact bytes; C236_EXPECTED_* values below can bind every
# receipt-derived identity exactly. No override path is ever written by this
# script.
get_filename_component(STEMGENRT_SOURCE_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

if(NOT DEFINED C236_CONTRACT_FILE OR C236_CONTRACT_FILE STREQUAL "")
  set(C236_CONTRACT_FILE
    "${STEMGENRT_SOURCE_ROOT}/cmake/QualifiedModelContract.cmake")
endif()
get_filename_component(C236_CONTRACT_FILE "${C236_CONTRACT_FILE}" ABSOLUTE
  BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")
if(NOT EXISTS "${C236_CONTRACT_FILE}")
  message(FATAL_ERROR
    "C236_CONTRACT_SETUP: contract file does not exist: ${C236_CONTRACT_FILE}")
endif()

if(NOT DEFINED C236_CONTRACT_PHASE OR C236_CONTRACT_PHASE STREQUAL "")
  set(C236_CONTRACT_PHASE "pending")
endif()
string(TOLOWER "${C236_CONTRACT_PHASE}" C236_CONTRACT_PHASE)
if(C236_CONTRACT_PHASE STREQUAL "pending")
  set(c236_contract_phase_final FALSE)
  set(c236_contract_phase_final_number 0)
elseif(C236_CONTRACT_PHASE STREQUAL "final")
  set(c236_contract_phase_final TRUE)
  set(c236_contract_phase_final_number 1)
else()
  message(FATAL_ERROR
    "C236_CONTRACT_SETUP: C236_CONTRACT_PHASE must be pending or final")
endif()

include("${C236_CONTRACT_FILE}")

# Dynamic mutations must be invalid in both phases. In the pending phase they
# substitute plausible-looking final values; in the final phase they restore
# the fail-closed sentinels. This keeps the same mutation set useful before and
# after the terminal receipt values become known.
set(c236_pending_sha
  "0000000000000000000000000000000000000000000000000000000000000000")
if(c236_contract_phase_final)
  set(c236_mutated_dynamic_sha "${c236_pending_sha}")
  set(c236_mutated_model_byte_size 1)
  set(c236_mutated_run_uuid "PENDING_C236_RUN_UUID")
  set(c236_mutated_qualification_status
    "PENDING_C236_QUALIFICATION_V2_CANDIDATE_STATUS")
  set(c236_mutated_evaluation_bound "false")
else()
  set(c236_mutated_dynamic_sha
    "1000000000000000000000000000000000000000000000000000000000000000")
  set(c236_mutated_model_byte_size 2)
  set(c236_mutated_run_uuid "NOT_THE_SELECTED_RUN")
  set(c236_mutated_qualification_status
    "qualified_by_declared_quality_budget")
  set(c236_mutated_evaluation_bound "true")
endif()

set(contract_header_template
  "${STEMGENRT_SOURCE_ROOT}/cmake/QualifiedModelContract.h.in")
set(runtime_header
  "${STEMGENRT_SOURCE_ROOT}/plugin/include/StemgenRT/OnnxRuntime.h")
set(runtime_source
  "${STEMGENRT_SOURCE_ROOT}/plugin/source/OnnxRuntime.cpp")
set(output_writer_source
  "${STEMGENRT_SOURCE_ROOT}/plugin/source/OutputWriter.cpp")

function(stemgenrt_write_mutated_copy source destination old_text new_text)
  file(READ "${source}" contents)
  string(FIND "${contents}" "${old_text}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "C236_MUTATION_SETUP:${C236_MUTATION_ID}: target text was not found")
  endif()
  string(REPLACE "${old_text}" "${new_text}" contents "${contents}")
  get_filename_component(destination_dir "${destination}" DIRECTORY)
  file(MAKE_DIRECTORY "${destination_dir}")
  file(WRITE "${destination}" "${contents}")
endfunction()

# Every mutation is applied after including the real contract, in a fresh
# subprocess. This avoids include_guard state leaking between cases and proves
# that one independently drifting field is rejected by the same positive
# assertions used for the draft.
if(DEFINED C236_MUTATION_ID AND NOT C236_MUTATION_ID STREQUAL "")
  if(NOT DEFINED C236_MUTATION_SCRATCH OR
     C236_MUTATION_SCRATCH STREQUAL "")
    message(FATAL_ERROR "C236_MUTATION_SETUP: missing scratch directory")
  endif()

  if(C236_MUTATION_ID STREQUAL "contract_id")
    set(STEMGENRT_QUALIFIED_CONTRACT_ID "wrong-contract")
  elseif(C236_MUTATION_ID STREQUAL "input_audio_name")
    set(STEMGENRT_QUALIFIED_INPUT_AUDIO_NAME "audio_chunk_bad")
  elseif(C236_MUTATION_ID STREQUAL "input_analysis_history_name")
    set(STEMGENRT_QUALIFIED_INPUT_ANALYSIS_HISTORY_NAME
      "analysis_history_bad")
  elseif(C236_MUTATION_ID STREQUAL "input_hidden_name")
    set(STEMGENRT_QUALIFIED_INPUT_HIDDEN_NAME "fusion_hidden_bad")
  elseif(C236_MUTATION_ID STREQUAL "input_emitted_db_history_name")
    set(STEMGENRT_QUALIFIED_INPUT_EMITTED_DB_HISTORY_NAME
      "emitted_db_history_bad")
  elseif(C236_MUTATION_ID STREQUAL "output_separated_name")
    set(STEMGENRT_QUALIFIED_OUTPUT_SEPARATED_NAME "separated_chunk_bad")
  elseif(C236_MUTATION_ID STREQUAL "output_analysis_history_name")
    set(STEMGENRT_QUALIFIED_OUTPUT_ANALYSIS_HISTORY_NAME
      "next_analysis_history_bad")
  elseif(C236_MUTATION_ID STREQUAL "output_hidden_name")
    set(STEMGENRT_QUALIFIED_OUTPUT_HIDDEN_NAME "next_fusion_hidden_bad")
  elseif(C236_MUTATION_ID STREQUAL "output_emitted_db_history_name")
    set(STEMGENRT_QUALIFIED_OUTPUT_EMITTED_DB_HISTORY_NAME
      "next_emitted_db_history_bad")
  elseif(C236_MUTATION_ID STREQUAL "metadata_input_names")
    set(STEMGENRT_QUALIFIED_METADATA_INPUT_NAMES
      "[\"audio_chunk_bad\",\"analysis_history\",\"fusion_hidden\",\"emitted_db_history\"]")
  elseif(C236_MUTATION_ID STREQUAL "metadata_output_names")
    set(STEMGENRT_QUALIFIED_METADATA_OUTPUT_NAMES
      "[\"separated_chunk_bad\",\"next_analysis_history\",\"next_fusion_hidden\",\"next_emitted_db_history\"]")
  elseif(C236_MUTATION_ID STREQUAL "hop_samples")
    set(STEMGENRT_QUALIFIED_HOP_SAMPLES 512)
  elseif(C236_MUTATION_ID STREQUAL "analysis_history_samples")
    set(STEMGENRT_QUALIFIED_ANALYSIS_HISTORY_SAMPLES 769)
  elseif(C236_MUTATION_ID STREQUAL "analysis_window_samples")
    set(STEMGENRT_QUALIFIED_ANALYSIS_WINDOW_SAMPLES 768)
  elseif(C236_MUTATION_ID STREQUAL "fusion_hidden_size")
    set(STEMGENRT_QUALIFIED_FUSION_HIDDEN_SIZE 999)
  elseif(C236_MUTATION_ID STREQUAL "emitted_db_history_samples")
    set(STEMGENRT_QUALIFIED_EMITTED_DB_HISTORY_SAMPLES 2047)
  elseif(C236_MUTATION_ID STREQUAL "input_audio_shape")
    set(STEMGENRT_QUALIFIED_INPUT_AUDIO_SHAPE "1, 2, 255")
  elseif(C236_MUTATION_ID STREQUAL "input_analysis_history_shape")
    set(STEMGENRT_QUALIFIED_INPUT_ANALYSIS_HISTORY_SHAPE "1, 2, 767")
  elseif(C236_MUTATION_ID STREQUAL "input_hidden_shape")
    set(STEMGENRT_QUALIFIED_INPUT_HIDDEN_SHAPE "2, 1, 999")
  elseif(C236_MUTATION_ID STREQUAL "input_emitted_db_history_shape")
    set(STEMGENRT_QUALIFIED_INPUT_EMITTED_DB_HISTORY_SHAPE "1, 4, 2047")
  elseif(C236_MUTATION_ID STREQUAL "output_separated_shape")
    set(STEMGENRT_QUALIFIED_OUTPUT_SEPARATED_SHAPE "1, 4, 2, 255")
  elseif(C236_MUTATION_ID STREQUAL "output_analysis_history_shape")
    set(STEMGENRT_QUALIFIED_OUTPUT_ANALYSIS_HISTORY_SHAPE "1, 2, 767")
  elseif(C236_MUTATION_ID STREQUAL "output_hidden_shape")
    set(STEMGENRT_QUALIFIED_OUTPUT_HIDDEN_SHAPE "2, 1, 999")
  elseif(C236_MUTATION_ID STREQUAL "output_emitted_db_history_shape")
    set(STEMGENRT_QUALIFIED_OUTPUT_EMITTED_DB_HISTORY_SHAPE "1, 4, 2047")
  elseif(C236_MUTATION_ID STREQUAL "metadata_input_shapes")
    set(STEMGENRT_QUALIFIED_METADATA_INPUT_SHAPES
      "[[1,2,255],[1,2,768],[2,1,1000],[1,4,2048]]")
  elseif(C236_MUTATION_ID STREQUAL "metadata_output_shapes")
    set(STEMGENRT_QUALIFIED_METADATA_OUTPUT_SHAPES
      "[[1,4,2,255],[1,2,768],[2,1,1000],[1,4,2048]]")
  elseif(C236_MUTATION_ID STREQUAL "state_count")
    set(STEMGENRT_QUALIFIED_METADATA_STATE_COUNT "4")
  elseif(C236_MUTATION_ID STREQUAL "state_names")
    set(STEMGENRT_QUALIFIED_METADATA_STATE_NAMES
      "[\"fusion_hidden\",\"analysis_history\",\"emitted_db_history\"]")
  elseif(C236_MUTATION_ID STREQUAL "state_reset")
    set(STEMGENRT_QUALIFIED_METADATA_RESET "zero_all_4_state_tensors")
  elseif(C236_MUTATION_ID STREQUAL "initial_state")
    set(STEMGENRT_QUALIFIED_METADATA_INITIAL_STATE "uninitialized")
  elseif(C236_MUTATION_ID STREQUAL "output_delay")
    set(STEMGENRT_QUALIFIED_MODEL_OUTPUT_DELAY_CHUNKS 1)
  elseif(C236_MUTATION_ID STREQUAL "output_alignment")
    set(STEMGENRT_QUALIFIED_OUTPUT_ALIGNMENT "previous_input_chunk")
  elseif(C236_MUTATION_ID STREQUAL "causal_current_chunk")
    set(STEMGENRT_QUALIFIED_METADATA_CAUSAL_CURRENT_CHUNK "false")
  elseif(C236_MUTATION_ID STREQUAL "external_overlap_add")
    set(STEMGENRT_QUALIFIED_METADATA_EXTERNAL_OVERLAP_ADD "true")
  elseif(C236_MUTATION_ID STREQUAL "first_callback")
    set(STEMGENRT_QUALIFIED_METADATA_FIRST_CALLBACK "after_one_hop")
  elseif(C236_MUTATION_ID STREQUAL "flush")
    set(STEMGENRT_QUALIFIED_METADATA_FLUSH "one_hop")
  elseif(C236_MUTATION_ID STREQUAL "preroll")
    set(STEMGENRT_QUALIFIED_METADATA_PREROLL "one_hop")
  elseif(C236_MUTATION_ID STREQUAL "residual_source_index")
    set(STEMGENRT_QUALIFIED_RESIDUAL_SOURCE_INDEX 2)
  elseif(C236_MUTATION_ID STREQUAL "other_source_index")
    set(STEMGENRT_QUALIFIED_OTHER_SOURCE_INDEX 2)
  elseif(C236_MUTATION_ID STREQUAL "mixture_consistency")
    set(STEMGENRT_QUALIFIED_METADATA_MIXTURE_CONSISTENCY "none")
  elseif(C236_MUTATION_ID STREQUAL "source_order")
    set(STEMGENRT_QUALIFIED_METADATA_SOURCE_ORDER
      "[\"bass\",\"drums\",\"vocals\",\"other\"]")
  elseif(C236_MUTATION_ID STREQUAL "model_sha256")
    set(STEMGENRT_QUALIFIED_MODEL_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "model_byte_size")
    set(STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE "${c236_mutated_model_byte_size}")
  elseif(C236_MUTATION_ID STREQUAL "export_receipt_sha256")
    set(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "export_receipt_schema_version")
    set(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SCHEMA_VERSION 4)
  elseif(C236_MUTATION_ID STREQUAL "export_receipt_kind")
    set(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_KIND "wrong_export_receipt")
  elseif(C236_MUTATION_ID STREQUAL "export_receipt_status")
    set(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_STATUS "fail")
  elseif(C236_MUTATION_ID STREQUAL "export_receipt_deployment_status")
    set(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_DEPLOYMENT_STATUS
      "promoted_production")
  elseif(C236_MUTATION_ID STREQUAL "export_receipt_gate_pass")
    set(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_GATE_PASS "false")
  elseif(C236_MUTATION_ID STREQUAL "export_receipt_cpu_only")
    set(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_CPU_ONLY "false")
  elseif(C236_MUTATION_ID STREQUAL "export_receipt_promotion_performed")
    set(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_PROMOTION_PERFORMED "true")
  elseif(C236_MUTATION_ID STREQUAL "selected_update")
    set(STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE 1)
  elseif(C236_MUTATION_ID STREQUAL "run_uuid")
    set(STEMGENRT_QUALIFIED_C236_RUN_UUID "${c236_mutated_run_uuid}")
  elseif(C236_MUTATION_ID STREQUAL "contract_identity_sha256")
    set(STEMGENRT_QUALIFIED_C236_CONTRACT_IDENTITY_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "static_identity_sha256")
    set(STEMGENRT_QUALIFIED_C236_STATIC_IDENTITY_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "source_checkpoint_sha256")
    set(STEMGENRT_QUALIFIED_C236_SOURCE_CHECKPOINT_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "materialized_artifact_sha256")
    set(STEMGENRT_QUALIFIED_C236_MATERIALIZED_ARTIFACT_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "materialization_receipt_sha256")
    set(STEMGENRT_QUALIFIED_C236_MATERIALIZATION_RECEIPT_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "model_state_sha256")
    set(STEMGENRT_QUALIFIED_C236_MODEL_STATE_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "composite_runtime_state_sha256")
    set(STEMGENRT_QUALIFIED_C236_COMPOSITE_RUNTIME_STATE_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "materializer_sha256")
    set(STEMGENRT_QUALIFIED_C236_MATERIALIZER_SHA256
      "6da7c0d97faeabdfb87fbdf3d37e5a2af2847f8a1595fc0467b01a14ea59ca2d")
  elseif(C236_MUTATION_ID STREQUAL "shared_authority_sha256")
    set(STEMGENRT_QUALIFIED_C236_SHARED_AUTHORITY_SHA256
      "7fc64af8968199e926f529ac9626f20c5f9fd6c28bcea02ae426353ae19246d2")
  elseif(C236_MUTATION_ID STREQUAL "recovery_receipt_authority_sha256")
    set(STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_AUTHORITY_SHA256
      "ecb8edac6e94512d9be3b98d225541cf08458a3f3e015a139f36ac774bb76aa8")
  elseif(C236_MUTATION_ID STREQUAL "terminal_chain_authority_sha256")
    set(STEMGENRT_QUALIFIED_C236_TERMINAL_CHAIN_AUTHORITY_SHA256
      "f8d23b5e78f36482e5f60b2e0d4b5b40d5cb91a242882f26a4f85f362a376cc9")
  elseif(C236_MUTATION_ID STREQUAL "recovery_contract_sha256")
    set(STEMGENRT_QUALIFIED_C236_RECOVERY_CONTRACT_SHA256
      "b1125772696b9f05d699c9de2f92caaa92cfaa1a58d74b80713ea6d021db4501")
  elseif(C236_MUTATION_ID STREQUAL "recovery_receipt_sha256")
    set(STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "final_chain_receipt_sha256")
    set(STEMGENRT_QUALIFIED_C236_FINAL_CHAIN_RECEIPT_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "qualification_v2_sha256")
    set(STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_SHA256
      "${c236_mutated_dynamic_sha}")
  elseif(C236_MUTATION_ID STREQUAL "qualification_v2_candidate_status")
    set(STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_CANDIDATE_STATUS
      "${c236_mutated_qualification_status}")
  elseif(C236_MUTATION_ID STREQUAL "evaluation_result_bound")
    set(STEMGENRT_QUALIFIED_C236_EVALUATION_RESULT_BOUND
      "${c236_mutated_evaluation_bound}")
  elseif(C236_MUTATION_ID STREQUAL "c191_payload_sha256")
    set(STEMGENRT_QUALIFIED_C191_PAYLOAD_SHA256
      "b80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b")
  elseif(C236_MUTATION_ID STREQUAL "c191_head_state_sha256")
    set(STEMGENRT_QUALIFIED_C191_HEAD_STATE_SHA256
      "338991eced221734b5e931bbc4256412dd6ed361656f2cbe4864abccedd22bcf")
  elseif(C236_MUTATION_ID STREQUAL "c236_family")
    set(STEMGENRT_QUALIFIED_C236_FAMILY "wrong-family")
  elseif(C236_MUTATION_ID STREQUAL "candidate_kind")
    set(STEMGENRT_QUALIFIED_MODEL_CANDIDATE_KIND "wrong_candidate")
  elseif(C236_MUTATION_ID STREQUAL "deployment_status")
    set(STEMGENRT_QUALIFIED_METADATA_DEPLOYMENT_STATUS "plugin_qualified")
  elseif(C236_MUTATION_ID STREQUAL "external_data")
    set(STEMGENRT_QUALIFIED_METADATA_EXTERNAL_DATA "true")
  elseif(C236_MUTATION_ID STREQUAL "header_dtype")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.export.dtype", "float32"}]=]
      [=[{"hs_tasnet.export.dtype", "float16"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_evaluation_bound")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.evaluation_result_bound", kC236EvaluationResultBound}]=]
      [=[{"hs_tasnet.c236.evaluation_result_bound", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_previous_qualification")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.previous_qualification_transferred", "false"}]=]
      [=[{"hs_tasnet.c236.previous_qualification_transferred", "true"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_c191_qualification")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.streaming.c191_qualification_transferred", "false"}]=]
      [=[{"hs_tasnet.streaming.c191_qualification_transferred", "true"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_opset")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.export.opset_version", "17"}]=]
      [=[{"hs_tasnet.export.opset_version", "18"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_filename")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.export.output_filename", "model.onnx"}]=]
      [=[{"hs_tasnet.export.output_filename", "candidate.onnx"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_future_context")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.streaming.future_context_samples", "0"}]=]
      [=[{"hs_tasnet.streaming.future_context_samples", "256"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_residual_association")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      "audio_chunk-minus-sum-drums-bass-vocals-dim1-float32"
      "audio_chunk-minus-sum-drums-bass-dim1-float32")
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_residual_name")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.export.residual_source_name", "other"}]=]
      [=[{"hs_tasnet.export.residual_source_name", "bass"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_composite_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}" "     kC236CompositeRuntimeStateSha256},"
      "     \"mismatch\"},")
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_contract_identity_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.contract_identity_sha256", kC236ContractIdentitySha256}]=]
      [=[{"hs_tasnet.c236.contract_identity_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_static_identity_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.static_identity_sha256", kC236StaticIdentitySha256}]=]
      [=[{"hs_tasnet.c236.static_identity_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_model_state_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.model_state_sha256", kC236ModelStateSha256}]=]
      [=[{"hs_tasnet.c236.model_state_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_materializer_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.materializer_sha256", kC236MaterializerSha256}]=]
      [=[{"hs_tasnet.c236.materializer_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_shared_authority_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.shared_authority_sha256", kC236SharedAuthoritySha256}]=]
      [=[{"hs_tasnet.c236.shared_authority_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_recovery_receipt_authority_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.recovery_receipt_authority_sha256",]=]
      [=[{"hs_tasnet.c236.recovery_receipt_authority_sha256_bad",]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_terminal_chain_authority_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.terminal_chain_authority_sha256",]=]
      [=[{"hs_tasnet.c236.terminal_chain_authority_sha256_bad",]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_recovery_contract_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.recovery_contract_sha256", kC236RecoveryContractSha256}]=]
      [=[{"hs_tasnet.c236.recovery_contract_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_recovery_receipt_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.recovery_receipt_sha256", kC236RecoveryReceiptSha256}]=]
      [=[{"hs_tasnet.c236.recovery_receipt_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_final_chain_receipt_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}" "     kC236FinalChainReceiptSha256},"
      "     \"mismatch\"},")
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_qualification_v2_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.qualification_v2_sha256", kC236QualificationV2Sha256}]=]
      [=[{"hs_tasnet.c236.qualification_v2_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_qualification_status_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}" "     kC236QualificationV2CandidateStatus},"
      "     \"mismatch\"},")
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_family_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.family", kC236Family}]=]
      [=[{"hs_tasnet.c236.family", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_candidate_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.model.candidate_kind", kModelCandidateKind}]=]
      [=[{"hs_tasnet.model.candidate_kind", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_artifact_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}" "     kC236MaterializedArtifactSha256},"
      "     \"mismatch\"},")
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_receipt_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}" "     kC236MaterializationReceiptSha256},"
      "     \"mismatch\"},")
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_checkpoint_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.model.checkpoint_sha256", kC236MaterializedArtifactSha256}]=]
      [=[{"hs_tasnet.model.checkpoint_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_checkpoint_state_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.model.checkpoint_state_sha256", kC236ModelStateSha256}]=]
      [=[{"hs_tasnet.model.checkpoint_state_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_source_checkpoint_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.source_checkpoint_sha256", kC236SourceCheckpointSha256}]=]
      [=[{"hs_tasnet.c236.source_checkpoint_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_run_uuid_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c236.run_uuid", kC236RunUuid}]=]
      [=[{"hs_tasnet.c236.run_uuid", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_c191_payload_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c191.payload_sha256", kC191PayloadSha256}]=]
      [=[{"hs_tasnet.c191.payload_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_c191_head_binding")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c191.head_state_sha256", kC191HeadStateSha256}]=]
      [=[{"hs_tasnet.c191.head_state_sha256", "mismatch"}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_duplicate_metadata_key")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      "hs_tasnet.c236.static_identity_sha256"
      "hs_tasnet.c236.contract_identity_sha256")
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "header_embed_export_receipt")
    set(mutated_header "${C236_MUTATION_SCRATCH}/QualifiedModelContract.h.in")
    stemgenrt_write_mutated_copy("${contract_header_template}"
      "${mutated_header}"
      [=[{"hs_tasnet.c191.emitted_db_history_shape", kEmittedDbHistoryShapeJson}]=]
      [=[{"hs_tasnet.c236.export_receipt_sha256", kExportReceiptSha256}]=])
    set(contract_header_template "${mutated_header}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_dtype_guard")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||"
      "elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_name_guard")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "if (actualName != expectedName)"
      "if (actualName == expectedName)")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_shape_guard")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "actualShape != expectedShape)"
      "actualShape == expectedShape)")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_metadata_guard")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "if (actualValue != expectedValue)"
      "if (actualValue == expectedValue)")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_model_size_guard")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "if (modelFile.getSize() != qualified_model::kModelByteSize)"
      "if (modelFile.getSize() == qualified_model::kModelByteSize)")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_model_sha_guard")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "if (!actualModelSha.equalsIgnoreCase("
      "if (actualModelSha.equalsIgnoreCase(")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_fusion_state_output")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "nextFusionHiddenBuffer_" "nextFusionHiddenBufferBad_")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_fusion_state_reset")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "std::fill(fusionHidden_.begin(), fusionHidden_.end(), 0.0f);"
      "std::fill(fusionHidden_.begin(), fusionHidden_.end(), 1.0f);")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "runtime_fusion_state_copy")
    set(mutated_runtime "${C236_MUTATION_SCRATCH}/OnnxRuntime.cpp")
    stemgenrt_write_mutated_copy("${runtime_source}" "${mutated_runtime}"
      "std::memcpy(fusionHidden_.data(), nextFusionHiddenBuffer_.data(),"
      "std::memcpy(fusionHidden_.data(), analysisHistory_.data(),")
    set(runtime_source "${mutated_runtime}")
  elseif(C236_MUTATION_ID STREQUAL "output_residual_formula")
    set(mutated_output_writer "${C236_MUTATION_SCRATCH}/OutputWriter.cpp")
    stemgenrt_write_mutated_copy("${output_writer_source}"
      "${mutated_output_writer}"
      "stems[kStemBass] - stems[kStemVocals];"
      "stems[kStemBass] + stems[kStemVocals];")
    set(output_writer_source "${mutated_output_writer}")
  else()
    message(FATAL_ERROR
      "C236_MUTATION_SETUP: unknown mutation '${C236_MUTATION_ID}'")
  endif()
endif()

function(stemgenrt_expect_equal variable expected)
  if(NOT "${${variable}}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "C236_CONTRACT_REJECT:${variable}: expected '${expected}', got '${${variable}}'")
  endif()
endfunction()

function(stemgenrt_expect_nonzero_sha variable)
  string(LENGTH "${${variable}}" sha_length)
  if(NOT sha_length EQUAL 64 OR
     "${${variable}}" MATCHES "[^0-9a-f]" OR
     "${${variable}}" STREQUAL "${c236_pending_sha}")
    message(FATAL_ERROR
      "C236_CONTRACT_REJECT:${variable}: expected a non-sentinel lowercase SHA-256, got '${${variable}}'")
  endif()
endfunction()

function(stemgenrt_expect_optional_exact expected_variable contract_variable)
  if(DEFINED ${expected_variable} AND
     NOT "${${expected_variable}}" STREQUAL "")
    stemgenrt_expect_equal(${contract_variable} "${${expected_variable}}")
  endif()
endfunction()

function(stemgenrt_expect_source_contains path needle tag)
  file(READ "${path}" source)
  string(FIND "${source}" "${needle}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "C236_CONTRACT_REJECT:${tag}: ${path} lacks required text: ${needle}")
  endif()
endfunction()

function(stemgenrt_expect_source_excludes path needle)
  file(READ "${path}" source)
  string(FIND "${source}" "${needle}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR
      "C236_CONTRACT_REJECT:obsolete_state_abi: ${path} contains: ${needle}")
  endif()
endfunction()

stemgenrt_expect_equal(STEMGENRT_QUALIFIED_CONTRACT_ID
  "c236-terminal-selected-hop256-audition-v1")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SCHEMA_VERSION "3")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_KIND
  "hs_tasnet_c236_recovery_checked_onnx_export_v3")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_STATUS "pass")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_DEPLOYMENT_STATUS
  "qualified_unpromoted_listening_candidate")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_GATE_PASS "true")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_CPU_ONLY "true")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_PROMOTION_PERFORMED
  "false")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C236_FAMILY
  "c236-c214-native-db-separator-v1")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_MODEL_CANDIDATE_KIND
  "c236_terminal_selected_plus_exact_c191_hop256")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_SAMPLE_RATE "44100")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_CHANNELS "2")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_STEMS "4")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_HOP_SAMPLES "256")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_ANALYSIS_WINDOW_SAMPLES "1024")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_ANALYSIS_HISTORY_SAMPLES "768")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_FUSION_HIDDEN_LAYERS "2")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_FUSION_HIDDEN_SIZE "1000")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EMITTED_DB_CHANNELS "4")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EMITTED_DB_HISTORY_SAMPLES "2048")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_MODEL_OUTPUT_DELAY_CHUNKS "0")

stemgenrt_expect_equal(STEMGENRT_QUALIFIED_INPUT_AUDIO_NAME "audio_chunk")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_INPUT_ANALYSIS_HISTORY_NAME
  "analysis_history")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_INPUT_HIDDEN_NAME "fusion_hidden")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_INPUT_EMITTED_DB_HISTORY_NAME
  "emitted_db_history")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_SEPARATED_NAME
  "separated_chunk")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_ANALYSIS_HISTORY_NAME
  "next_analysis_history")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_HIDDEN_NAME
  "next_fusion_hidden")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_EMITTED_DB_HISTORY_NAME
  "next_emitted_db_history")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_INPUT_AUDIO_SHAPE "1, 2, 256")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_INPUT_ANALYSIS_HISTORY_SHAPE
  "1, 2, 768")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_INPUT_HIDDEN_SHAPE "2, 1, 1000")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_INPUT_EMITTED_DB_HISTORY_SHAPE
  "1, 4, 2048")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_SEPARATED_SHAPE
  "1, 4, 2, 256")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_ANALYSIS_HISTORY_SHAPE
  "1, 2, 768")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_HIDDEN_SHAPE "2, 1, 1000")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_EMITTED_DB_HISTORY_SHAPE
  "1, 4, 2048")

stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_EXPORT_MODE "streaming")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_EXTERNAL_DATA "false")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_INPUT_NAMES
  "[\"audio_chunk\",\"analysis_history\",\"fusion_hidden\",\"emitted_db_history\"]")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_OUTPUT_NAMES
  "[\"separated_chunk\",\"next_analysis_history\",\"next_fusion_hidden\",\"next_emitted_db_history\"]")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_INPUT_SHAPES
  "[[1,2,256],[1,2,768],[2,1,1000],[1,4,2048]]")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_OUTPUT_SHAPES
  "[[1,4,2,256],[1,2,768],[2,1,1000],[1,4,2048]]")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_STATE_COUNT "3")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_STATE_NAMES
  "[\"analysis_history\",\"fusion_hidden\",\"emitted_db_history\"]")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_RESET
  "zero_all_3_state_tensors")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_CAUSAL_CURRENT_CHUNK
  "true")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_EXTERNAL_OVERLAP_ADD
  "false")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OUTPUT_ALIGNMENT
  "current_input_chunk")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_FIRST_CALLBACK
  "current_input_chunk")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_FLUSH "none")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_PREROLL "none")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_INITIAL_STATE
  "all_positive_zero")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_MIXTURE_CONSISTENCY
  "exact_float32_residual_to_other")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_SOURCE_ORDER
  "[\"drums\",\"bass\",\"vocals\",\"other\"]")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_DRUMS_SOURCE_INDEX "0")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_BASS_SOURCE_INDEX "1")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_VOCALS_SOURCE_INDEX "2")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_OTHER_SOURCE_INDEX "3")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_RESIDUAL_SOURCE_INDEX "3")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_METADATA_DEPLOYMENT_STATUS
  "post_training_export_candidate_not_plugin_qualified")

set(c236_dynamic_sha_variables
    STEMGENRT_QUALIFIED_MODEL_SHA256
    STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SHA256
    STEMGENRT_QUALIFIED_C236_CONTRACT_IDENTITY_SHA256
    STEMGENRT_QUALIFIED_C236_STATIC_IDENTITY_SHA256
    STEMGENRT_QUALIFIED_C236_SOURCE_CHECKPOINT_SHA256
    STEMGENRT_QUALIFIED_C236_MATERIALIZED_ARTIFACT_SHA256
    STEMGENRT_QUALIFIED_C236_MATERIALIZATION_RECEIPT_SHA256
    STEMGENRT_QUALIFIED_C236_MODEL_STATE_SHA256
    STEMGENRT_QUALIFIED_C236_COMPOSITE_RUNTIME_STATE_SHA256
    STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_SHA256
    STEMGENRT_QUALIFIED_C236_FINAL_CHAIN_RECEIPT_SHA256
    STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_SHA256)

if(c236_contract_phase_final)
  # Final-mode validation is deliberately usable before the exact terminal
  # identities are known. Optional C236_EXPECTED_* inputs below tighten each
  # value when a receipt-driven substitution tool has those identities.
  foreach(final_sha_variable IN LISTS c236_dynamic_sha_variables)
    stemgenrt_expect_nonzero_sha(${final_sha_variable})
  endforeach()
  if(NOT STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE MATCHES "^[0-9]+$" OR
     STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE LESS_EQUAL 1)
    message(FATAL_ERROR
      "C236_CONTRACT_REJECT:STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE: final model must be larger than the one-byte sentinel")
  endif()
  if(NOT STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE STREQUAL "32768" AND
     NOT STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE STREQUAL "65536" AND
     NOT STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE STREQUAL "100000")
    message(FATAL_ERROR
      "C236_CONTRACT_REJECT:STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE: final update must be 32768, 65536, or 100000")
  endif()
  if(STEMGENRT_QUALIFIED_C236_RUN_UUID STREQUAL "" OR
     STEMGENRT_QUALIFIED_C236_RUN_UUID STREQUAL "PENDING_C236_RUN_UUID")
    message(FATAL_ERROR
      "C236_CONTRACT_REJECT:STEMGENRT_QUALIFIED_C236_RUN_UUID: final run UUID must be bound")
  endif()
  stemgenrt_expect_equal(
    STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_CANDIDATE_STATUS
    "qualified_by_declared_quality_budget")
  stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C236_EVALUATION_RESULT_BOUND
    "true")
else()
  # The default draft remains impossible to package until every receipt-bound
  # value and the checked model are installed atomically.
  foreach(pending_sha_variable IN LISTS c236_dynamic_sha_variables)
    stemgenrt_expect_equal(${pending_sha_variable} "${c236_pending_sha}")
  endforeach()
  stemgenrt_expect_equal(STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE "1")
  stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE "0")
  stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C236_RUN_UUID
    "PENDING_C236_RUN_UUID")
  stemgenrt_expect_equal(
    STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_CANDIDATE_STATUS
    "PENDING_C236_QUALIFICATION_V2_CANDIDATE_STATUS")
  stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C236_EVALUATION_RESULT_BOUND
    "false")
endif()

set(c236_expected_identity_bindings
  "C236_EXPECTED_MODEL_SHA256|STEMGENRT_QUALIFIED_MODEL_SHA256"
  "C236_EXPECTED_MODEL_BYTE_SIZE|STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE"
  "C236_EXPECTED_EXPORT_RECEIPT_SHA256|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SHA256"
  "C236_EXPECTED_SELECTED_UPDATE|STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE"
  "C236_EXPECTED_RUN_UUID|STEMGENRT_QUALIFIED_C236_RUN_UUID"
  "C236_EXPECTED_CONTRACT_IDENTITY_SHA256|STEMGENRT_QUALIFIED_C236_CONTRACT_IDENTITY_SHA256"
  "C236_EXPECTED_STATIC_IDENTITY_SHA256|STEMGENRT_QUALIFIED_C236_STATIC_IDENTITY_SHA256"
  "C236_EXPECTED_SOURCE_CHECKPOINT_SHA256|STEMGENRT_QUALIFIED_C236_SOURCE_CHECKPOINT_SHA256"
  "C236_EXPECTED_MATERIALIZED_ARTIFACT_SHA256|STEMGENRT_QUALIFIED_C236_MATERIALIZED_ARTIFACT_SHA256"
  "C236_EXPECTED_MATERIALIZATION_RECEIPT_SHA256|STEMGENRT_QUALIFIED_C236_MATERIALIZATION_RECEIPT_SHA256"
  "C236_EXPECTED_MODEL_STATE_SHA256|STEMGENRT_QUALIFIED_C236_MODEL_STATE_SHA256"
  "C236_EXPECTED_COMPOSITE_RUNTIME_STATE_SHA256|STEMGENRT_QUALIFIED_C236_COMPOSITE_RUNTIME_STATE_SHA256"
  "C236_EXPECTED_RECOVERY_RECEIPT_SHA256|STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_SHA256"
  "C236_EXPECTED_FINAL_CHAIN_RECEIPT_SHA256|STEMGENRT_QUALIFIED_C236_FINAL_CHAIN_RECEIPT_SHA256"
  "C236_EXPECTED_QUALIFICATION_V2_SHA256|STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_SHA256"
  "C236_EXPECTED_QUALIFICATION_V2_CANDIDATE_STATUS|STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_CANDIDATE_STATUS")
foreach(expected_identity_binding IN LISTS c236_expected_identity_bindings)
  string(REPLACE "|" ";" expected_identity_parts
    "${expected_identity_binding}")
  list(GET expected_identity_parts 0 expected_identity_variable)
  list(GET expected_identity_parts 1 contract_identity_variable)
  stemgenrt_expect_optional_exact(${expected_identity_variable}
    ${contract_identity_variable})
endforeach()

# Optional staged artifact paths cross-check the values that will be copied in
# the final atomic substitution. They are read-only and are never defaulted to
# the live model/evidence paths.
if(DEFINED C236_MODEL_FILE AND NOT C236_MODEL_FILE STREQUAL "")
  get_filename_component(C236_MODEL_FILE "${C236_MODEL_FILE}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")
  if(NOT EXISTS "${C236_MODEL_FILE}" OR IS_DIRECTORY "${C236_MODEL_FILE}")
    message(FATAL_ERROR
      "C236_CONTRACT_SETUP: staged model file does not exist: ${C236_MODEL_FILE}")
  endif()
  file(SHA256 "${C236_MODEL_FILE}" staged_model_sha256)
  file(SIZE "${C236_MODEL_FILE}" staged_model_byte_size)
  stemgenrt_expect_equal(STEMGENRT_QUALIFIED_MODEL_SHA256
    "${staged_model_sha256}")
  stemgenrt_expect_equal(STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE
    "${staged_model_byte_size}")
endif()
if(DEFINED C236_EXPORT_RECEIPT_FILE AND
   NOT C236_EXPORT_RECEIPT_FILE STREQUAL "")
  get_filename_component(C236_EXPORT_RECEIPT_FILE
    "${C236_EXPORT_RECEIPT_FILE}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")
  if(NOT EXISTS "${C236_EXPORT_RECEIPT_FILE}" OR
     IS_DIRECTORY "${C236_EXPORT_RECEIPT_FILE}")
    message(FATAL_ERROR
      "C236_CONTRACT_SETUP: staged export receipt does not exist: ${C236_EXPORT_RECEIPT_FILE}")
  endif()
  file(SHA256 "${C236_EXPORT_RECEIPT_FILE}" staged_export_receipt_sha256)
  stemgenrt_expect_equal(STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SHA256
    "${staged_export_receipt_sha256}")
endif()
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C236_MATERIALIZER_SHA256
  "5da7c0d97faeabdfb87fbdf3d37e5a2af2847f8a1595fc0467b01a14ea59ca2d")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C236_SHARED_AUTHORITY_SHA256
  "6fc64af8968199e926f529ac9626f20c5f9fd6c28bcea02ae426353ae19246d2")
stemgenrt_expect_equal(
  STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_AUTHORITY_SHA256
  "fcb8edac6e94512d9be3b98d225541cf08458a3f3e015a139f36ac774bb76aa8")
stemgenrt_expect_equal(
  STEMGENRT_QUALIFIED_C236_TERMINAL_CHAIN_AUTHORITY_SHA256
  "e8d23b5e78f36482e5f60b2e0d4b5b40d5cb91a242882f26a4f85f362a376cc9")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C236_RECOVERY_CONTRACT_SHA256
  "a1125772696b9f05d699c9de2f92caaa92cfaa1a58d74b80713ea6d021db4501")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C191_PAYLOAD_SHA256
  "a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b")
stemgenrt_expect_equal(STEMGENRT_QUALIFIED_C191_HEAD_STATE_SHA256
  "238991eced221734b5e931bbc4256412dd6ed361656f2cbe4864abccedd22bcf")

stemgenrt_expect_source_contains("${runtime_header}"
  "std::array<OrtValue*, 4> inputTensorValues_" "runtime_four_inputs")
stemgenrt_expect_source_contains("${runtime_header}"
  "std::array<OrtValue*, 4> outputTensorValues_" "runtime_four_outputs")
stemgenrt_expect_source_contains("${runtime_source}"
  "std::array<const char*, 4> inputNames" "runtime_four_input_names")
stemgenrt_expect_source_contains("${runtime_source}"
  "std::array<const char*, 4> outputNames" "runtime_four_output_names")
stemgenrt_expect_source_contains("${runtime_source}"
  "elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||"
  "runtime_float32_guard")
stemgenrt_expect_source_contains("${runtime_source}"
  "if (actualName != expectedName)" "runtime_name_guard")
stemgenrt_expect_source_contains("${runtime_source}"
  "actualShape != expectedShape)" "runtime_shape_guard")
stemgenrt_expect_source_contains("${runtime_source}"
  "for (const auto& [key, expectedValue] : qualified_model::kMetadata)"
  "runtime_all_metadata_guard")
stemgenrt_expect_source_contains("${runtime_source}"
  "if (actualValue != expectedValue)" "runtime_metadata_guard")
stemgenrt_expect_source_contains("${runtime_source}"
  "if (modelFile.getSize() != qualified_model::kModelByteSize)"
  "runtime_model_size_guard")
stemgenrt_expect_source_contains("${runtime_source}"
  "if (!actualModelSha.equalsIgnoreCase(" "runtime_model_sha_guard")
stemgenrt_expect_source_contains("${runtime_source}"
  "nextAnalysisHistoryBuffer_" "runtime_analysis_history_state")
stemgenrt_expect_source_contains("${runtime_source}"
  "nextFusionHiddenBuffer_" "runtime_fusion_hidden_state")
stemgenrt_expect_source_contains("${runtime_source}"
  "nextEmittedDbHistoryBuffer_" "runtime_emitted_db_state")
stemgenrt_expect_source_contains("${runtime_source}"
  "std::fill(analysisHistory_.begin(), analysisHistory_.end(), 0.0f);"
  "runtime_analysis_history_reset")
stemgenrt_expect_source_contains("${runtime_source}"
  "std::fill(fusionHidden_.begin(), fusionHidden_.end(), 0.0f);"
  "runtime_fusion_hidden_reset")
stemgenrt_expect_source_contains("${runtime_source}"
  "std::fill(emittedDbHistory_.begin(), emittedDbHistory_.end(), 0.0f);"
  "runtime_emitted_db_reset")
stemgenrt_expect_source_contains("${runtime_source}"
  "std::memcpy(analysisHistory_.data(), nextAnalysisHistoryBuffer_.data(),"
  "runtime_analysis_history_copy")
stemgenrt_expect_source_contains("${runtime_source}"
  "std::memcpy(fusionHidden_.data(), nextFusionHiddenBuffer_.data(),"
  "runtime_fusion_hidden_copy")
stemgenrt_expect_source_contains("${runtime_source}"
  "std::memcpy(emittedDbHistory_.data(),"
  "runtime_emitted_db_copy")
stemgenrt_expect_source_contains("${output_writer_source}"
  "for (const int stem : {kStemDrums, kStemBass, kStemVocals})"
  "output_three_retained_sources")
stemgenrt_expect_source_contains("${output_writer_source}"
  "stems[kStemOther] = mainOutput[ch] - stems[kStemDrums] -"
  "output_residual_to_other")
stemgenrt_expect_source_contains("${output_writer_source}"
  "stems[kStemBass] - stems[kStemVocals];"
  "output_exact_reconstruction")
stemgenrt_expect_source_contains("${output_writer_source}"
  "stems[kStemOther] = mainOutput[ch];" "output_lossless_fallback")
foreach(obsolete IN ITEMS
    pastAudio_ c130History_ previousHidden_ adapterValid_
    rawParentHistory_ nextPastAudioBuffer_ nextC130HistoryBuffer_
    nextPreviousHiddenBuffer_ nextAdapterValidBuffer_
    nextRawParentHistoryBuffer_
    "inputTensorValues_[4" "inputTensorValues_[5" "inputTensorValues_[6"
    "inputTensorValues_[7" "outputTensorValues_[4" "outputTensorValues_[5"
    "outputTensorValues_[6" "outputTensorValues_[7"
    "kInputNames[4" "kInputNames[5" "kInputNames[6" "kInputNames[7"
    "kOutputNames[4" "kOutputNames[5" "kOutputNames[6" "kOutputNames[7")
  stemgenrt_expect_source_excludes("${runtime_header}" "${obsolete}")
  stemgenrt_expect_source_excludes("${runtime_source}" "${obsolete}")
endforeach()

if(DEFINED C236_MUTATION_SCRATCH AND
   NOT C236_MUTATION_SCRATCH STREQUAL "")
  set(scratch "${C236_MUTATION_SCRATCH}/generated")
else()
  set(scratch "$ENV{TMPDIR}")
  if(scratch STREQUAL "")
    set(scratch "/tmp")
  endif()
  string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef scratch_suffix)
  set(scratch "${scratch}/stemgenrt-c236-contract-static-${scratch_suffix}")
endif()
file(REMOVE_RECURSE "${scratch}")
file(MAKE_DIRECTORY "${scratch}/StemgenRT")
configure_file("${contract_header_template}"
  "${scratch}/StemgenRT/QualifiedModelContract.h" @ONLY)

find_program(STEMGENRT_CXX NAMES clang++ g++ c++)
if(NOT STEMGENRT_CXX)
  message(FATAL_ERROR "No C++20 compiler found for the contract syntax check")
endif()
execute_process(
  COMMAND "${STEMGENRT_CXX}" -std=c++20 -Wall -Wextra -Werror
          -fsyntax-only
          "-DC236_CONTRACT_PHASE_FINAL=${c236_contract_phase_final_number}"
          -I "${scratch}"
          -I "${STEMGENRT_SOURCE_ROOT}/plugin/include"
          "${STEMGENRT_SOURCE_ROOT}/test/source/C236ContractCompileOnly.cpp"
  RESULT_VARIABLE compile_result
  OUTPUT_VARIABLE compile_output
  ERROR_VARIABLE compile_error)
file(REMOVE_RECURSE "${scratch}")
if(NOT compile_result EQUAL 0)
  message(FATAL_ERROR
    "c236 generated-contract compile check failed:\n${compile_output}${compile_error}")
endif()

if(DEFINED C236_MUTATION_ID AND NOT C236_MUTATION_ID STREQUAL "")
  message(STATUS "C236_MUTATION_SURVIVED:${C236_MUTATION_ID}")
  return()
endif()

set(mutation_cases
  "contract_id|STEMGENRT_QUALIFIED_CONTRACT_ID"
  "input_audio_name|STEMGENRT_QUALIFIED_INPUT_AUDIO_NAME"
  "input_analysis_history_name|STEMGENRT_QUALIFIED_INPUT_ANALYSIS_HISTORY_NAME"
  "input_hidden_name|STEMGENRT_QUALIFIED_INPUT_HIDDEN_NAME"
  "input_emitted_db_history_name|STEMGENRT_QUALIFIED_INPUT_EMITTED_DB_HISTORY_NAME"
  "output_separated_name|STEMGENRT_QUALIFIED_OUTPUT_SEPARATED_NAME"
  "output_analysis_history_name|STEMGENRT_QUALIFIED_OUTPUT_ANALYSIS_HISTORY_NAME"
  "output_hidden_name|STEMGENRT_QUALIFIED_OUTPUT_HIDDEN_NAME"
  "output_emitted_db_history_name|STEMGENRT_QUALIFIED_OUTPUT_EMITTED_DB_HISTORY_NAME"
  "metadata_input_names|STEMGENRT_QUALIFIED_METADATA_INPUT_NAMES"
  "metadata_output_names|STEMGENRT_QUALIFIED_METADATA_OUTPUT_NAMES"
  "hop_samples|STEMGENRT_QUALIFIED_HOP_SAMPLES"
  "analysis_history_samples|STEMGENRT_QUALIFIED_ANALYSIS_HISTORY_SAMPLES"
  "analysis_window_samples|STEMGENRT_QUALIFIED_ANALYSIS_WINDOW_SAMPLES"
  "fusion_hidden_size|STEMGENRT_QUALIFIED_FUSION_HIDDEN_SIZE"
  "emitted_db_history_samples|STEMGENRT_QUALIFIED_EMITTED_DB_HISTORY_SAMPLES"
  "input_audio_shape|STEMGENRT_QUALIFIED_INPUT_AUDIO_SHAPE"
  "input_analysis_history_shape|STEMGENRT_QUALIFIED_INPUT_ANALYSIS_HISTORY_SHAPE"
  "input_hidden_shape|STEMGENRT_QUALIFIED_INPUT_HIDDEN_SHAPE"
  "input_emitted_db_history_shape|STEMGENRT_QUALIFIED_INPUT_EMITTED_DB_HISTORY_SHAPE"
  "output_separated_shape|STEMGENRT_QUALIFIED_OUTPUT_SEPARATED_SHAPE"
  "output_analysis_history_shape|STEMGENRT_QUALIFIED_OUTPUT_ANALYSIS_HISTORY_SHAPE"
  "output_hidden_shape|STEMGENRT_QUALIFIED_OUTPUT_HIDDEN_SHAPE"
  "output_emitted_db_history_shape|STEMGENRT_QUALIFIED_OUTPUT_EMITTED_DB_HISTORY_SHAPE"
  "metadata_input_shapes|STEMGENRT_QUALIFIED_METADATA_INPUT_SHAPES"
  "metadata_output_shapes|STEMGENRT_QUALIFIED_METADATA_OUTPUT_SHAPES"
  "state_count|STEMGENRT_QUALIFIED_METADATA_STATE_COUNT"
  "state_names|STEMGENRT_QUALIFIED_METADATA_STATE_NAMES"
  "state_reset|STEMGENRT_QUALIFIED_METADATA_RESET"
  "initial_state|STEMGENRT_QUALIFIED_METADATA_INITIAL_STATE"
  "output_delay|STEMGENRT_QUALIFIED_MODEL_OUTPUT_DELAY_CHUNKS"
  "output_alignment|STEMGENRT_QUALIFIED_OUTPUT_ALIGNMENT"
  "causal_current_chunk|STEMGENRT_QUALIFIED_METADATA_CAUSAL_CURRENT_CHUNK"
  "external_overlap_add|STEMGENRT_QUALIFIED_METADATA_EXTERNAL_OVERLAP_ADD"
  "first_callback|STEMGENRT_QUALIFIED_METADATA_FIRST_CALLBACK"
  "flush|STEMGENRT_QUALIFIED_METADATA_FLUSH"
  "preroll|STEMGENRT_QUALIFIED_METADATA_PREROLL"
  "residual_source_index|STEMGENRT_QUALIFIED_RESIDUAL_SOURCE_INDEX"
  "other_source_index|STEMGENRT_QUALIFIED_OTHER_SOURCE_INDEX"
  "mixture_consistency|STEMGENRT_QUALIFIED_METADATA_MIXTURE_CONSISTENCY"
  "source_order|STEMGENRT_QUALIFIED_METADATA_SOURCE_ORDER"
  "model_sha256|STEMGENRT_QUALIFIED_MODEL_SHA256"
  "model_byte_size|STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE"
  "export_receipt_sha256|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SHA256"
  "export_receipt_schema_version|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SCHEMA_VERSION"
  "export_receipt_kind|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_KIND"
  "export_receipt_status|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_STATUS"
  "export_receipt_deployment_status|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_DEPLOYMENT_STATUS"
  "export_receipt_gate_pass|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_GATE_PASS"
  "export_receipt_cpu_only|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_CPU_ONLY"
  "export_receipt_promotion_performed|STEMGENRT_QUALIFIED_EXPORT_RECEIPT_PROMOTION_PERFORMED"
  "selected_update|STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE"
  "run_uuid|STEMGENRT_QUALIFIED_C236_RUN_UUID"
  "contract_identity_sha256|STEMGENRT_QUALIFIED_C236_CONTRACT_IDENTITY_SHA256"
  "static_identity_sha256|STEMGENRT_QUALIFIED_C236_STATIC_IDENTITY_SHA256"
  "source_checkpoint_sha256|STEMGENRT_QUALIFIED_C236_SOURCE_CHECKPOINT_SHA256"
  "materialized_artifact_sha256|STEMGENRT_QUALIFIED_C236_MATERIALIZED_ARTIFACT_SHA256"
  "materialization_receipt_sha256|STEMGENRT_QUALIFIED_C236_MATERIALIZATION_RECEIPT_SHA256"
  "model_state_sha256|STEMGENRT_QUALIFIED_C236_MODEL_STATE_SHA256"
  "composite_runtime_state_sha256|STEMGENRT_QUALIFIED_C236_COMPOSITE_RUNTIME_STATE_SHA256"
  "materializer_sha256|STEMGENRT_QUALIFIED_C236_MATERIALIZER_SHA256"
  "shared_authority_sha256|STEMGENRT_QUALIFIED_C236_SHARED_AUTHORITY_SHA256"
  "recovery_receipt_authority_sha256|STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_AUTHORITY_SHA256"
  "terminal_chain_authority_sha256|STEMGENRT_QUALIFIED_C236_TERMINAL_CHAIN_AUTHORITY_SHA256"
  "recovery_contract_sha256|STEMGENRT_QUALIFIED_C236_RECOVERY_CONTRACT_SHA256"
  "recovery_receipt_sha256|STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_SHA256"
  "final_chain_receipt_sha256|STEMGENRT_QUALIFIED_C236_FINAL_CHAIN_RECEIPT_SHA256"
  "qualification_v2_sha256|STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_SHA256"
  "qualification_v2_candidate_status|STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_CANDIDATE_STATUS"
  "evaluation_result_bound|STEMGENRT_QUALIFIED_C236_EVALUATION_RESULT_BOUND"
  "c191_payload_sha256|STEMGENRT_QUALIFIED_C191_PAYLOAD_SHA256"
  "c191_head_state_sha256|STEMGENRT_QUALIFIED_C191_HEAD_STATE_SHA256"
  "c236_family|STEMGENRT_QUALIFIED_C236_FAMILY"
  "candidate_kind|STEMGENRT_QUALIFIED_MODEL_CANDIDATE_KIND"
  "deployment_status|STEMGENRT_QUALIFIED_METADATA_DEPLOYMENT_STATUS"
  "external_data|STEMGENRT_QUALIFIED_METADATA_EXTERNAL_DATA"
  "header_dtype|export_dtype"
  "header_evaluation_bound|evaluation_result_bound"
  "header_previous_qualification|previous_qualification_transferred"
  "header_c191_qualification|c191_qualification_transferred"
  "header_opset|export_opset"
  "header_filename|export_filename"
  "header_future_context|future_context_samples"
  "header_residual_association|residual_association"
  "header_residual_name|residual_source_name"
  "header_composite_binding|composite_runtime_state_binding"
  "header_contract_identity_binding|contract_identity_binding"
  "header_static_identity_binding|static_identity_binding"
  "header_model_state_binding|model_state_binding"
  "header_materializer_binding|materializer_binding"
  "header_shared_authority_binding|shared_authority_binding"
  "header_recovery_receipt_authority_binding|recovery_receipt_authority_binding"
  "header_terminal_chain_authority_binding|terminal_chain_authority_binding"
  "header_recovery_contract_binding|recovery_contract_binding"
  "header_recovery_receipt_binding|recovery_receipt_binding"
  "header_final_chain_receipt_binding|final_chain_receipt_binding"
  "header_qualification_v2_binding|qualification_v2_binding"
  "header_qualification_status_binding|qualification_v2_candidate_status_binding"
  "header_family_binding|c236_family_binding"
  "header_candidate_binding|candidate_kind_binding"
  "header_artifact_binding|materialized_artifact_binding"
  "header_receipt_binding|materialization_receipt_binding"
  "header_checkpoint_binding|checkpoint_binding"
  "header_checkpoint_state_binding|checkpoint_state_binding"
  "header_source_checkpoint_binding|source_checkpoint_binding"
  "header_run_uuid_binding|run_uuid_binding"
  "header_c191_payload_binding|c191_payload_binding"
  "header_c191_head_binding|c191_head_state_binding"
  "header_duplicate_metadata_key|metadata_keys_unique"
  "header_embed_export_receipt|export_receipt_must_be_evidence_only"
  "runtime_dtype_guard|runtime_float32_guard"
  "runtime_name_guard|runtime_name_guard"
  "runtime_shape_guard|runtime_shape_guard"
  "runtime_metadata_guard|runtime_metadata_guard"
  "runtime_model_size_guard|runtime_model_size_guard"
  "runtime_model_sha_guard|runtime_model_sha_guard"
  "runtime_fusion_state_output|runtime_fusion_hidden_state"
  "runtime_fusion_state_reset|runtime_fusion_hidden_reset"
  "runtime_fusion_state_copy|runtime_fusion_hidden_copy"
  "output_residual_formula|output_exact_reconstruction")

set(mutation_root "$ENV{TMPDIR}")
if(mutation_root STREQUAL "")
  set(mutation_root "/tmp")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef mutation_suffix)
set(mutation_root
  "${mutation_root}/stemgenrt-c236-contract-mutations-${mutation_suffix}")
file(REMOVE_RECURSE "${mutation_root}")
file(MAKE_DIRECTORY "${mutation_root}")
set(c236_mutation_forward_arguments
  "-DC236_CONTRACT_FILE=${C236_CONTRACT_FILE}"
  "-DC236_CONTRACT_PHASE=${C236_CONTRACT_PHASE}")
foreach(c236_forward_variable IN ITEMS
    C236_MODEL_FILE
    C236_EXPORT_RECEIPT_FILE
    C236_EXPECTED_MODEL_SHA256
    C236_EXPECTED_MODEL_BYTE_SIZE
    C236_EXPECTED_EXPORT_RECEIPT_SHA256
    C236_EXPECTED_SELECTED_UPDATE
    C236_EXPECTED_RUN_UUID
    C236_EXPECTED_CONTRACT_IDENTITY_SHA256
    C236_EXPECTED_STATIC_IDENTITY_SHA256
    C236_EXPECTED_SOURCE_CHECKPOINT_SHA256
    C236_EXPECTED_MATERIALIZED_ARTIFACT_SHA256
    C236_EXPECTED_MATERIALIZATION_RECEIPT_SHA256
    C236_EXPECTED_MODEL_STATE_SHA256
    C236_EXPECTED_COMPOSITE_RUNTIME_STATE_SHA256
    C236_EXPECTED_RECOVERY_RECEIPT_SHA256
    C236_EXPECTED_FINAL_CHAIN_RECEIPT_SHA256
    C236_EXPECTED_QUALIFICATION_V2_SHA256
    C236_EXPECTED_QUALIFICATION_V2_CANDIDATE_STATUS)
  if(DEFINED ${c236_forward_variable} AND
     NOT "${${c236_forward_variable}}" STREQUAL "")
    list(APPEND c236_mutation_forward_arguments
      "-D${c236_forward_variable}=${${c236_forward_variable}}")
  endif()
endforeach()
set(mutation_count 0)
foreach(mutation_case IN LISTS mutation_cases)
  string(REPLACE "|" ";" mutation_parts "${mutation_case}")
  list(GET mutation_parts 0 mutation_id)
  list(GET mutation_parts 1 expected_tag)
  math(EXPR mutation_count "${mutation_count} + 1")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DC236_MUTATION_ID=${mutation_id}"
      "-DC236_MUTATION_SCRATCH=${mutation_root}/${mutation_id}"
      ${c236_mutation_forward_arguments}
      -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE mutation_result
    OUTPUT_VARIABLE mutation_output
    ERROR_VARIABLE mutation_error)
  set(mutation_log "${mutation_output}${mutation_error}")
  if(mutation_result EQUAL 0)
    file(REMOVE_RECURSE "${mutation_root}")
    message(FATAL_ERROR
      "Mutation '${mutation_id}' survived the c236 contract checks:\n${mutation_log}")
  endif()
  string(FIND "${mutation_log}" "C236_CONTRACT_REJECT:${expected_tag}"
    expected_failure_position)
  if(expected_failure_position EQUAL -1)
    file(REMOVE_RECURSE "${mutation_root}")
    message(FATAL_ERROR
      "Mutation '${mutation_id}' failed for the wrong reason; expected tag C236_CONTRACT_REJECT:${expected_tag}:\n${mutation_log}")
  endif()
endforeach()
file(REMOVE_RECURSE "${mutation_root}")

if(c236_contract_phase_final)
  message(STATUS
    "c236 final staged contract source/static checks passed; ${mutation_count} one-field mutations rejected")
else()
  message(STATUS
    "c236 draft contract source/static checks passed; ${mutation_count} one-field mutations rejected")
endif()
