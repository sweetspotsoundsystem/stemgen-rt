#include <StemgenRT/Constants.h>
#include <StemgenRT/QualifiedModelContract.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace {

using namespace std::literals;
namespace contract = audio_plugin::qualified_model;

constexpr std::string_view metadataValue(std::string_view key) {
  for (const auto& entry : contract::kMetadata) {
    if (entry.key == key) {
      return entry.value;
    }
  }
  return {};
}

constexpr bool metadataKeysAreUnique() {
  for (std::size_t lhs = 0; lhs < contract::kMetadata.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < contract::kMetadata.size(); ++rhs) {
      if (contract::kMetadata[lhs].key == contract::kMetadata[rhs].key) {
        return false;
      }
    }
  }
  return true;
}

constexpr auto kPendingSha256 =
    "0000000000000000000000000000000000000000000000000000000000000000"sv;

#ifndef C236_CONTRACT_PHASE_FINAL
#define C236_CONTRACT_PHASE_FINAL 0
#endif

static_assert(C236_CONTRACT_PHASE_FINAL == 0 ||
              C236_CONTRACT_PHASE_FINAL == 1);

constexpr bool isLowercaseSha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

constexpr bool isFinalSha256(std::string_view value) {
  return isLowercaseSha256(value) && value != kPendingSha256;
}

constexpr bool isSelectableUpdate(std::size_t update) {
  return update == 32768 || update == 65536 || update == 100000;
}

constexpr std::string_view selectedUpdateText(std::size_t update) {
  if (update == 0) {
    return "0"sv;
  }
  if (update == 32768) {
    return "32768"sv;
  }
  if (update == 65536) {
    return "65536"sv;
  }
  if (update == 100000) {
    return "100000"sv;
  }
  return {};
}

static_assert(audio_plugin::kOutputChunkSize == 256);
static_assert(audio_plugin::kPluginLatencySamples == 256);
static_assert(audio_plugin::kAnalysisWindowSize == 1024);
static_assert(audio_plugin::kAnalysisHistorySamples == 768);
static_assert(audio_plugin::kModelOutputDelayChunks == 0);
static_assert(audio_plugin::kAsyncQueueDelayChunks == 1);
static_assert(contract::kInputNames.size() == 4);
static_assert(contract::kOutputNames.size() == 4);
static_assert(contract::kMetadata.size() == 68);
static_assert(metadataKeysAreUnique(),
              "C236_CONTRACT_REJECT:metadata_keys_unique");

static_assert(contract::kContractId ==
                  "c236-terminal-selected-hop256-audition-v1",
              "C236_CONTRACT_REJECT:contract_id");
#if C236_CONTRACT_PHASE_FINAL
static_assert(isFinalSha256(contract::kModelSha256),
              "C236_CONTRACT_REJECT:model_sha256");
static_assert(contract::kModelByteSize > 1,
              "C236_CONTRACT_REJECT:model_byte_size");
static_assert(isFinalSha256(contract::kExportReceiptSha256),
              "C236_CONTRACT_REJECT:export_receipt_sha256");
#else
static_assert(contract::kModelSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:model_sha256");
static_assert(contract::kModelByteSize == 1,
              "C236_CONTRACT_REJECT:model_byte_size");
static_assert(contract::kExportReceiptSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:export_receipt_sha256");
#endif
static_assert(contract::kExportReceiptSchemaVersion == 3,
              "C236_CONTRACT_REJECT:export_receipt_schema_version");
static_assert(contract::kExportReceiptKind ==
                  "hs_tasnet_c236_recovery_checked_onnx_export_v3",
              "C236_CONTRACT_REJECT:export_receipt_kind");
static_assert(contract::kExportReceiptStatus == "pass",
              "C236_CONTRACT_REJECT:export_receipt_status");
static_assert(contract::kExportReceiptDeploymentStatus ==
                  "qualified_unpromoted_listening_candidate",
              "C236_CONTRACT_REJECT:export_receipt_deployment_status");
static_assert(contract::kExportReceiptGatePass == "true",
              "C236_CONTRACT_REJECT:export_receipt_gate_pass");
static_assert(contract::kExportReceiptCpuOnly == "true",
              "C236_CONTRACT_REJECT:export_receipt_cpu_only");
static_assert(contract::kExportReceiptPromotionPerformed == "false",
              "C236_CONTRACT_REJECT:export_receipt_promotion_performed");
static_assert(contract::kC236Family == "c236-c214-native-db-separator-v1",
              "C236_CONTRACT_REJECT:c236_family");
#if C236_CONTRACT_PHASE_FINAL
static_assert(isSelectableUpdate(contract::kC236SelectedUpdate),
              "C236_CONTRACT_REJECT:c236_selected_update");
static_assert(!contract::kC236RunUuid.empty() &&
                  contract::kC236RunUuid != "PENDING_C236_RUN_UUID",
              "C236_CONTRACT_REJECT:c236_run_uuid");
static_assert(isFinalSha256(contract::kC236ContractIdentitySha256),
              "C236_CONTRACT_REJECT:c236_contract_identity_sha256");
static_assert(isFinalSha256(contract::kC236StaticIdentitySha256),
              "C236_CONTRACT_REJECT:c236_static_identity_sha256");
static_assert(isFinalSha256(contract::kC236SourceCheckpointSha256),
              "C236_CONTRACT_REJECT:c236_source_checkpoint_sha256");
static_assert(isFinalSha256(contract::kC236MaterializedArtifactSha256),
              "C236_CONTRACT_REJECT:c236_materialized_artifact_sha256");
static_assert(isFinalSha256(contract::kC236MaterializationReceiptSha256),
              "C236_CONTRACT_REJECT:c236_materialization_receipt_sha256");
static_assert(isFinalSha256(contract::kC236ModelStateSha256),
              "C236_CONTRACT_REJECT:c236_model_state_sha256");
static_assert(isFinalSha256(contract::kC236CompositeRuntimeStateSha256),
              "C236_CONTRACT_REJECT:c236_composite_runtime_state_sha256");
#else
static_assert(contract::kC236SelectedUpdate == 0,
              "C236_CONTRACT_REJECT:c236_selected_update");
static_assert(contract::kC236RunUuid == "PENDING_C236_RUN_UUID",
              "C236_CONTRACT_REJECT:c236_run_uuid");
static_assert(contract::kC236ContractIdentitySha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_contract_identity_sha256");
static_assert(contract::kC236StaticIdentitySha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_static_identity_sha256");
static_assert(contract::kC236SourceCheckpointSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_source_checkpoint_sha256");
static_assert(contract::kC236MaterializedArtifactSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_materialized_artifact_sha256");
static_assert(contract::kC236MaterializationReceiptSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_materialization_receipt_sha256");
static_assert(contract::kC236ModelStateSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_model_state_sha256");
static_assert(contract::kC236CompositeRuntimeStateSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_composite_runtime_state_sha256");
#endif
static_assert(
    contract::kC236MaterializerSha256 ==
        "5da7c0d97faeabdfb87fbdf3d37e5a2af2847f8a1595fc0467b01a14ea59ca2d",
    "C236_CONTRACT_REJECT:c236_materializer_sha256");
static_assert(
    contract::kC236SharedAuthoritySha256 ==
        "6fc64af8968199e926f529ac9626f20c5f9fd6c28bcea02ae426353ae19246d2",
    "C236_CONTRACT_REJECT:c236_shared_authority_sha256");
static_assert(
    contract::kC236RecoveryReceiptAuthoritySha256 ==
        "fcb8edac6e94512d9be3b98d225541cf08458a3f3e015a139f36ac774bb76aa8",
    "C236_CONTRACT_REJECT:c236_recovery_receipt_authority_sha256");
static_assert(
    contract::kC236TerminalChainAuthoritySha256 ==
        "e8d23b5e78f36482e5f60b2e0d4b5b40d5cb91a242882f26a4f85f362a376cc9",
    "C236_CONTRACT_REJECT:c236_terminal_chain_authority_sha256");
static_assert(
    contract::kC236RecoveryContractSha256 ==
        "a1125772696b9f05d699c9de2f92caaa92cfaa1a58d74b80713ea6d021db4501",
    "C236_CONTRACT_REJECT:c236_recovery_contract_sha256");
#if C236_CONTRACT_PHASE_FINAL
static_assert(isFinalSha256(contract::kC236RecoveryReceiptSha256),
              "C236_CONTRACT_REJECT:c236_recovery_receipt_sha256");
static_assert(isFinalSha256(contract::kC236FinalChainReceiptSha256),
              "C236_CONTRACT_REJECT:c236_final_chain_receipt_sha256");
static_assert(isFinalSha256(contract::kC236QualificationV2Sha256),
              "C236_CONTRACT_REJECT:c236_qualification_v2_sha256");
static_assert(contract::kC236QualificationV2CandidateStatus ==
                  "qualified_by_declared_quality_budget",
              "C236_CONTRACT_REJECT:c236_qualification_v2_candidate_status");
static_assert(contract::kC236EvaluationResultBound == "true",
              "C236_CONTRACT_REJECT:c236_evaluation_result_bound");
#else
static_assert(contract::kC236RecoveryReceiptSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_recovery_receipt_sha256");
static_assert(contract::kC236FinalChainReceiptSha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_final_chain_receipt_sha256");
static_assert(contract::kC236QualificationV2Sha256 == kPendingSha256,
              "C236_CONTRACT_REJECT:c236_qualification_v2_sha256");
static_assert(contract::kC236QualificationV2CandidateStatus ==
                  "PENDING_C236_QUALIFICATION_V2_CANDIDATE_STATUS",
              "C236_CONTRACT_REJECT:c236_qualification_v2_candidate_status");
static_assert(contract::kC236EvaluationResultBound == "false",
              "C236_CONTRACT_REJECT:c236_evaluation_result_bound");
#endif
static_assert(
    contract::kC191PayloadSha256 ==
        "a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b",
    "C236_CONTRACT_REJECT:c191_payload_sha256");
static_assert(
    contract::kC191HeadStateSha256 ==
        "238991eced221734b5e931bbc4256412dd6ed361656f2cbe4864abccedd22bcf",
    "C236_CONTRACT_REJECT:c191_head_state_sha256");
static_assert(contract::kModelCandidateKind ==
                  "c236_terminal_selected_plus_exact_c191_hop256",
              "C236_CONTRACT_REJECT:model_candidate_kind");

static_assert(contract::kInputNames ==
                  std::array<std::string_view, 4>{
                      "audio_chunk", "analysis_history", "fusion_hidden",
                      "emitted_db_history"},
              "C236_CONTRACT_REJECT:input_names");
static_assert(contract::kOutputNames ==
                  std::array<std::string_view, 4>{
                      "separated_chunk", "next_analysis_history",
                      "next_fusion_hidden", "next_emitted_db_history"},
              "C236_CONTRACT_REJECT:output_names");
static_assert(contract::kInputAudioShape ==
                  std::array<std::int64_t, 3>{1, 2, 256},
              "C236_CONTRACT_REJECT:input_audio_shape");
static_assert(contract::kInputAnalysisHistoryShape ==
                  std::array<std::int64_t, 3>{1, 2, 768},
              "C236_CONTRACT_REJECT:input_analysis_history_shape");
static_assert(contract::kInputHiddenShape ==
                  std::array<std::int64_t, 3>{2, 1, 1000},
              "C236_CONTRACT_REJECT:input_hidden_shape");
static_assert(contract::kInputEmittedDbHistoryShape ==
                  std::array<std::int64_t, 3>{1, 4, 2048},
              "C236_CONTRACT_REJECT:input_emitted_db_history_shape");
static_assert(contract::kOutputSeparatedShape ==
                  std::array<std::int64_t, 4>{1, 4, 2, 256},
              "C236_CONTRACT_REJECT:output_separated_shape");
static_assert(contract::kOutputAnalysisHistoryShape ==
                  contract::kInputAnalysisHistoryShape,
              "C236_CONTRACT_REJECT:output_analysis_history_shape");
static_assert(contract::kOutputHiddenShape == contract::kInputHiddenShape,
              "C236_CONTRACT_REJECT:output_hidden_shape");
static_assert(contract::kOutputEmittedDbHistoryShape ==
                  contract::kInputEmittedDbHistoryShape,
              "C236_CONTRACT_REJECT:output_emitted_db_history_shape");
static_assert(contract::kResidualSourceIndex == 3,
              "C236_CONTRACT_REJECT:residual_source_index");
static_assert(contract::kOtherSourceIndex == 3,
              "C236_CONTRACT_REJECT:other_source_index");
static_assert(contract::kOutputAlignment == "current_input_chunk",
              "C236_CONTRACT_REJECT:output_alignment");
static_assert(contract::kStateNamesJson ==
                  "[\"analysis_history\",\"fusion_hidden\",\"emitted_db_history\"]",
              "C236_CONTRACT_REJECT:state_names");

static_assert(metadataValue("hs_tasnet.export.dtype") == "float32",
              "C236_CONTRACT_REJECT:export_dtype");
static_assert(metadataValue("hs_tasnet.export.input_names") ==
                  "[\"audio_chunk\",\"analysis_history\",\"fusion_hidden\",\"emitted_db_history\"]",
              "C236_CONTRACT_REJECT:metadata_input_names");
static_assert(metadataValue("hs_tasnet.export.output_names") ==
                  "[\"separated_chunk\",\"next_analysis_history\",\"next_fusion_hidden\",\"next_emitted_db_history\"]",
              "C236_CONTRACT_REJECT:metadata_output_names");
static_assert(metadataValue("hs_tasnet.export.input_shapes_batch1") ==
                  "[[1,2,256],[1,2,768],[2,1,1000],[1,4,2048]]",
              "C236_CONTRACT_REJECT:metadata_input_shapes");
static_assert(metadataValue("hs_tasnet.export.output_shapes_batch1") ==
                  "[[1,4,2,256],[1,2,768],[2,1,1000],[1,4,2048]]",
              "C236_CONTRACT_REJECT:metadata_output_shapes");
static_assert(metadataValue("hs_tasnet.streaming.state_count") == "3",
              "C236_CONTRACT_REJECT:state_count");
static_assert(metadataValue("hs_tasnet.streaming.state_names") ==
                  contract::kStateNamesJson,
              "C236_CONTRACT_REJECT:state_names");
static_assert(metadataValue("hs_tasnet.streaming.reset") ==
                  "zero_all_3_state_tensors",
              "C236_CONTRACT_REJECT:state_reset");
static_assert(metadataValue("hs_tasnet.streaming.chunk_samples") == "256",
              "C236_CONTRACT_REJECT:chunk_samples");
static_assert(metadataValue("hs_tasnet.streaming.pdc_samples") == "256",
              "C236_CONTRACT_REJECT:pdc_samples");
static_assert(metadataValue("hs_tasnet.streaming.output_delay_hops") == "0",
              "C236_CONTRACT_REJECT:output_delay_hops");
static_assert(metadataValue("hs_tasnet.streaming.output_alignment") ==
                  "current_input_chunk",
              "C236_CONTRACT_REJECT:output_alignment");
static_assert(metadataValue("hs_tasnet.streaming.future_context_samples") ==
                  "0",
              "C236_CONTRACT_REJECT:future_context_samples");
static_assert(metadataValue("hs_tasnet.streaming.first_callback") ==
                  "current_input_chunk",
              "C236_CONTRACT_REJECT:first_callback");
static_assert(metadataValue("hs_tasnet.streaming.flush") == "none",
              "C236_CONTRACT_REJECT:flush");
static_assert(metadataValue("hs_tasnet.streaming.preroll") == "none",
              "C236_CONTRACT_REJECT:preroll");
static_assert(metadataValue("hs_tasnet.streaming.external_overlap_add") ==
                  "false",
              "C236_CONTRACT_REJECT:external_overlap_add");
static_assert(metadataValue("hs_tasnet.export.mixture_consistency") ==
                  "exact_float32_residual_to_other",
              "C236_CONTRACT_REJECT:mixture_consistency");
static_assert(metadataValue("hs_tasnet.export.residual_association") ==
                  "audio_chunk-minus-sum-drums-bass-vocals-dim1-float32",
              "C236_CONTRACT_REJECT:residual_association");
static_assert(metadataValue("hs_tasnet.export.residual_source_index") == "3",
              "C236_CONTRACT_REJECT:residual_source_index");
static_assert(metadataValue("hs_tasnet.export.residual_source_name") ==
                  "other",
              "C236_CONTRACT_REJECT:residual_source_name");
static_assert(metadataValue("hs_tasnet.c236.evaluation_result_bound") ==
                  contract::kC236EvaluationResultBound,
              "C236_CONTRACT_REJECT:evaluation_result_bound");
static_assert(metadataValue("hs_tasnet.c236.recovery_receipt_authority_sha256") ==
                  contract::kC236RecoveryReceiptAuthoritySha256,
              "C236_CONTRACT_REJECT:recovery_receipt_authority_binding");
static_assert(metadataValue("hs_tasnet.c236.terminal_chain_authority_sha256") ==
                  contract::kC236TerminalChainAuthoritySha256,
              "C236_CONTRACT_REJECT:terminal_chain_authority_binding");
static_assert(metadataValue("hs_tasnet.c236.recovery_contract_sha256") ==
                  contract::kC236RecoveryContractSha256,
              "C236_CONTRACT_REJECT:recovery_contract_binding");
static_assert(metadataValue("hs_tasnet.c236.recovery_receipt_sha256") ==
                  contract::kC236RecoveryReceiptSha256,
              "C236_CONTRACT_REJECT:recovery_receipt_binding");
static_assert(metadataValue("hs_tasnet.c236.final_chain_receipt_sha256") ==
                  contract::kC236FinalChainReceiptSha256,
              "C236_CONTRACT_REJECT:final_chain_receipt_binding");
static_assert(metadataValue("hs_tasnet.c236.qualification_v2_sha256") ==
                  contract::kC236QualificationV2Sha256,
              "C236_CONTRACT_REJECT:qualification_v2_binding");
static_assert(
    metadataValue("hs_tasnet.c236.qualification_v2_candidate_status") ==
        contract::kC236QualificationV2CandidateStatus,
    "C236_CONTRACT_REJECT:qualification_v2_candidate_status_binding");
static_assert(metadataValue("hs_tasnet.c236.export_receipt_sha256").empty(),
              "C236_CONTRACT_REJECT:export_receipt_must_be_evidence_only");
static_assert(
    metadataValue("hs_tasnet.c236.previous_qualification_transferred") ==
        "false",
    "C236_CONTRACT_REJECT:previous_qualification_transferred");
static_assert(
    metadataValue("hs_tasnet.streaming.c191_qualification_transferred") ==
        "false",
    "C236_CONTRACT_REJECT:c191_qualification_transferred");
static_assert(metadataValue("hs_tasnet.deployment.status") ==
                  "post_training_export_candidate_not_plugin_qualified",
              "C236_CONTRACT_REJECT:deployment_status");
static_assert(metadataValue("hs_tasnet.export.mode") == "streaming",
              "C236_CONTRACT_REJECT:export_mode");
static_assert(metadataValue("hs_tasnet.export.opset_version") == "17",
              "C236_CONTRACT_REJECT:export_opset");
static_assert(metadataValue("hs_tasnet.export.output_filename") ==
                  "model.onnx",
              "C236_CONTRACT_REJECT:export_filename");
static_assert(metadataValue("hs_tasnet.export.external_data") == "false",
              "C236_CONTRACT_REJECT:external_data");
static_assert(metadataValue("hs_tasnet.c236.materialized_artifact_sha256") ==
                  contract::kC236MaterializedArtifactSha256,
              "C236_CONTRACT_REJECT:materialized_artifact_binding");
static_assert(
    metadataValue("hs_tasnet.c236.composite_runtime_state_sha256") ==
        contract::kC236CompositeRuntimeStateSha256,
    "C236_CONTRACT_REJECT:composite_runtime_state_binding");
static_assert(metadataValue("hs_tasnet.c236.contract_identity_sha256") ==
                  contract::kC236ContractIdentitySha256,
              "C236_CONTRACT_REJECT:contract_identity_binding");
static_assert(metadataValue("hs_tasnet.c236.static_identity_sha256") ==
                  contract::kC236StaticIdentitySha256,
              "C236_CONTRACT_REJECT:static_identity_binding");
static_assert(metadataValue("hs_tasnet.c236.model_state_sha256") ==
                  contract::kC236ModelStateSha256,
              "C236_CONTRACT_REJECT:model_state_binding");
static_assert(metadataValue("hs_tasnet.c236.materializer_sha256") ==
                  contract::kC236MaterializerSha256,
              "C236_CONTRACT_REJECT:materializer_binding");
static_assert(metadataValue("hs_tasnet.c236.shared_authority_sha256") ==
                  contract::kC236SharedAuthoritySha256,
              "C236_CONTRACT_REJECT:shared_authority_binding");
static_assert(metadataValue("hs_tasnet.c236.family") == contract::kC236Family,
              "C236_CONTRACT_REJECT:c236_family_binding");
static_assert(metadataValue("hs_tasnet.model.candidate_kind") ==
                  contract::kModelCandidateKind,
              "C236_CONTRACT_REJECT:candidate_kind_binding");
static_assert(
    metadataValue("hs_tasnet.c236.materialization_receipt_sha256") ==
        contract::kC236MaterializationReceiptSha256,
    "C236_CONTRACT_REJECT:materialization_receipt_binding");
static_assert(metadataValue("hs_tasnet.model.checkpoint_sha256") ==
                  contract::kC236MaterializedArtifactSha256,
              "C236_CONTRACT_REJECT:checkpoint_binding");
static_assert(metadataValue("hs_tasnet.model.checkpoint_state_sha256") ==
                  contract::kC236ModelStateSha256,
              "C236_CONTRACT_REJECT:checkpoint_state_binding");
static_assert(metadataValue("hs_tasnet.c236.source_checkpoint_sha256") ==
                  contract::kC236SourceCheckpointSha256,
              "C236_CONTRACT_REJECT:source_checkpoint_binding");
static_assert(metadataValue("hs_tasnet.c236.run_uuid") ==
                  contract::kC236RunUuid,
              "C236_CONTRACT_REJECT:run_uuid_binding");
static_assert(metadataValue("hs_tasnet.c236.selected_update") ==
                  selectedUpdateText(contract::kC236SelectedUpdate),
              "C236_CONTRACT_REJECT:selected_update_binding");
static_assert(metadataValue("hs_tasnet.c191.payload_sha256") ==
                  contract::kC191PayloadSha256,
              "C236_CONTRACT_REJECT:c191_payload_binding");
static_assert(metadataValue("hs_tasnet.c191.head_state_sha256") ==
                  contract::kC191HeadStateSha256,
              "C236_CONTRACT_REJECT:c191_head_state_binding");

static_assert(audio_plugin::kOrtAutomaticIntraOpThreadCap ==
#if defined(__APPLE__)
              2
#else
              4
#endif
);

}  // namespace
