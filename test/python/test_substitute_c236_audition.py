"""Destructive-path tests for the c236 audition substitution tool.

Every mutation in this module is confined to a caller-created synthetic
workspace.  The real detached worktree is only read so the tests can prove its
protected model/evidence bytes did not change.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "substitute-c236-audition.py"
ZERO_SHA256 = "0" * 64
MODEL_SHA_PLACEHOLDER = "model-sha-is-filled-after-bytes-are-written"

MATERIALIZER_SHA256 = (
    "5da7c0d97faeabdfb87fbdf3d37e5a2af2847f8a1595fc0467b01a14ea59ca2d"
)
SHARED_AUTHORITY_SHA256 = (
    "6fc64af8968199e926f529ac9626f20c5f9fd6c28bcea02ae426353ae19246d2"
)
RECOVERY_RECEIPT_AUTHORITY_SHA256 = (
    "fcb8edac6e94512d9be3b98d225541cf08458a3f3e015a139f36ac774bb76aa8"
)
TERMINAL_CHAIN_AUTHORITY_SHA256 = (
    "e8d23b5e78f36482e5f60b2e0d4b5b40d5cb91a242882f26a4f85f362a376cc9"
)
RECOVERY_CONTRACT_SHA256 = (
    "a1125772696b9f05d699c9de2f92caaa92cfaa1a58d74b80713ea6d021db4501"
)
C191_PAYLOAD_SHA256 = (
    "a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b"
)
C191_HEAD_STATE_SHA256 = (
    "238991eced221734b5e931bbc4256412dd6ed361656f2cbe4864abccedd22bcf"
)

RUN_UUID = "024ef8bb-c0f2-4663-ae60-03c4fc4a2b9e"
CONTRACT_IDENTITY_SHA256 = "a" * 64
STATIC_IDENTITY_SHA256 = "b" * 64
SOURCE_CHECKPOINT_SHA256 = "c" * 64
MATERIALIZED_ARTIFACT_SHA256 = "d" * 64
MODEL_STATE_SHA256 = "e" * 64
COMPOSITE_RUNTIME_STATE_SHA256 = "f" * 64
MATERIALIZATION_RECEIPT_SHA256 = "1" * 64
RECOVERY_RECEIPT_SHA256 = "2" * 64
QUALIFIED_STATUS = "qualified_by_declared_quality_budget"

QUARTET_NAMES = (
    "model.onnx",
    "model.onnx.sha256",
    "model.onnx.export.json",
    "model.onnx.export.json.sha256",
)
CREATION_BOUNDARIES = (
    "transaction_directory_created",
    "journal_body_published",
    "journal_published",
    "stage_directories_created",
    "stage_model_published",
    "stage_contract_published",
    "stage_receipt_sidecar_published",
    "stage_receipt_published",
)
ACTIVATION_BOUNDARIES = (
    "journal_prepared",
    "model_exchanged",
    "receipt_sidecar_published",
    "receipt_published",
    "legacy_candidate_retired",
    "legacy_sidecar_retired",
    "legacy_qualification_retired",
    "contract_activated",
    "final_verified",
)
CLEANUP_BOUNDARIES = (
    "cleanup_payload_removed",
    "cleanup_retired",
    "cleanup_sidecar_removed",
    "cleanup_journal_moved",
    "cleanup_retired_removed",
    "cleanup_complete",
)
MUTATION_BOUNDARIES = (
    *CREATION_BOUNDARIES,
    *ACTIVATION_BOUNDARIES,
    *CLEANUP_BOUNDARIES,
)
COMMITTED_BOUNDARIES = {"contract_activated", "final_verified", *CLEANUP_BOUNDARIES}
LEGACY_NAMES = (
    "FULL_CORRECTION_CANDIDATE.json",
    "FULL_CORRECTION_CANDIDATE.json.sha256",
    "FULL_CORRECTION_QUALIFICATION.md",
)
LIVE_PROTECTED = (ROOT / "model" / "model.onnx",) + tuple(
    ROOT / "model" / name for name in LEGACY_NAMES
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sidecar(subject: Path) -> Path:
    sidecar = Path(str(subject) + ".sha256")
    sidecar.write_text(
        f"{file_sha256(subject)}  {subject.name}\n", encoding="utf-8"
    )
    return sidecar


def write_json_pair(path: Path, value: Any) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    return path, write_sidecar(path)


def pair_identity(subject: Path) -> dict[str, Any]:
    sidecar = Path(str(subject) + ".sha256")
    return {
        "path": str(subject),
        "bytes": subject.stat().st_size,
        "sha256": file_sha256(subject),
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
    }


def protected_snapshot() -> dict[str, tuple[int, str]]:
    return {
        str(path): (path.stat().st_size, file_sha256(path))
        for path in LIVE_PROTECTED
    }


@pytest.fixture(scope="session", autouse=True)
def real_worktree_is_read_only() -> Iterator[None]:
    before = protected_snapshot()
    yield
    assert protected_snapshot() == before


def tree_snapshot(root: Path) -> dict[str, tuple[Any, ...]]:
    """Describe a fixture tree without following links or opening special files."""

    result: dict[str, tuple[Any, ...]] = {}
    if not os.path.lexists(root):
        return result
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted((*directories, *files)):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            mode = stat.S_IFMT(info.st_mode)
            if stat.S_ISREG(info.st_mode):
                result[relative] = (mode, info.st_nlink, file_sha256(path))
            elif stat.S_ISLNK(info.st_mode):
                result[relative] = (mode, os.readlink(path))
            else:
                result[relative] = (mode, info.st_rdev)
    return result


def expected_metadata(
    model_sha256: str,
    *,
    final_chain_sha256: str = "3" * 64,
    qualification_sha256: str = "4" * 64,
) -> dict[str, str]:
    """Independent oracle for the exact 68-entry final plugin metadata map."""

    return {
        "hs_tasnet.c191.emitted_db_history_shape": "[1,4,2048]",
        "hs_tasnet.c191.head_state_sha256": C191_HEAD_STATE_SHA256,
        "hs_tasnet.c191.partition_protocol": (
            "sample_domain_causal_same_weights_new_256_parent_requires_requalification"
        ),
        "hs_tasnet.c191.payload_sha256": C191_PAYLOAD_SHA256,
        "hs_tasnet.c236.composite_runtime_state_sha256": (
            COMPOSITE_RUNTIME_STATE_SHA256
        ),
        "hs_tasnet.c236.contract_identity_sha256": CONTRACT_IDENTITY_SHA256,
        "hs_tasnet.c236.evaluation_result_bound": "true",
        "hs_tasnet.c236.family": "c236-c214-native-db-separator-v1",
        "hs_tasnet.c236.final_chain_receipt_sha256": (
            final_chain_sha256
        ),
        "hs_tasnet.c236.materialization_receipt_sha256": (
            MATERIALIZATION_RECEIPT_SHA256
        ),
        "hs_tasnet.c236.materialized_artifact_sha256": (
            MATERIALIZED_ARTIFACT_SHA256
        ),
        "hs_tasnet.c236.materializer_sha256": MATERIALIZER_SHA256,
        "hs_tasnet.c236.materializer_verify_existing": "true",
        "hs_tasnet.c236.model_state_sha256": MODEL_STATE_SHA256,
        "hs_tasnet.c236.previous_qualification_transferred": "false",
        "hs_tasnet.c236.qualification_v2_candidate_status": QUALIFIED_STATUS,
        "hs_tasnet.c236.qualification_v2_sha256": qualification_sha256,
        "hs_tasnet.c236.recovery_contract_sha256": RECOVERY_CONTRACT_SHA256,
        "hs_tasnet.c236.recovery_receipt_authority_sha256": (
            RECOVERY_RECEIPT_AUTHORITY_SHA256
        ),
        "hs_tasnet.c236.recovery_receipt_sha256": RECOVERY_RECEIPT_SHA256,
        "hs_tasnet.c236.run_uuid": RUN_UUID,
        "hs_tasnet.c236.selected_update": "100000",
        "hs_tasnet.c236.shared_authority_sha256": SHARED_AUTHORITY_SHA256,
        "hs_tasnet.c236.source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "hs_tasnet.c236.static_identity_sha256": STATIC_IDENTITY_SHA256,
        "hs_tasnet.c236.terminal_chain_authority_sha256": (
            TERMINAL_CHAIN_AUTHORITY_SHA256
        ),
        "hs_tasnet.contract.schema_version": "1",
        "hs_tasnet.deployment.status": (
            "post_training_export_candidate_not_plugin_qualified"
        ),
        "hs_tasnet.export.dtype": "float32",
        "hs_tasnet.export.dynamo": "false",
        "hs_tasnet.export.external_data": "false",
        "hs_tasnet.export.input_names": (
            '["audio_chunk","analysis_history","fusion_hidden",'
            '"emitted_db_history"]'
        ),
        "hs_tasnet.export.input_shapes_batch1": (
            "[[1,2,256],[1,2,768],[2,1,1000],[1,4,2048]]"
        ),
        "hs_tasnet.export.mixture_consistency": (
            "exact_float32_residual_to_other"
        ),
        "hs_tasnet.export.mode": "streaming",
        "hs_tasnet.export.opset_version": "17",
        "hs_tasnet.export.output_filename": "model.onnx",
        "hs_tasnet.export.output_names": (
            '["separated_chunk","next_analysis_history",'
            '"next_fusion_hidden","next_emitted_db_history"]'
        ),
        "hs_tasnet.export.output_shapes_batch1": (
            "[[1,4,2,256],[1,2,768],[2,1,1000],[1,4,2048]]"
        ),
        "hs_tasnet.export.parity_semantics": "bounded_numeric_not_bit_exact",
        "hs_tasnet.export.residual_association": (
            "audio_chunk-minus-sum-drums-bass-vocals-dim1-float32"
        ),
        "hs_tasnet.export.residual_source_index": "3",
        "hs_tasnet.export.residual_source_name": "other",
        "hs_tasnet.model.candidate_kind": (
            "c236_terminal_selected_plus_exact_c191_hop256"
        ),
        "hs_tasnet.model.causal_current_chunk": "true",
        "hs_tasnet.model.checkpoint_sha256": MATERIALIZED_ARTIFACT_SHA256,
        "hs_tasnet.model.checkpoint_state_sha256": MODEL_STATE_SHA256,
        "hs_tasnet.model.source_order": '["drums","bass","vocals","other"]',
        "hs_tasnet.streaming.analysis_history_samples": "768",
        "hs_tasnet.streaming.analysis_history_shape": "[1,2,768]",
        "hs_tasnet.streaming.analysis_window_samples": "1024",
        "hs_tasnet.streaming.c191_correction": "true",
        "hs_tasnet.streaming.c191_qualification_transferred": "false",
        "hs_tasnet.streaming.chunk_samples": "256",
        "hs_tasnet.streaming.external_overlap_add": "false",
        "hs_tasnet.streaming.first_callback": "current_input_chunk",
        "hs_tasnet.streaming.flush": "none",
        "hs_tasnet.streaming.fusion_hidden_shape": "[2,1,1000]",
        "hs_tasnet.streaming.future_context_samples": "0",
        "hs_tasnet.streaming.initial_state": "all_positive_zero",
        "hs_tasnet.streaming.output_alignment": "current_input_chunk",
        "hs_tasnet.streaming.output_delay_hops": "0",
        "hs_tasnet.streaming.pdc_samples": "256",
        "hs_tasnet.streaming.preroll": "none",
        "hs_tasnet.streaming.reset": "zero_all_3_state_tensors",
        "hs_tasnet.streaming.sample_rate": "44100",
        "hs_tasnet.streaming.state_count": "3",
        "hs_tasnet.streaming.state_names": (
            '["analysis_history","fusion_hidden","emitted_db_history"]'
        ),
    }


def valid_receipt(
    model_bytes: bytes,
    *,
    attempt: Path,
    final_chain_identity: dict[str, Any],
    qualification_identity: dict[str, Any],
) -> dict[str, Any]:
    model_sha256 = sha256_bytes(model_bytes)
    final_chain_sha256 = str(final_chain_identity["sha256"])
    qualification_sha256 = str(qualification_identity["sha256"])
    metadata = expected_metadata(
        model_sha256,
        final_chain_sha256=final_chain_sha256,
        qualification_sha256=qualification_sha256,
    )
    return {
        "schema_version": 3,
        "kind": "hs_tasnet_c236_recovery_checked_onnx_export_v3",
        "status": "pass",
        "created_at_utc": "2026-08-11T00:00:00+00:00",
        "cpu_only": True,
        "deployment_status": "qualified_unpromoted_listening_candidate",
        "promotion_performed": False,
        "plugin_tree_touched": False,
        "training_executed": False,
        "output_name": "model.onnx",
        "export_sources": {},
        "terminal_authority": {
            "schema_version": 1,
            "kind": "hs_tasnet_c236_terminal_recovery_export_authority_v1",
            "selected_update": 100000,
            "terminal_run": {
                "terminal_recovery": {
                    "recovery_contract": {"sha256": RECOVERY_CONTRACT_SHA256},
                    "recovery_receipt": {"sha256": RECOVERY_RECEIPT_SHA256},
                    "semantic_terminal_assertion_correction_count": 1,
                    "replay_parity": {
                        "compared_update_count": 390,
                        "mismatch_count": 0,
                    },
                    "protected_audio_used": False,
                    "validation_opened": False,
                    "promotion_performed": False,
                }
            },
            "terminal_recovery_chain": {
                "schema_version": 1,
                "kind": (
                    "hs_tasnet_c236_terminal_recovery_export_chain_authority_v1"
                ),
                "status": "pass",
                "selected_update": 100000,
                "protected_audio_used_during_training": False,
                "validation_opened_only_by_qualification_v2": True,
                "promotion_performed": False,
                "final_chain_bundle": {
                    "receipt": dict(final_chain_identity)
                },
                "qualification_v2": {
                    **dict(qualification_identity),
                    "candidate_status": QUALIFIED_STATUS,
                },
            },
            "materialized_candidate": {
                "kind": "c236_terminal_selected_materialized_candidate_v1",
                "id": "c236-c214-native-db-separator-v1",
                "selected_update": 100000,
                "run_uuid": RUN_UUID,
                "contract_identity_sha256": CONTRACT_IDENTITY_SHA256,
                "static_identity_sha256": STATIC_IDENTITY_SHA256,
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "candidate_artifact": {
                    "path": str(
                        attempt.parent.parent / "materialized" / "candidate.pt"
                    ),
                    "sha256": MATERIALIZED_ARTIFACT_SHA256,
                },
                "materialization_receipt": {
                    "sha256": MATERIALIZATION_RECEIPT_SHA256
                },
                "validation_opened_before_evaluation": False,
                "promotion_performed": False,
                "protected_audio_used": False,
                "c191_payload_sha256": C191_PAYLOAD_SHA256,
                "c191_runtime_head_state_sha256": C191_HEAD_STATE_SHA256,
            },
        },
        "runtime_identity": {
            "materialized_file_sha256": MATERIALIZED_ARTIFACT_SHA256,
            "model_state_sha256": MODEL_STATE_SHA256,
            "composite_runtime_state_sha256": COMPOSITE_RUNTIME_STATE_SHA256,
            "c191_payload_sha256": C191_PAYLOAD_SHA256,
            "c191_head_state_sha256": C191_HEAD_STATE_SHA256,
            "c191_payload_restricted_single_read_sha256": C191_PAYLOAD_SHA256,
        },
        "export": {
            "output_path": str(attempt / "model.onnx"),
            "file_sha256": model_sha256,
            "bytes": len(model_bytes),
            "metadata": metadata,
            "qualified_publication_snapshot": {
                "path": str(attempt / "model.onnx"),
                "file_sha256": model_sha256,
                "bytes": len(model_bytes),
                "source_fd_retained_through_publication": True,
                "publication_source": "exact_qualified_inode_via_linux_linkat",
            },
            "final_native_dft_audit": {
                "required": True,
                "validated_from_exact_serialized_bytes": True,
                "raw_dft_count": 3,
                "forward_rfft_count": 2,
                "inverse_irfft_count": 1,
                "axis": 1,
                "n_fft": 1024,
                "forward_onesided": 1,
                "inverse_onesided": 0,
                "external_data": False,
            },
        },
        "qualification": {
            "schema_version": 2,
            "kind": "hs_tasnet_c236_64_hop_eager_onnx_qualification_v2",
            "parity_hops": 64,
            "normal_onnx_exact_residual_other_calls": 64,
            "normal_onnx_exact_residual_other_every_call": True,
            "same_hop_boundary_impulses_influence_current_dbv": True,
            "audio_chunk_reaches_separated_and_all_next_states": True,
            "every_state_live_eager": True,
            "every_state_live_onnx": True,
            "every_state_live_graph": True,
            "deterministic_positive_zero_reset": True,
            "all_outputs_and_states_finite": True,
            "eager_other_exact_float32_residual": True,
            "onnx_other_exact_float32_residual": True,
            "cross_backend_bit_exact": False,
            "bounded_numeric": True,
            "intra_backend_reset_replay_bit_exact": True,
            "provider": "CPUExecutionProvider",
            "onnxruntime_version": "1.26.0",
        },
        "parity_policy": {
            "semantics": "bounded_numeric_not_bit_exact",
            "bit_exact_claimed": False,
            "cross_backend_bit_exact": False,
            "bounded_numeric": True,
            "intra_backend_reset_replay_bit_exact": True,
        },
        "abi": {
            "sample_rate": 44100,
            "hop_samples": 256,
            "host_visible_latency_ms": 256 * 1000.0 / 44100,
            "host_visible_pdc_samples": 256,
            "analysis_window_samples": 1024,
            "analysis_history_samples": 768,
            "future_context_samples": 0,
            "flush_required": False,
            "input_names": [
                "audio_chunk",
                "analysis_history",
                "fusion_hidden",
                "emitted_db_history",
            ],
            "output_names": [
                "separated_chunk",
                "next_analysis_history",
                "next_fusion_hidden",
                "next_emitted_db_history",
            ],
            "input_shapes_batch1": [
                [1, 2, 256],
                [1, 2, 768],
                [2, 1, 1000],
                [1, 4, 2048],
            ],
            "output_shapes_batch1": [
                [1, 4, 2, 256],
                [1, 2, 768],
                [2, 1, 1000],
                [1, 4, 2048],
            ],
            "source_order": ["drums", "bass", "vocals", "other"],
            "residual_source": "other",
            "dtype": "float32",
        },
        "evaluator_coordination": {
            "shared_publication_authority_sha256": SHARED_AUTHORITY_SHA256,
            "recovery_receipt_authority_sha256": (
                RECOVERY_RECEIPT_AUTHORITY_SHA256
            ),
            "terminal_chain_authority_sha256": TERMINAL_CHAIN_AUTHORITY_SHA256,
            "materialization_receipt_sha256": MATERIALIZATION_RECEIPT_SHA256,
            "candidate_file_sha256": MATERIALIZED_ARTIFACT_SHA256,
            "candidate_model_state_sha256": MODEL_STATE_SHA256,
            "composite_runtime_state_sha256": COMPOSITE_RUNTIME_STATE_SHA256,
            "recovery_contract_sha256": RECOVERY_CONTRACT_SHA256,
            "recovery_receipt_sha256": RECOVERY_RECEIPT_SHA256,
            "semantic_terminal_assertion_correction_count": 1,
            "replay_compared_update_count": 390,
            "replay_mismatch_count": 0,
            "final_chain_receipt_sha256": final_chain_sha256,
            "qualification_v2_sha256": qualification_sha256,
            "qualification_v2_candidate_status": QUALIFIED_STATUS,
            "post_training_evaluation_result_bound": True,
            "production_promotion_requires_target_mac_evidence": True,
        },
        "gate_pass": True,
    }


@dataclass
class SyntheticFixture:
    root: Path
    workspace: Path
    completion: Path
    attempt: Path
    final_chain_receipt: Path
    qualification: Path
    model_bytes: bytes
    receipt: dict[str, Any]
    receipt_sha256: str
    final_chain_sha256: str
    qualification_sha256: str

    def write_receipt(self, *, refresh_sidecar: bool = True) -> str:
        receipt_path = self.attempt / "model.onnx.export.json"
        receipt_bytes = canonical_json_bytes(self.receipt)
        receipt_path.write_bytes(receipt_bytes)
        receipt_sha256 = sha256_bytes(receipt_bytes)
        if refresh_sidecar:
            (self.attempt / "model.onnx.export.json.sha256").write_text(
                f"{receipt_sha256}  model.onnx.export.json\n",
                encoding="utf-8",
            )
        self.receipt_sha256 = receipt_sha256
        return receipt_sha256

    def apply_args(self, *, include_fixture_flag: bool = True) -> list[str]:
        arguments = [
            "--apply",
            "--workspace",
            str(self.workspace),
            "--export-completion",
            str(self.completion),
            "--expected-export-receipt-sha256",
            self.receipt_sha256,
            "--expected-final-chain-receipt-sha256",
            self.final_chain_sha256,
        ]
        if include_fixture_flag:
            arguments.append("--allow-test-fixtures")
            arguments.append("--skip-static-contract-test")
        return arguments


def make_fixture(tmp_path: Path) -> SyntheticFixture:
    workspace = tmp_path / "synthetic-workspace"
    completion = workspace / "export-complete"
    attempt = workspace / "export-attempts" / "attempt-001"
    chain_root = workspace / "chain-authority"
    final_chain_receipt = chain_root / "chain-complete" / "receipt.json"
    qualification = chain_root / "attempt-001" / "qualification-v2-001.json"
    (workspace / "cmake").mkdir(parents=True)
    (workspace / "model").mkdir()
    (workspace / "test" / "cmake").mkdir(parents=True)
    (workspace / "test" / "source").mkdir(parents=True)
    completion.mkdir()
    attempt.mkdir(parents=True)
    (workspace / ".c236-synthetic-fixture").write_bytes(
        b"stemgenrt-c236-synthetic-fixture-v1\n"
    )

    for relative in (
        "CMakeLists.txt",
        "C236_AUDITION_DRAFT.md",
        "cmake/QualifiedModelContract.cmake",
        "cmake/QualifiedModelContract.h.in",
        "test/cmake/C236DraftContractTest.cmake",
        "test/source/C236ContractCompileOnly.cpp",
    ):
        source = ROOT / relative
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    (workspace / "model" / "model.onnx").write_bytes(
        b"synthetic legacy 512-hop payload; not an ONNX model\n"
    )
    for index, name in enumerate(LEGACY_NAMES):
        (workspace / "model" / name).write_bytes(
            f"synthetic legacy evidence {index}\n".encode()
        )

    model_bytes = (
        b"synthetic c236 256-hop payload; no parser may load this fixture\n"
        + bytes(range(64))
    )
    candidate_path = workspace / "materialized" / "candidate.pt"
    candidate_artifact = {
        "path": str(candidate_path),
        "sha256": MATERIALIZED_ARTIFACT_SHA256,
    }
    publication_authority = {
        "kind": "c236_terminal_selected_materialized_candidate_v1",
        "candidate_artifact": dict(candidate_artifact),
    }
    tradeoff_table = {
        "all_primary_quality_costs_within_allowance": True,
        "baseline": "full-c191-step128",
        "candidate": "candidate_corrected",
    }
    qualification_document = {
        "schema_version": 2,
        "kind": "hs_tasnet_c236_post_training_evaluation_v2",
        "status": "complete",
        "unavailable_gaps": [],
        "requested_scopes": ["full14", "electronic"],
        "completed_scopes": ["full14", "electronic"],
        "models": {
            "candidate_corrected": {
                "id": "c236-selected-trained-runtime",
                **dict(candidate_artifact),
                "publication_authority": publication_authority,
            },
            "c126": {"id": "c126-step100000"},
            "c91": {"id": "c91-step100000"},
            "c191": {"id": "full-c191-step128"},
        },
        "qualification_v2_validation": {
            "status": "pass",
            "fail_closed_structural_validation": True,
            "sir_was_used_as_veto": False,
            "candidate_status": QUALIFIED_STATUS,
            "quality_tradeoff_table": tradeoff_table,
        },
    }
    write_json_pair(qualification, qualification_document)
    qualification_identity = pair_identity(qualification)

    final_document = {
        "schema_version": 1,
        "kind": "hs_tasnet_c236_terminal_recovery_post_training_chain_v1",
        "status": "complete_stopped_before_export",
        "selected_update": 100000,
        "run_uuid": RUN_UUID,
        "protected_audio_used_during_training": False,
        "validation_opened_only_by_qualification_v2": True,
        "exclusive_gpu_lock": {
            "held_through_materialization_and_qualification": True
        },
        "deliberate_stop": {
            "onnx_export_started": False,
            "onnx_artifact_published": False,
            "plugin_repository_touched": False,
            "model_promoted": False,
            "branch_pushed": False,
            "listening_approval_claimed": False,
        },
        "qualification_v2": {
            **dict(qualification_identity),
            "candidate_status": QUALIFIED_STATUS,
            "quality_tradeoff_table": tradeoff_table,
            "required_scopes": ["full14", "electronic"],
            "required_models": ["candidate_corrected", "c126", "c91", "c191"],
        },
    }
    write_json_pair(final_chain_receipt, final_document)
    final_chain_identity = pair_identity(final_chain_receipt)

    (attempt / "model.onnx").write_bytes(model_bytes)
    model_sha256 = sha256_bytes(model_bytes)
    (attempt / "model.onnx.sha256").write_text(
        f"{model_sha256}  model.onnx\n", encoding="utf-8"
    )
    receipt = valid_receipt(
        model_bytes,
        attempt=attempt,
        final_chain_identity=final_chain_identity,
        qualification_identity=qualification_identity,
    )
    fixture = SyntheticFixture(
        root=tmp_path,
        workspace=workspace,
        completion=completion,
        attempt=attempt,
        final_chain_receipt=final_chain_receipt,
        qualification=qualification,
        model_bytes=model_bytes,
        receipt=receipt,
        receipt_sha256="",
        final_chain_sha256=str(final_chain_identity["sha256"]),
        qualification_sha256=str(qualification_identity["sha256"]),
    )
    fixture.write_receipt()

    completion_document = {
        "schema_version": 1,
        "kind": "hs_tasnet_c236_terminal_recovery_checked_export_completion_v1",
        "status": "complete_qualified_unpromoted",
        "candidate_status": QUALIFIED_STATUS,
        "cpu_only": True,
        "gpu_lock_acquired": False,
        "plugin_tree_touched": False,
        "promotion_performed": False,
        "git_operation_performed": False,
        "selected_update": 100000,
        "publication": {
            "directory": str(attempt),
            "model": pair_identity(attempt / "model.onnx"),
            "export_receipt": pair_identity(attempt / "model.onnx.export.json"),
            "chain_final_receipt_sha256": fixture.final_chain_sha256,
            "qualification_v2_sha256": fixture.qualification_sha256,
        },
        "chain_final_receipt": final_chain_identity,
        "qualification_v2": qualification_identity,
    }
    write_json_pair(completion / "receipt.json", completion_document)
    return fixture


@pytest.fixture
def synthetic(tmp_path: Path) -> SyntheticFixture:
    return make_fixture(tmp_path)


def fixture_environment(**updates: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "STEMGENRT_C236_SUBSTITUTION_TEST_MODE": "1",
        }
    )
    environment.update(updates)
    return environment


def run_tool(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def assert_rejected_without_workspace_mutation(
    fixture: SyntheticFixture,
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    before = tree_snapshot(fixture.workspace)
    result = run_tool(arguments, environment=environment, timeout=timeout)
    assert result.returncode != 0, result.stdout
    assert tree_snapshot(fixture.workspace) == before
    return result


def recover_args(fixture: SyntheticFixture) -> list[str]:
    return [
        "--recover",
        "--workspace",
        str(fixture.workspace),
        "--allow-test-fixtures",
    ]


def assert_pending_tree(
    fixture: SyntheticFixture, initial: dict[str, tuple[Any, ...]]
) -> None:
    assert tree_snapshot(fixture.workspace) == initial
    assert not (fixture.workspace / ".c236-substitution-txn").exists()
    assert not (fixture.workspace / ".c236-substitution-txn.retired").exists()


def assert_final_tree(fixture: SyntheticFixture) -> None:
    contract = (
        fixture.workspace / "cmake" / "QualifiedModelContract.cmake"
    ).read_bytes()
    assert ZERO_SHA256.encode() not in contract
    assert b"PENDING_C236" not in contract
    assert (fixture.workspace / "model" / "model.onnx").read_bytes() == (
        fixture.model_bytes
    )
    assert all(
        not os.path.lexists(fixture.workspace / "model" / name)
        for name in LEGACY_NAMES
    )
    assert (
        fixture.workspace / "model" / "C236_AUDITION_SUBSTITUTION.json"
    ).is_file()
    assert (
        fixture.workspace / "model" / "C236_AUDITION_SUBSTITUTION.json.sha256"
    ).is_file()
    assert not (fixture.workspace / ".c236-substitution-txn").exists()
    assert not (fixture.workspace / ".c236-substitution-txn.retired").exists()


def mutate_receipt(
    fixture: SyntheticFixture, mutator: Callable[[dict[str, Any]], None]
) -> None:
    fixture.receipt = copy.deepcopy(fixture.receipt)
    mutator(fixture.receipt)
    fixture.write_receipt()


def set_nested(value: dict[str, Any], path: str, replacement: Any) -> None:
    components = path.split(".")
    parent: Any = value
    for component in components[:-1]:
        parent = parent[component]
    parent[components[-1]] = replacement


def delete_nested(value: dict[str, Any], path: str) -> None:
    components = path.split(".")
    parent: Any = value
    for component in components[:-1]:
        parent = parent[component]
    del parent[components[-1]]


def test_metadata_oracle_has_exact_inventory_and_no_receipt_self_reference() -> None:
    metadata = expected_metadata(MODEL_SHA_PLACEHOLDER)
    assert len(metadata) == 68
    assert len(set(metadata)) == 68
    assert "hs_tasnet.c236.export_receipt_sha256" not in metadata


def test_no_arguments_is_an_inert_plan() -> None:
    before = protected_snapshot()
    result = run_tool([], environment=os.environ.copy())
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "prepared_source_only_inert"
    assert payload["default_opened_real_artifacts"] is False
    assert payload["default_mutated_workspace"] is False
    assert protected_snapshot() == before
    assert not (ROOT / ".c236-substitution-txn").exists()
    assert not (ROOT / ".c236-substitution.lock").exists()


def test_explicit_source_check_is_byte_inert(synthetic: SyntheticFixture) -> None:
    before = tree_snapshot(synthetic.workspace)
    result = run_tool(
        ["--source-check", "--workspace", str(synthetic.workspace)],
        environment=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "prepared_source_only_inert"
    assert payload["default_opened_real_artifacts"] is False
    assert payload["default_mutated_workspace"] is False
    assert tree_snapshot(synthetic.workspace) == before


def test_source_check_does_not_open_model_or_completion_artifacts(
    synthetic: SyntheticFixture,
) -> None:
    """FIFOs make an accidental artifact read block instead of passing silently."""

    for path in (
        synthetic.workspace / "model" / "model.onnx",
        *(synthetic.attempt / name for name in QUARTET_NAMES),
    ):
        path.unlink()
        os.mkfifo(path)
    before = tree_snapshot(synthetic.workspace)
    result = run_tool(
        ["--source-check", "--workspace", str(synthetic.workspace)],
        environment=os.environ.copy(),
        timeout=3.0,
    )
    assert result.returncode == 0, result.stderr
    assert tree_snapshot(synthetic.workspace) == before


@pytest.mark.parametrize("anchor", ["neither", "export_only", "chain_only"])
def test_apply_requires_both_explicit_trust_anchors(
    synthetic: SyntheticFixture, anchor: str
) -> None:
    arguments = [
        "--apply",
        "--workspace",
        str(synthetic.workspace),
        "--export-completion",
        str(synthetic.completion),
        "--allow-test-fixtures",
    ]
    if anchor == "export_only":
        arguments.extend(
            ["--expected-export-receipt-sha256", synthetic.receipt_sha256]
        )
    if anchor == "chain_only":
        arguments.extend(
            [
                "--expected-final-chain-receipt-sha256",
                synthetic.final_chain_sha256,
            ]
        )
    assert_rejected_without_workspace_mutation(
        synthetic, arguments, environment=fixture_environment()
    )


@pytest.mark.parametrize("which", ["export", "final_chain"])
def test_mismatched_trust_anchor_fails_before_mutation(
    synthetic: SyntheticFixture, which: str
) -> None:
    arguments = synthetic.apply_args()
    flag = (
        "--expected-export-receipt-sha256"
        if which == "export"
        else "--expected-final-chain-receipt-sha256"
    )
    arguments[arguments.index(flag) + 1] = "9" * 64
    assert_rejected_without_workspace_mutation(
        synthetic, arguments, environment=fixture_environment()
    )


def test_fixture_mode_requires_both_cli_opt_in_and_marker(
    synthetic: SyntheticFixture,
) -> None:
    assert_rejected_without_workspace_mutation(
        synthetic,
        synthetic.apply_args(include_fixture_flag=False),
        environment=fixture_environment(),
    )
    marker = synthetic.workspace / ".c236-synthetic-fixture"
    marker.unlink()
    assert_rejected_without_workspace_mutation(
        synthetic,
        synthetic.apply_args(),
        environment=fixture_environment(),
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("schema_version", 2),
        ("kind", "hs_tasnet_c236_checked_onnx_export_v2"),
        ("status", "fail"),
        ("cpu_only", False),
        ("deployment_status", "production_promoted"),
        ("promotion_performed", True),
        ("gate_pass", False),
        ("export.file_sha256", "5" * 64),
        ("export.bytes", 1),
        ("export.qualified_publication_snapshot.file_sha256", "5" * 64),
        ("export.qualified_publication_snapshot.bytes", 1),
        ("abi.hop_samples", 512),
        ("abi.host_visible_pdc_samples", 512),
        ("abi.analysis_window_samples", 512),
        ("abi.flush_required", True),
        ("qualification.schema_version", 1),
        ("qualification.parity_hops", 63),
        ("qualification.normal_onnx_exact_residual_other_every_call", False),
        ("qualification.every_state_live_onnx", False),
        ("qualification.bounded_numeric", False),
        (
            "terminal_authority.terminal_run.terminal_recovery."
            "recovery_contract.sha256",
            "5" * 64,
        ),
        (
            "terminal_authority.terminal_recovery_chain.final_chain_bundle."
            "receipt.sha256",
            "5" * 64,
        ),
        (
            "terminal_authority.terminal_recovery_chain.qualification_v2.sha256",
            ZERO_SHA256,
        ),
        (
            "terminal_authority.terminal_recovery_chain.qualification_v2."
            "candidate_status",
            "rejected",
        ),
        ("evaluator_coordination.final_chain_receipt_sha256", "5" * 64),
        ("evaluator_coordination.qualification_v2_sha256", "5" * 64),
        (
            "evaluator_coordination.post_training_evaluation_result_bound",
            False,
        ),
    ],
    ids=lambda value: str(value).replace(".", "_")[:80],
)
def test_receipt_semantic_mutations_are_rejected_before_workspace_mutation(
    synthetic: SyntheticFixture, path: str, replacement: Any
) -> None:
    mutate_receipt(
        synthetic, lambda receipt: set_nested(receipt, path, replacement)
    )
    assert_rejected_without_workspace_mutation(
        synthetic, synthetic.apply_args(), environment=fixture_environment()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_metadata",
        "extra_metadata",
        "wrong_metadata_value",
        "receipt_self_reference",
    ],
)
def test_metadata_inventory_must_match_exactly(
    synthetic: SyntheticFixture, mutation: str
) -> None:
    def alter(receipt: dict[str, Any]) -> None:
        metadata = receipt["export"]["metadata"]
        if mutation == "missing_metadata":
            del metadata["hs_tasnet.streaming.chunk_samples"]
        elif mutation == "extra_metadata":
            metadata["hs_tasnet.synthetic.extra"] = "forbidden"
        elif mutation == "wrong_metadata_value":
            metadata["hs_tasnet.streaming.chunk_samples"] = "512"
        else:
            metadata["hs_tasnet.c236.export_receipt_sha256"] = "5" * 64

    mutate_receipt(synthetic, alter)
    assert_rejected_without_workspace_mutation(
        synthetic, synthetic.apply_args(), environment=fixture_environment()
    )


@pytest.mark.parametrize(
    "missing_path",
    [
        "terminal_authority.terminal_recovery_chain.qualification_v2",
        "export.metadata",
        "export.qualified_publication_snapshot",
        "evaluator_coordination.materialization_receipt_sha256",
        "qualification",
        "abi",
    ],
)
def test_required_receipt_sections_cannot_be_omitted(
    synthetic: SyntheticFixture, missing_path: str
) -> None:
    mutate_receipt(
        synthetic, lambda receipt: delete_nested(receipt, missing_path)
    )
    assert_rejected_without_workspace_mutation(
        synthetic, synthetic.apply_args(), environment=fixture_environment()
    )


@pytest.mark.parametrize("sidecar", ["model", "receipt"])
@pytest.mark.parametrize("mutation", ["digest", "basename", "grammar"])
def test_sidecars_are_strict_and_independently_verified(
    synthetic: SyntheticFixture, sidecar: str, mutation: str
) -> None:
    path = synthetic.attempt / (
        "model.onnx.sha256"
        if sidecar == "model"
        else "model.onnx.export.json.sha256"
    )
    digest = (
        sha256_bytes(synthetic.model_bytes)
        if sidecar == "model"
        else synthetic.receipt_sha256
    )
    basename = (
        "model.onnx" if sidecar == "model" else "model.onnx.export.json"
    )
    if mutation == "digest":
        path.write_text(f"{'9' * 64}  {basename}\n", encoding="utf-8")
    elif mutation == "basename":
        path.write_text(f"{digest}  wrong-name\n", encoding="utf-8")
    else:
        path.write_text(f"SHA256 ({basename}) = {digest}\n", encoding="utf-8")
    assert_rejected_without_workspace_mutation(
        synthetic, synthetic.apply_args(), environment=fixture_environment()
    )


def test_noncanonical_receipt_is_rejected_even_with_matching_hashes(
    synthetic: SyntheticFixture,
) -> None:
    receipt_path = synthetic.attempt / "model.onnx.export.json"
    noncanonical = json.dumps(synthetic.receipt, sort_keys=False).encode("utf-8")
    receipt_path.write_bytes(noncanonical)
    synthetic.receipt_sha256 = sha256_bytes(noncanonical)
    (synthetic.attempt / "model.onnx.export.json.sha256").write_text(
        f"{synthetic.receipt_sha256}  model.onnx.export.json\n",
        encoding="utf-8",
    )
    assert_rejected_without_workspace_mutation(
        synthetic, synthetic.apply_args(), environment=fixture_environment()
    )


def test_duplicate_json_keys_are_rejected_even_with_matching_receipt_anchor(
    synthetic: SyntheticFixture,
) -> None:
    receipt_path = synthetic.attempt / "model.onnx.export.json"
    original = receipt_path.read_bytes()
    duplicate = original.replace(
        b'  "status": "pass",\n',
        b'  "status": "pass",\n  "status": "pass",\n',
        1,
    )
    assert duplicate != original
    receipt_path.write_bytes(duplicate)
    synthetic.receipt_sha256 = sha256_bytes(duplicate)
    (synthetic.attempt / "model.onnx.export.json.sha256").write_text(
        f"{synthetic.receipt_sha256}  model.onnx.export.json\n",
        encoding="utf-8",
    )
    assert_rejected_without_workspace_mutation(
        synthetic, synthetic.apply_args(), environment=fixture_environment()
    )


def test_model_bytes_must_match_both_receipt_and_sidecar(
    synthetic: SyntheticFixture,
) -> None:
    (synthetic.attempt / "model.onnx").write_bytes(
        synthetic.model_bytes + b"mutated"
    )
    assert_rejected_without_workspace_mutation(
        synthetic, synthetic.apply_args(), environment=fixture_environment()
    )


@pytest.mark.parametrize("extra_name", ["model.onnx.data", "unexpected.txt"])
def test_completion_inventory_is_exact(
    synthetic: SyntheticFixture, extra_name: str
) -> None:
    (synthetic.attempt / extra_name).write_bytes(b"forbidden extra")
    assert_rejected_without_workspace_mutation(
        synthetic, synthetic.apply_args(), environment=fixture_environment()
    )


@pytest.mark.parametrize("name", QUARTET_NAMES)
@pytest.mark.parametrize("file_kind", ["symlink", "hardlink", "fifo"])
def test_every_completion_member_must_be_a_single_link_regular_file(
    synthetic: SyntheticFixture, name: str, file_kind: str
) -> None:
    path = synthetic.attempt / name
    backing = synthetic.root / f"{name}.{file_kind}.backing"
    if file_kind == "symlink":
        shutil.copy2(path, backing)
        path.unlink()
        path.symlink_to(backing)
    elif file_kind == "hardlink":
        os.link(path, backing)
    else:
        path.unlink()
        os.mkfifo(path)
    assert_rejected_without_workspace_mutation(
        synthetic,
        synthetic.apply_args(),
        environment=fixture_environment(),
        timeout=5.0,
    )


def test_valid_synthetic_apply_is_atomic_and_retires_legacy_evidence(
    synthetic: SyntheticFixture,
) -> None:
    completion_before = tree_snapshot(synthetic.completion)
    attempt_before = tree_snapshot(synthetic.attempt)
    result = run_tool(
        synthetic.apply_args(), environment=fixture_environment(), timeout=30.0
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "substituted_unpromoted_audition_candidate"
    assert (synthetic.workspace / "model" / "model.onnx").read_bytes() == (
        synthetic.model_bytes
    )
    assert all(
        not os.path.lexists(synthetic.workspace / "model" / name)
        for name in LEGACY_NAMES
    )
    assert tree_snapshot(synthetic.completion) == completion_before
    assert tree_snapshot(synthetic.attempt) == attempt_before
    report = synthetic.workspace / "model" / "C236_AUDITION_SUBSTITUTION.json"
    report_sidecar = Path(str(report) + ".sha256")
    assert report.is_file() and not report.is_symlink()
    assert report.stat().st_nlink == 1
    assert report_sidecar.read_text(encoding="utf-8") == (
        f"{file_sha256(report)}  C236_AUDITION_SUBSTITUTION.json\n"
    )
    report_payload = json.loads(report.read_bytes())
    assert report.read_bytes() == canonical_json_bytes(report_payload)
    assert (
        report_payload["checked_export"]["model"]["sha256"]
        == sha256_bytes(synthetic.model_bytes)
    )
    source_anchor = report_payload.get(
        "source_anchor", report_payload.get("trust_anchors")
    )
    assert source_anchor["expected_export_receipt_sha256"] == (
        synthetic.receipt_sha256
    )
    assert (
        source_anchor["expected_final_chain_receipt_sha256"]
        == synthetic.final_chain_sha256
    )
    assert report_payload["qualification"]["final_chain_receipt"]["sha256"] == (
        synthetic.final_chain_sha256
    )
    assert not (synthetic.workspace / ".c236-substitution-txn").exists()
    assert not (synthetic.workspace / ".c236-substitution.lock").exists()

    contract = (
        synthetic.workspace / "cmake" / "QualifiedModelContract.cmake"
    ).read_text(encoding="utf-8")
    for value in (
        sha256_bytes(synthetic.model_bytes),
        synthetic.receipt_sha256,
        synthetic.final_chain_sha256,
        synthetic.qualification_sha256,
        QUALIFIED_STATUS,
        RUN_UUID,
    ):
        assert value in contract
    assert 'set(STEMGENRT_QUALIFIED_C236_EVALUATION_RESULT_BOUND "true")' in contract


def test_exact_final_tree_replay_is_read_only_and_reauthenticates_receipt(
    synthetic: SyntheticFixture,
) -> None:
    first = run_tool(
        synthetic.apply_args(), environment=fixture_environment(), timeout=30.0
    )
    assert first.returncode == 0, first.stderr
    before = tree_snapshot(synthetic.workspace)
    second = run_tool(
        synthetic.apply_args(), environment=fixture_environment(), timeout=30.0
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["status"] == "already_applied"
    assert tree_snapshot(synthetic.workspace) == before


def test_final_tree_replay_rejects_rewritten_external_receipt(
    synthetic: SyntheticFixture,
) -> None:
    first = run_tool(
        synthetic.apply_args(), environment=fixture_environment(), timeout=30.0
    )
    assert first.returncode == 0, first.stderr
    report = synthetic.workspace / "model" / "C236_AUDITION_SUBSTITUTION.json"
    payload = json.loads(report.read_bytes())
    payload["scope"]["listening_approval_required"] = False
    report.chmod(0o644)
    report.write_bytes(canonical_json_bytes(payload))
    report_sidecar = Path(str(report) + ".sha256")
    report_sidecar.chmod(0o644)
    report_sidecar.write_text(
        f"{file_sha256(report)}  C236_AUDITION_SUBSTITUTION.json\n",
        encoding="utf-8",
    )
    before = tree_snapshot(synthetic.workspace)
    replay = run_tool(
        synthetic.apply_args(), environment=fixture_environment(), timeout=30.0
    )
    assert replay.returncode != 0
    assert "existing substitution receipt changed" in replay.stderr
    assert tree_snapshot(synthetic.workspace) == before


def test_final_tree_replay_rejects_an_altered_active_contract(
    synthetic: SyntheticFixture,
) -> None:
    first = run_tool(
        synthetic.apply_args(), environment=fixture_environment(), timeout=30.0
    )
    assert first.returncode == 0, first.stderr
    contract = synthetic.workspace / "cmake" / "QualifiedModelContract.cmake"
    contract.write_bytes(contract.read_bytes() + b"\nmessage(FATAL_ERROR \"tampered\")\n")
    before = tree_snapshot(synthetic.workspace)
    replay = run_tool(
        synthetic.apply_args(), environment=fixture_environment(), timeout=30.0
    )
    assert replay.returncode != 0
    assert "existing substitution receipt changed" in replay.stderr
    assert tree_snapshot(synthetic.workspace) == before


@pytest.mark.parametrize(
    "relative",
    [
        "cmake/QualifiedModelContract.cmake",
        "model/model.onnx",
        "model/FULL_CORRECTION_CANDIDATE.json",
        "model/FULL_CORRECTION_CANDIDATE.json.sha256",
        "model/FULL_CORRECTION_QUALIFICATION.md",
    ],
)
def test_uncooperative_baseline_replacement_is_detected_before_mutation(
    synthetic: SyntheticFixture, relative: str
) -> None:
    initial = tree_snapshot(synthetic.workspace)
    target = synthetic.workspace / relative
    original = target.read_bytes()
    original_mode = stat.S_IMODE(target.stat().st_mode)
    first = subprocess.Popen(
        [sys.executable, str(TOOL), *synthetic.apply_args()],
        cwd=ROOT,
        env=fixture_environment(STEMGENRT_C236_PAUSE_AFTER="journal_prepared"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paused = synthetic.workspace / ".c236-substitution-test-paused"
    resume = synthetic.workspace / ".c236-substitution-test-resume"
    deadline = time.monotonic() + 10.0
    while not paused.exists() and first.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert paused.read_text(encoding="utf-8") == "journal_prepared\n"
        attacker = target.with_name(target.name + ".attacker")
        attacker.write_bytes(b"uncooperative replacement must never be exchanged\n")
        attacker.chmod(original_mode)
        os.replace(attacker, target)
        resume.write_text("resume\n", encoding="utf-8")
        _stdout, stderr = first.communicate(timeout=30.0)
        assert first.returncode != 0, stderr
        assert target.read_bytes().startswith(b"uncooperative replacement")
        assert (synthetic.workspace / ".c236-substitution-txn").is_dir()

        restored = target.with_name(target.name + ".restored")
        restored.write_bytes(original)
        restored.chmod(original_mode)
        os.replace(restored, target)
        recovered = run_tool(
            recover_args(synthetic), environment=fixture_environment(), timeout=30.0
        )
        assert recovered.returncode == 0, recovered.stderr
        assert json.loads(recovered.stdout)["status"] == "recovered_pending"
        assert_pending_tree(synthetic, initial)
    finally:
        if first.poll() is None:
            resume.write_text("resume\n", encoding="utf-8")
            first.kill()
            first.wait(timeout=5.0)


@pytest.mark.parametrize("boundary", MUTATION_BOUNDARIES)
def test_ordinary_failure_reaches_a_verified_terminal_state(
    synthetic: SyntheticFixture, boundary: str
) -> None:
    """Every caught failure restores the exact pending tree, even post-exchange."""

    initial = tree_snapshot(synthetic.workspace)
    result = run_tool(
        synthetic.apply_args(),
        environment=fixture_environment(STEMGENRT_C236_FAIL_AFTER=boundary),
        timeout=30.0,
    )
    assert result.returncode != 0
    assert not (synthetic.workspace / ".c236-substitution-txn").exists()
    assert not (synthetic.workspace / ".c236-substitution-txn.retired").exists()
    assert not (
        synthetic.workspace / ".c236-substitution-cleanup-journal.json"
    ).exists()
    if boundary in CLEANUP_BOUNDARIES:
        assert_final_tree(synthetic)
    else:
        assert_pending_tree(synthetic, initial)


@pytest.mark.parametrize("boundary", MUTATION_BOUNDARIES)
def test_sigkill_at_every_mutation_boundary_requires_explicit_hash_recovery(
    synthetic: SyntheticFixture, boundary: str
) -> None:
    initial = tree_snapshot(synthetic.workspace)
    killed = run_tool(
        synthetic.apply_args(),
        environment=fixture_environment(STEMGENRT_C236_KILL_AFTER=boundary),
        timeout=30.0,
    )
    assert killed.returncode == -signal.SIGKILL
    if boundary == "cleanup_complete":
        assert_final_tree(synthetic)
        return
    assert (
        (synthetic.workspace / ".c236-substitution-txn").is_dir()
        or (synthetic.workspace / ".c236-substitution-txn.retired").is_dir()
        or (
            synthetic.workspace / ".c236-substitution-cleanup-journal.json"
        ).is_file()
    )

    recovered = run_tool(
        recover_args(synthetic), environment=fixture_environment(), timeout=30.0
    )
    assert recovered.returncode == 0, recovered.stderr
    payload = json.loads(recovered.stdout)
    if boundary in COMMITTED_BOUNDARIES:
        assert payload["status"] == "recovered_final"
        assert_final_tree(synthetic)
    else:
        assert payload["status"] == "recovered_pending"
        assert_pending_tree(synthetic, initial)


def test_recover_refuses_unknown_live_bytes_without_cleanup(
    synthetic: SyntheticFixture,
) -> None:
    killed = run_tool(
        synthetic.apply_args(),
        environment=fixture_environment(
            STEMGENRT_C236_KILL_AFTER="model_exchanged"
        ),
        timeout=30.0,
    )
    assert killed.returncode == -signal.SIGKILL
    live_model = synthetic.workspace / "model" / "model.onnx"
    live_model.write_bytes(b"unknown bytes injected after crash\n")
    unknown = tree_snapshot(synthetic.workspace)

    recovered = run_tool(
        recover_args(synthetic), environment=fixture_environment(), timeout=30.0
    )
    assert recovered.returncode != 0
    assert "unknown live model hash" in recovered.stderr
    assert tree_snapshot(synthetic.workspace) == unknown
    assert (synthetic.workspace / ".c236-substitution-txn").is_dir()


@pytest.mark.parametrize(
    ("boundary", "relative"),
    [
        ("journal_prepared", ".c236-substitution-txn/unexpected"),
        ("journal_prepared", ".c236-substitution-txn/stage/unexpected"),
        (
            "legacy_candidate_retired",
            ".c236-substitution-txn/backup/unexpected",
        ),
        ("cleanup_retired", ".c236-substitution-txn.retired/unexpected"),
    ],
)
def test_recovery_never_deletes_unknown_transaction_inventory(
    synthetic: SyntheticFixture, boundary: str, relative: str
) -> None:
    killed = run_tool(
        synthetic.apply_args(),
        environment=fixture_environment(STEMGENRT_C236_KILL_AFTER=boundary),
        timeout=30.0,
    )
    assert killed.returncode == -signal.SIGKILL
    unexpected = synthetic.workspace / relative
    unexpected.write_bytes(b"unknown transaction member must be preserved\n")
    before = tree_snapshot(synthetic.workspace)
    recovered = run_tool(
        recover_args(synthetic), environment=fixture_environment(), timeout=30.0
    )
    assert recovered.returncode != 0
    assert "inventory changed" in recovered.stderr
    assert tree_snapshot(synthetic.workspace) == before


def test_recovery_never_deletes_an_unknown_cleanup_marker(
    synthetic: SyntheticFixture,
) -> None:
    killed = run_tool(
        synthetic.apply_args(),
        environment=fixture_environment(
            STEMGENRT_C236_KILL_AFTER="cleanup_journal_moved"
        ),
        timeout=30.0,
    )
    assert killed.returncode == -signal.SIGKILL
    marker = (
        synthetic.workspace / ".c236-substitution-cleanup-journal.json"
    )
    marker.chmod(0o644)
    marker.write_bytes(b"unknown cleanup authority must be preserved\n")
    before = tree_snapshot(synthetic.workspace)

    recovered = run_tool(
        recover_args(synthetic), environment=fixture_environment(), timeout=30.0
    )
    assert recovered.returncode != 0
    assert "invalid JSON in cleanup journal marker" in recovered.stderr
    assert tree_snapshot(synthetic.workspace) == before


@pytest.mark.parametrize(
    "boundary", ["transaction_directory_created", "journal_prepared", "contract_activated"]
)
def test_recovery_never_discards_authority_with_unknown_live_model_inventory(
    synthetic: SyntheticFixture, boundary: str
) -> None:
    killed = run_tool(
        synthetic.apply_args(),
        environment=fixture_environment(STEMGENRT_C236_KILL_AFTER=boundary),
        timeout=30.0,
    )
    assert killed.returncode == -signal.SIGKILL
    unexpected = synthetic.workspace / "model" / "unexpected-live-member"
    unexpected.write_bytes(b"unknown live model evidence must be preserved\n")
    before = tree_snapshot(synthetic.workspace)

    recovered = run_tool(
        recover_args(synthetic), environment=fixture_environment(), timeout=30.0
    )
    assert recovered.returncode != 0
    assert tree_snapshot(synthetic.workspace) == before
    assert (
        (synthetic.workspace / ".c236-substitution-txn").exists()
        or (synthetic.workspace / ".c236-substitution-txn.retired").exists()
        or (
            synthetic.workspace / ".c236-substitution-cleanup-journal.json"
        ).exists()
    )


@pytest.mark.parametrize("injection", ["live_model", "retired_tree"])
def test_cleanup_revalidation_preserves_authority_against_late_unknown_entries(
    synthetic: SyntheticFixture, injection: str
) -> None:
    first = subprocess.Popen(
        [sys.executable, str(TOOL), *synthetic.apply_args()],
        cwd=ROOT,
        env=fixture_environment(
            STEMGENRT_C236_PAUSE_AFTER="cleanup_retired_removed"
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paused = synthetic.workspace / ".c236-substitution-test-paused"
    resume = synthetic.workspace / ".c236-substitution-test-resume"
    deadline = time.monotonic() + 10.0
    while not paused.exists() and first.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert paused.read_text(encoding="utf-8") == "cleanup_retired_removed\n"
        if injection == "live_model":
            unexpected = synthetic.workspace / "model" / "unexpected-live-member"
        else:
            retired = synthetic.workspace / ".c236-substitution-txn.retired"
            retired.mkdir()
            unexpected = retired / "unexpected-retired-member"
        unexpected.write_bytes(b"late unknown entry must preserve cleanup authority\n")
        resume.write_text("resume\n", encoding="utf-8")
        _stdout, stderr = first.communicate(timeout=30.0)
        assert first.returncode != 0, stderr
        assert unexpected.exists()
        assert (
            synthetic.workspace / ".c236-substitution-cleanup-journal.json"
        ).is_file()

        blocked = run_tool(
            recover_args(synthetic),
            environment=fixture_environment(),
            timeout=30.0,
        )
        assert blocked.returncode != 0
        assert unexpected.exists()
        assert (
            synthetic.workspace / ".c236-substitution-cleanup-journal.json"
        ).is_file()

        unexpected.unlink()
        recovered = run_tool(
            recover_args(synthetic),
            environment=fixture_environment(),
            timeout=30.0,
        )
        assert recovered.returncode == 0, recovered.stderr
        assert json.loads(recovered.stdout)["status"] == "recovered_final"
        assert_final_tree(synthetic)
    finally:
        if first.poll() is None:
            resume.write_text("resume\n", encoding="utf-8")
            first.kill()
            first.wait(timeout=5.0)


def test_recover_without_a_transaction_is_inert_and_fails_closed(
    synthetic: SyntheticFixture,
) -> None:
    before = tree_snapshot(synthetic.workspace)
    recovered = run_tool(
        recover_args(synthetic), environment=fixture_environment(), timeout=10.0
    )
    assert recovered.returncode != 0
    assert "no c236 substitution transaction exists" in recovered.stderr
    assert tree_snapshot(synthetic.workspace) == before


def test_concurrent_writer_is_rejected_while_first_writer_owns_lock(
    synthetic: SyntheticFixture,
) -> None:
    first_environment = fixture_environment(
        STEMGENRT_C236_PAUSE_AFTER="journal_prepared"
    )
    first = subprocess.Popen(
        [sys.executable, str(TOOL), *synthetic.apply_args()],
        cwd=ROOT,
        env=first_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paused = synthetic.workspace / ".c236-substitution-test-paused"
    resume = synthetic.workspace / ".c236-substitution-test-resume"
    deadline = time.monotonic() + 10.0
    while not paused.exists() and first.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert paused.read_text(encoding="utf-8") == "journal_prepared\n"
        second = run_tool(
            synthetic.apply_args(),
            environment=fixture_environment(),
            timeout=10.0,
        )
        assert second.returncode != 0
        assert "owns the workspace lock" in second.stderr
        resume.write_text("resume\n", encoding="utf-8")
        first_stdout, first_stderr = first.communicate(timeout=30.0)
        assert first.returncode == 0, first_stderr
        assert json.loads(first_stdout)["status"] == (
            "substituted_unpromoted_audition_candidate"
        )
        assert_final_tree(synthetic)
    finally:
        if first.poll() is None:
            resume.write_text("resume\n", encoding="utf-8")
            first.kill()
            first.wait(timeout=5.0)
