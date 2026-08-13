#!/usr/bin/env python3
"""Fail-closed, contract-last installation of the checked c236 audition graph.

The default invocation is an inert source-plan check.  Real substitution needs
two independently reviewed trust anchors and the immutable checked-export
completion bundle.  No invocation builds, runs, listens to, commits, pushes,
or promotes the model.

The live ``cmake/QualifiedModelContract.cmake`` sentinel is the transaction's
single activation point.  Model/evidence changes are made durable while that
sentinel is still live; the fully validated final contract is exchanged last.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT = Path(__file__).resolve()
CANONICAL_WORKSPACE = Path(
    "/home/axel/autoresearch/codex/HS-TasNet-latency11-v1-state/"
    "workspaces/c236-plugin-audition-preparation-v1/stemgen-rt"
)
STATE_ROOT = Path("/home/axel/autoresearch/codex/HS-TasNet-latency11-v1-state")
CANONICAL_EXPORT_COMPLETION = (
    STATE_ROOT
    / "artifacts/c236-terminal-recovery-checked-onnx-export-v3/export-complete"
)
CANONICAL_EXPORT_ATTEMPTS = (
    STATE_ROOT
    / "artifacts/c236-terminal-recovery-checked-onnx-export-v3/attempts"
)
CANONICAL_CHAIN_ROOT = (
    STATE_ROOT / "artifacts/c236-terminal-recovery-selected-chain-v1"
)
BASE_COMMIT = "8b0f2a041aaf546637f89559324f4c6d362611c1"

ZERO_SHA256 = "0" * 64
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ATTEMPT_NAME = re.compile(r"^attempt-[0-9]{3}$")
QUALIFICATION_NAME = re.compile(r"^qualification-v2-[0-9]{3}\.json$")
SELECTABLE_UPDATES = {32_768, 65_536, 100_000}
QUALIFIED_STATUS = "qualified_by_declared_quality_budget"

EXPECTED_MATERIALIZER_SHA256 = (
    "5da7c0d97faeabdfb87fbdf3d37e5a2af2847f8a1595fc0467b01a14ea59ca2d"
)
EXPECTED_SHARED_AUTHORITY_SHA256 = (
    "6fc64af8968199e926f529ac9626f20c5f9fd6c28bcea02ae426353ae19246d2"
)
EXPECTED_RECOVERY_AUTHORITY_SHA256 = (
    "fcb8edac6e94512d9be3b98d225541cf08458a3f3e015a139f36ac774bb76aa8"
)
EXPECTED_CHAIN_AUTHORITY_SHA256 = (
    "e8d23b5e78f36482e5f60b2e0d4b5b40d5cb91a242882f26a4f85f362a376cc9"
)
EXPECTED_RECOVERY_CONTRACT_SHA256 = (
    "a1125772696b9f05d699c9de2f92caaa92cfaa1a58d74b80713ea6d021db4501"
)
EXPECTED_C191_PAYLOAD_SHA256 = (
    "a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b"
)
EXPECTED_C191_HEAD_SHA256 = (
    "238991eced221734b5e931bbc4256412dd6ed361656f2cbe4864abccedd22bcf"
)

PENDING_CONTRACT_SHA256 = (
    "abcc6f2e210d244461be307cb87c664a95ca48ca8570a3d9ca25ba48039684d3"
)
PENDING_CONTRACT_BYTES = 7_965
BASELINE_MODEL = {
    "bytes": 114_646_796,
    "sha256": "370d0a8971b405bd9c7f49928ccdea66e5b28fb028f6f5425c9c1ba5dc162f91",
}
LEGACY_EVIDENCE = {
    "FULL_CORRECTION_CANDIDATE.json": {
        "bytes": 7_275,
        "sha256": "c9246d27cf9cd1fff69d50b4fbb6d651a009fd1af563ad07025be024ad56b030",
    },
    "FULL_CORRECTION_CANDIDATE.json.sha256": {
        "bytes": 97,
        "sha256": "bae56315f3dcf943a4d728123bdb7a42cf0ad89135bf303c9baefd59988a6b36",
    },
    "FULL_CORRECTION_QUALIFICATION.md": {
        "bytes": 4_055,
        "sha256": "c10ae4b27ab25d411de26dafd35bab1f19b67b10cc735f2a325379a8f827410c",
    },
}
PENDING_MODEL_INVENTORY = {"model.onnx", *LEGACY_EVIDENCE}

TXN_NAME = ".c236-substitution-txn"
RETIRED_TXN_NAME = ".c236-substitution-txn.retired"
CLEANUP_JOURNAL_NAME = ".c236-substitution-cleanup-journal.json"
STAGE_NAME = "stage"
BACKUP_NAME = "backup"
JOURNAL_NAME = "journal.json"
CONTRACT_RELATIVE = Path("cmake/QualifiedModelContract.cmake")
MODEL_RELATIVE = Path("model/model.onnx")
SUBSTITUTION_RECEIPT_NAME = "C236_AUDITION_SUBSTITUTION.json"
SUBSTITUTION_SIDECAR_NAME = f"{SUBSTITUTION_RECEIPT_NAME}.sha256"
SYNTHETIC_MARKER = ".c236-synthetic-fixture"
SYNTHETIC_MARKER_BYTES = b"stemgenrt-c236-synthetic-fixture-v1\n"

MUTATION_BOUNDARIES = (
    "transaction_directory_created",
    "journal_body_published",
    "journal_published",
    "stage_directories_created",
    "stage_model_published",
    "stage_contract_published",
    "stage_receipt_sidecar_published",
    "stage_receipt_published",
    "journal_prepared",
    "model_exchanged",
    "receipt_sidecar_published",
    "receipt_published",
    "legacy_candidate_retired",
    "legacy_sidecar_retired",
    "legacy_qualification_retired",
    "contract_activated",
    "final_verified",
    "cleanup_payload_removed",
    "cleanup_retired",
    "cleanup_sidecar_removed",
    "cleanup_journal_moved",
    "cleanup_retired_removed",
    "cleanup_complete",
)


class SubstitutionError(RuntimeError):
    """A fail-closed substitution or recovery condition."""


class SyntheticFault(SubstitutionError):
    """Deterministic ordinary-exception injection for disposable tests."""


_FIRED_SYNTHETIC_FAILURES: set[str] = set()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SubstitutionError(message)


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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SubstitutionError(f"invalid JSON in {label}: {error}") from error
    require(isinstance(value, Mapping), f"JSON root is not an object: {label}")
    require(raw == canonical_json_bytes(value), f"JSON is not canonical: {label}")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stat_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _moved_stat_tuple(value: os.stat_result) -> tuple[int, ...]:
    # A rename legitimately advances ctime while leaving the retained inode,
    # bytes, mode, ownership, link count, and content mtime unchanged.
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
    )


def _unlinked_stat_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
    )


def _absolute_real(path: Path, label: str, *, directory: bool) -> Path:
    requested = Path(os.path.abspath(path.expanduser()))
    try:
        resolved = requested.resolve(strict=True)
        metadata = requested.lstat()
    except OSError as error:
        raise SubstitutionError(f"missing {label}: {requested}") from error
    require(resolved == requested, f"{label} contains a symlink component: {requested}")
    if directory:
        require(stat.S_ISDIR(metadata.st_mode), f"{label} is not a directory")
    else:
        require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
    require(not stat.S_ISLNK(metadata.st_mode), f"{label} is a symlink")
    return requested


def _open_directory(path: Path, label: str) -> int:
    requested = _absolute_real(path, label, directory=True)
    before = requested.lstat()
    descriptor = os.open(
        requested,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if _stat_tuple(before) != _stat_tuple(opened):
        os.close(descriptor)
        raise SubstitutionError(f"{label} changed during open: {requested}")
    return descriptor


@dataclass
class RetainedFile:
    path: Path
    label: str
    descriptor: int
    stat: os.stat_result
    sha256: str

    @property
    def size(self) -> int:
        return self.stat.st_size

    def identity(self, *, sidecar: "RetainedFile | None" = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": str(self.path),
            "bytes": self.size,
            "sha256": self.sha256,
        }
        if sidecar is not None:
            result.update(
                {
                    "sidecar": str(sidecar.path),
                    "sidecar_sha256": sidecar.sha256,
                }
            )
        return result

    def read(self) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while offset < self.size:
            block = os.pread(
                self.descriptor, min(1 << 20, self.size - offset), offset
            )
            require(block != b"", f"short read from retained {self.label}")
            chunks.append(block)
            offset += len(block)
        self.revalidate()
        return b"".join(chunks)

    def copy_to(self, directory_fd: int, name: str, *, mode: int = 0o644) -> None:
        output = _open_unnamed_file(directory_fd)
        digest = hashlib.sha256()
        try:
            offset = 0
            while offset < self.size:
                block = os.pread(
                    self.descriptor, min(1 << 20, self.size - offset), offset
                )
                require(block != b"", f"short copy from retained {self.label}")
                digest.update(block)
                written = 0
                while written < len(block):
                    written += os.write(output, block[written:])
                offset += len(block)
            os.fchmod(output, mode)
            os.fsync(output)
            copied = os.fstat(output)
            require(
                copied.st_size == self.size
                and copied.st_nlink == 0
                and digest.hexdigest() == self.sha256,
                f"staged copy differs from retained {self.label}",
            )
            _publish_unnamed_file(directory_fd, output, name)
        finally:
            os.close(output)
        self.revalidate()

    def revalidate(self) -> None:
        opened = os.fstat(self.descriptor)
        try:
            current = self.path.lstat()
        except OSError as error:
            raise SubstitutionError(
                f"retained {self.label} pathname disappeared: {self.path}"
            ) from error
        require(
            _stat_tuple(opened) == _stat_tuple(self.stat) == _stat_tuple(current),
            f"retained {self.label} changed: {self.path}",
        )

    def revalidate_descriptor(self) -> None:
        require(
            _stat_tuple(os.fstat(self.descriptor)) == _stat_tuple(self.stat),
            f"retained {self.label} inode changed",
        )

    def revalidate_descriptor_after_move(self) -> None:
        require(
            _unlinked_stat_tuple(os.fstat(self.descriptor))
            == _unlinked_stat_tuple(self.stat),
            f"retained {self.label} inode changed after namespace mutation",
        )

    def revalidate_at(self, path: Path, label: str) -> None:
        try:
            current = path.lstat()
        except OSError as error:
            raise SubstitutionError(f"moved {label} disappeared: {path}") from error
        require(
            _moved_stat_tuple(os.fstat(self.descriptor))
            == _moved_stat_tuple(self.stat)
            == _moved_stat_tuple(current),
            f"moved {label} is not the retained preimage: {path}",
        )

    def close(self) -> None:
        os.close(self.descriptor)


def _open_retained(path: Path, label: str, *, one_link: bool = True) -> RetainedFile:
    requested = _absolute_real(path, label, directory=False)
    parent_fd = _open_directory(requested.parent, f"{label} parent")
    try:
        before = requested.lstat()
        descriptor = os.open(
            requested.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or _stat_tuple(before) != _stat_tuple(opened)
        or (one_link and opened.st_nlink != 1)
    ):
        os.close(descriptor)
        raise SubstitutionError(f"unsafe or changed {label}: {requested}")
    digest = hashlib.sha256()
    offset = 0
    while offset < opened.st_size:
        block = os.pread(descriptor, min(1 << 20, opened.st_size - offset), offset)
        if not block:
            os.close(descriptor)
            raise SubstitutionError(f"short read while hashing {label}: {requested}")
        digest.update(block)
        offset += len(block)
    result = RetainedFile(requested, label, descriptor, opened, digest.hexdigest())
    result.revalidate()
    return result


@contextlib.contextmanager
def _retained(path: Path, label: str, *, one_link: bool = True):
    value = _open_retained(path, label, one_link=one_link)
    try:
        yield value
    finally:
        value.close()


def _exact_sidecar_bytes(subject: RetainedFile) -> bytes:
    return f"{subject.sha256}  {subject.path.name}\n".encode("ascii")


def _identity_matches(record: Any, identity: Mapping[str, Any], label: str) -> None:
    require(isinstance(record, Mapping), f"{label} identity is absent")
    for key in ("path", "bytes", "sha256"):
        require(record.get(key) == identity.get(key), f"{label} {key} changed")
    if "sidecar" in identity:
        require(
            record.get("sidecar") == identity.get("sidecar")
            and record.get("sidecar_sha256") == identity.get("sidecar_sha256"),
            f"{label} sidecar identity changed",
        )


def _verify_sidecar(subject: RetainedFile, sidecar: RetainedFile, label: str) -> None:
    require(
        sidecar.read() == _exact_sidecar_bytes(subject),
        f"non-exact SHA-256 sidecar for {label}",
    )


def _directory_inventory(path: Path, expected: set[str], label: str) -> None:
    observed = _directory_names(path, label)
    require(observed == expected, f"{label} inventory changed: {sorted(observed)}")


def _directory_names(path: Path, label: str) -> set[str]:
    descriptor = _open_directory(path, label)
    try:
        observed = set(os.listdir(descriptor))
    finally:
        os.close(descriptor)
    return observed


def _under(path: Path, root: Path, label: str) -> Path:
    requested = Path(os.path.abspath(path.expanduser()))
    resolved = requested.resolve(strict=True)
    require(requested == resolved, f"{label} contains a symlink component")
    require(resolved.is_relative_to(root), f"{label} escaped its authority root")
    return resolved


def _sha_record(value: Any, label: str, *, allow_zero: bool = False) -> str:
    require(isinstance(value, str) and HEX_SHA256.fullmatch(value), f"invalid {label}")
    require(allow_zero or value != ZERO_SHA256, f"sentinel {label} is forbidden")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} is absent")
    return value


def _expected_metadata(binding: Mapping[str, Any]) -> dict[str, str]:
    metadata = {
        "hs_tasnet.c191.emitted_db_history_shape": "[1,4,2048]",
        "hs_tasnet.c191.head_state_sha256": EXPECTED_C191_HEAD_SHA256,
        "hs_tasnet.c191.partition_protocol": "sample_domain_causal_same_weights_new_256_parent_requires_requalification",
        "hs_tasnet.c191.payload_sha256": EXPECTED_C191_PAYLOAD_SHA256,
        "hs_tasnet.c236.composite_runtime_state_sha256": str(binding["composite_runtime_state_sha256"]),
        "hs_tasnet.c236.contract_identity_sha256": str(binding["contract_identity_sha256"]),
        "hs_tasnet.c236.evaluation_result_bound": "true",
        "hs_tasnet.c236.family": "c236-c214-native-db-separator-v1",
        "hs_tasnet.c236.final_chain_receipt_sha256": str(binding["final_chain_receipt_sha256"]),
        "hs_tasnet.c236.materialization_receipt_sha256": str(binding["materialization_receipt_sha256"]),
        "hs_tasnet.c236.materialized_artifact_sha256": str(binding["materialized_artifact_sha256"]),
        "hs_tasnet.c236.materializer_sha256": EXPECTED_MATERIALIZER_SHA256,
        "hs_tasnet.c236.materializer_verify_existing": "true",
        "hs_tasnet.c236.model_state_sha256": str(binding["model_state_sha256"]),
        "hs_tasnet.c236.previous_qualification_transferred": "false",
        "hs_tasnet.c236.qualification_v2_candidate_status": QUALIFIED_STATUS,
        "hs_tasnet.c236.qualification_v2_sha256": str(binding["qualification_v2_sha256"]),
        "hs_tasnet.c236.recovery_contract_sha256": EXPECTED_RECOVERY_CONTRACT_SHA256,
        "hs_tasnet.c236.recovery_receipt_authority_sha256": EXPECTED_RECOVERY_AUTHORITY_SHA256,
        "hs_tasnet.c236.recovery_receipt_sha256": str(binding["recovery_receipt_sha256"]),
        "hs_tasnet.c236.run_uuid": str(binding["run_uuid"]),
        "hs_tasnet.c236.selected_update": str(binding["selected_update"]),
        "hs_tasnet.c236.shared_authority_sha256": EXPECTED_SHARED_AUTHORITY_SHA256,
        "hs_tasnet.c236.source_checkpoint_sha256": str(binding["source_checkpoint_sha256"]),
        "hs_tasnet.c236.static_identity_sha256": str(binding["static_identity_sha256"]),
        "hs_tasnet.c236.terminal_chain_authority_sha256": EXPECTED_CHAIN_AUTHORITY_SHA256,
        "hs_tasnet.contract.schema_version": "1",
        "hs_tasnet.deployment.status": "post_training_export_candidate_not_plugin_qualified",
        "hs_tasnet.export.dtype": "float32",
        "hs_tasnet.export.dynamo": "false",
        "hs_tasnet.export.external_data": "false",
        "hs_tasnet.export.input_names": '["audio_chunk","analysis_history","fusion_hidden","emitted_db_history"]',
        "hs_tasnet.export.input_shapes_batch1": "[[1,2,256],[1,2,768],[2,1,1000],[1,4,2048]]",
        "hs_tasnet.export.mixture_consistency": "exact_float32_residual_to_other",
        "hs_tasnet.export.mode": "streaming",
        "hs_tasnet.export.opset_version": "17",
        "hs_tasnet.export.output_filename": "model.onnx",
        "hs_tasnet.export.output_names": '["separated_chunk","next_analysis_history","next_fusion_hidden","next_emitted_db_history"]',
        "hs_tasnet.export.output_shapes_batch1": "[[1,4,2,256],[1,2,768],[2,1,1000],[1,4,2048]]",
        "hs_tasnet.export.parity_semantics": "bounded_numeric_not_bit_exact",
        "hs_tasnet.export.residual_association": "audio_chunk-minus-sum-drums-bass-vocals-dim1-float32",
        "hs_tasnet.export.residual_source_index": "3",
        "hs_tasnet.export.residual_source_name": "other",
        "hs_tasnet.model.candidate_kind": "c236_terminal_selected_plus_exact_c191_hop256",
        "hs_tasnet.model.causal_current_chunk": "true",
        "hs_tasnet.model.checkpoint_sha256": str(binding["materialized_artifact_sha256"]),
        "hs_tasnet.model.checkpoint_state_sha256": str(binding["model_state_sha256"]),
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
        "hs_tasnet.streaming.state_names": '["analysis_history","fusion_hidden","emitted_db_history"]',
    }
    require(len(metadata) == 68, "internal c236 metadata inventory changed")
    return metadata


@dataclass(frozen=True)
class AuthenticatedArtifacts:
    completion: Mapping[str, Any]
    completion_identity: Mapping[str, Any]
    model: RetainedFile
    model_sidecar: RetainedFile
    export_receipt_file: RetainedFile
    export_receipt_sidecar: RetainedFile
    export_receipt: Mapping[str, Any]
    final_chain_file: RetainedFile
    final_chain_sidecar: RetainedFile
    final_chain: Mapping[str, Any]
    qualification_file: RetainedFile
    qualification_sidecar: RetainedFile
    qualification: Mapping[str, Any]
    binding: Mapping[str, Any]


@dataclass(frozen=True)
class RetainedBaseline:
    contract: RetainedFile
    model: RetainedFile
    legacy: Mapping[str, RetainedFile]

    def snapshot(self) -> dict[str, Any]:
        return {
            "contract": self.contract.identity(),
            "model": self.model.identity(),
            "legacy": {
                name: self.legacy[name].identity() for name in LEGACY_EVIDENCE
            },
        }

    def revalidate(self) -> None:
        self.contract.revalidate()
        self.model.revalidate()
        for name in LEGACY_EVIDENCE:
            self.legacy[name].revalidate()


def _validate_qualification(
    document: Mapping[str, Any],
    *,
    final_record: Mapping[str, Any],
    qualification_identity: Mapping[str, Any],
    materialized: Mapping[str, Any],
) -> None:
    require(
        document.get("schema_version") == 2
        and document.get("kind") == "hs_tasnet_c236_post_training_evaluation_v2"
        and document.get("status") == "complete"
        and document.get("unavailable_gaps") == []
        and document.get("requested_scopes") == ["full14", "electronic"]
        and document.get("completed_scopes") == ["full14", "electronic"],
        "qualification-v2 result is incomplete",
    )
    models = _mapping(document.get("models"), "qualification models")
    expected_models = {
        "candidate_corrected": "c236-selected-trained-runtime",
        "c126": "c126-step100000",
        "c91": "c91-step100000",
        "c191": "full-c191-step128",
    }
    require(set(models) == set(expected_models), "qualification model inventory changed")
    for name, expected_id in expected_models.items():
        require(
            isinstance(models[name], Mapping) and models[name].get("id") == expected_id,
            f"qualification model identity changed: {name}",
        )
    candidate = _mapping(models["candidate_corrected"], "candidate model")
    authority = _mapping(candidate.get("publication_authority"), "candidate authority")
    artifact = _mapping(materialized.get("candidate_artifact"), "materialized artifact")
    require(
        authority.get("kind") == "c236_terminal_selected_materialized_candidate_v1"
        and candidate.get("path") == artifact.get("path")
        and candidate.get("sha256") == artifact.get("sha256")
        and authority.get("candidate_artifact", {}).get("path") == artifact.get("path")
        and authority.get("candidate_artifact", {}).get("sha256") == artifact.get("sha256"),
        "qualification candidate differs from materialized authority",
    )
    validation = _mapping(
        document.get("qualification_v2_validation"), "qualification validation"
    )
    tradeoffs = _mapping(validation.get("quality_tradeoff_table"), "tradeoff table")
    require(
        validation.get("status") == "pass"
        and validation.get("fail_closed_structural_validation") is True
        and validation.get("sir_was_used_as_veto") is False
        and validation.get("candidate_status") == QUALIFIED_STATUS
        and tradeoffs.get("all_primary_quality_costs_within_allowance") is True
        and tradeoffs.get("baseline") == "full-c191-step128"
        and tradeoffs.get("candidate") == "candidate_corrected",
        "qualification did not pass the declared quality budget",
    )
    _identity_matches(final_record, qualification_identity, "final qualification")
    require(
        final_record.get("candidate_status") == QUALIFIED_STATUS
        and final_record.get("quality_tradeoff_table") == tradeoffs
        and final_record.get("required_scopes") == ["full14", "electronic"]
        and final_record.get("required_models")
        == ["candidate_corrected", "c126", "c91", "c191"],
        "final-chain qualification summary changed",
    )


def _authenticate_artifacts(
    completion_directory: Path,
    expected_export_receipt_sha256: str,
    expected_final_chain_receipt_sha256: str,
    *,
    fixture: bool,
    workspace: Path,
    stack: contextlib.ExitStack,
) -> AuthenticatedArtifacts:
    completion_directory = _absolute_real(
        completion_directory, "checked-export completion", directory=True
    )
    if not fixture:
        require(
            completion_directory == CANONICAL_EXPORT_COMPLETION,
            "real substitution requires the canonical checked-export completion",
        )
    else:
        require(
            completion_directory.is_relative_to(workspace),
            "synthetic completion escaped its fixture workspace",
        )
    _directory_inventory(
        completion_directory,
        {"receipt.json", "receipt.json.sha256"},
        "checked-export completion",
    )
    completion_file = stack.enter_context(
        _retained(completion_directory / "receipt.json", "completion receipt")
    )
    completion_sidecar = stack.enter_context(
        _retained(
            completion_directory / "receipt.json.sha256", "completion receipt sidecar"
        )
    )
    _verify_sidecar(completion_file, completion_sidecar, "completion receipt")
    completion = strict_json(completion_file.read(), "completion receipt")
    require(
        completion.get("schema_version") == 1
        and completion.get("kind")
        == "hs_tasnet_c236_terminal_recovery_checked_export_completion_v1"
        and completion.get("status") == "complete_qualified_unpromoted"
        and completion.get("candidate_status") == QUALIFIED_STATUS
        and completion.get("cpu_only") is True
        and completion.get("gpu_lock_acquired") is False
        and completion.get("plugin_tree_touched") is False
        and completion.get("promotion_performed") is False
        and completion.get("git_operation_performed") is False,
        "checked-export completion status changed",
    )
    publication = _mapping(completion.get("publication"), "completion publication")
    attempt = Path(str(publication.get("directory")))
    attempt = _absolute_real(attempt, "checked-export attempt", directory=True)
    if not fixture:
        require(
            attempt.parent == CANONICAL_EXPORT_ATTEMPTS
            and ATTEMPT_NAME.fullmatch(attempt.name) is not None,
            "checked-export attempt path changed",
        )
    else:
        require(
            attempt.is_relative_to(workspace)
            and ATTEMPT_NAME.fullmatch(attempt.name) is not None,
            "synthetic checked-export attempt path changed",
        )
    quartet_names = {
        "model.onnx",
        "model.onnx.sha256",
        "model.onnx.export.json",
        "model.onnx.export.json.sha256",
    }
    _directory_inventory(attempt, quartet_names, "checked-export quartet")
    model = stack.enter_context(_retained(attempt / "model.onnx", "checked ONNX"))
    model_sidecar = stack.enter_context(
        _retained(attempt / "model.onnx.sha256", "checked ONNX sidecar")
    )
    receipt_file = stack.enter_context(
        _retained(attempt / "model.onnx.export.json", "checked-export receipt")
    )
    receipt_sidecar = stack.enter_context(
        _retained(
            attempt / "model.onnx.export.json.sha256",
            "checked-export receipt sidecar",
        )
    )
    _verify_sidecar(model, model_sidecar, "checked ONNX")
    _verify_sidecar(receipt_file, receipt_sidecar, "checked-export receipt")
    require(
        receipt_file.sha256 == expected_export_receipt_sha256,
        "checked-export receipt differs from the independently reviewed hash",
    )
    receipt = strict_json(receipt_file.read(), "checked-export receipt")
    expected_top = {
        "schema_version",
        "kind",
        "status",
        "created_at_utc",
        "cpu_only",
        "deployment_status",
        "promotion_performed",
        "plugin_tree_touched",
        "training_executed",
        "output_name",
        "export_sources",
        "terminal_authority",
        "runtime_identity",
        "export",
        "qualification",
        "parity_policy",
        "abi",
        "evaluator_coordination",
        "gate_pass",
    }
    require(set(receipt) == expected_top, "checked-export receipt inventory changed")
    require(
        receipt.get("schema_version") == 3
        and receipt.get("kind") == "hs_tasnet_c236_recovery_checked_onnx_export_v3"
        and receipt.get("status") == "pass"
        and receipt.get("cpu_only") is True
        and receipt.get("deployment_status")
        == "qualified_unpromoted_listening_candidate"
        and receipt.get("promotion_performed") is False
        and receipt.get("plugin_tree_touched") is False
        and receipt.get("training_executed") is False
        and receipt.get("output_name") == "model.onnx"
        and receipt.get("gate_pass") is True,
        "checked-export receipt protection flags changed",
    )
    model_identity = model.identity(sidecar=model_sidecar)
    receipt_identity = receipt_file.identity(sidecar=receipt_sidecar)
    _identity_matches(publication.get("model"), model_identity, "completion model")
    _identity_matches(
        publication.get("export_receipt"), receipt_identity, "completion export receipt"
    )
    export = _mapping(receipt.get("export"), "checked export")
    snapshot = _mapping(
        export.get("qualified_publication_snapshot"), "qualified model snapshot"
    )
    require(
        export.get("output_path") == str(model.path)
        and export.get("file_sha256") == model.sha256
        and export.get("bytes") == model.size
        # The qualified snapshot records the exporter-private staging name.
        # That name is deliberately gone after publication; its sealed bytes,
        # not its historical pathname, are the authentication boundary.
        and isinstance(snapshot.get("path"), str)
        and bool(snapshot.get("path"))
        and snapshot.get("file_sha256") == model.sha256
        and snapshot.get("bytes") == model.size
        and snapshot.get("source_fd_retained_through_publication") is True
        and snapshot.get("publication_source")
        == "exact_qualified_inode_via_linux_linkat",
        "checked ONNX differs from its qualified receipt snapshot",
    )
    native_dft = _mapping(export.get("final_native_dft_audit"), "native DFT audit")
    require(
        native_dft.get("required") is True
        and native_dft.get("validated_from_exact_serialized_bytes") is True
        and native_dft.get("raw_dft_count") == 3
        and native_dft.get("forward_rfft_count") == 2
        and native_dft.get("inverse_irfft_count") == 1
        and native_dft.get("axis") == 1
        and native_dft.get("n_fft") == 1024
        and native_dft.get("forward_onesided") == 1
        and native_dft.get("inverse_onesided") == 0
        and native_dft.get("external_data") is False,
        "final native-DFT proof changed",
    )
    abi = _mapping(receipt.get("abi"), "checked-export ABI")
    require(
        abi.get("sample_rate") == 44_100
        and abi.get("hop_samples") == 256
        and abi.get("host_visible_pdc_samples") == 256
        and abi.get("analysis_window_samples") == 1_024
        and abi.get("analysis_history_samples") == 768
        and abi.get("future_context_samples") == 0
        and abi.get("flush_required") is False
        and abi.get("input_names")
        == ["audio_chunk", "analysis_history", "fusion_hidden", "emitted_db_history"]
        and abi.get("output_names")
        == [
            "separated_chunk",
            "next_analysis_history",
            "next_fusion_hidden",
            "next_emitted_db_history",
        ]
        and abi.get("input_shapes_batch1")
        == [[1, 2, 256], [1, 2, 768], [2, 1, 1000], [1, 4, 2048]]
        and abi.get("output_shapes_batch1")
        == [[1, 4, 2, 256], [1, 2, 768], [2, 1, 1000], [1, 4, 2048]]
        and abi.get("source_order") == ["drums", "bass", "vocals", "other"]
        and abi.get("residual_source") == "other"
        and abi.get("dtype") == "float32",
        "checked-export ABI changed",
    )
    graph = _mapping(receipt.get("qualification"), "ONNX graph qualification")
    require(
        graph.get("schema_version") == 2
        and graph.get("kind") == "hs_tasnet_c236_64_hop_eager_onnx_qualification_v2"
        and graph.get("parity_hops") == 64
        and graph.get("provider") == "CPUExecutionProvider"
        and graph.get("onnxruntime_version") == "1.26.0"
        and graph.get("cross_backend_bit_exact") is False
        and graph.get("bounded_numeric") is True
        and graph.get("intra_backend_reset_replay_bit_exact") is True
        and graph.get("normal_onnx_exact_residual_other_calls") == 64
        and graph.get("normal_onnx_exact_residual_other_every_call") is True
        and graph.get("same_hop_boundary_impulses_influence_current_dbv") is True
        and graph.get("audio_chunk_reaches_separated_and_all_next_states") is True
        and graph.get("every_state_live_eager") is True
        and graph.get("every_state_live_onnx") is True
        and graph.get("every_state_live_graph") is True
        and graph.get("deterministic_positive_zero_reset") is True
        and graph.get("all_outputs_and_states_finite") is True,
        "ONNX graph qualification changed",
    )
    parity_policy = _mapping(receipt.get("parity_policy"), "parity policy")
    require(
        parity_policy.get("semantics") == "bounded_numeric_not_bit_exact"
        and parity_policy.get("bit_exact_claimed") is False
        and parity_policy.get("cross_backend_bit_exact") is False
        and parity_policy.get("bounded_numeric") is True
        and parity_policy.get("intra_backend_reset_replay_bit_exact") is True,
        "parity policy changed",
    )

    terminal = _mapping(receipt.get("terminal_authority"), "terminal authority")
    require(
        terminal.get("schema_version") == 1
        and terminal.get("kind")
        == "hs_tasnet_c236_terminal_recovery_export_authority_v1",
        "terminal export authority changed",
    )
    selected_update = terminal.get("selected_update")
    require(
        type(selected_update) is int and selected_update in SELECTABLE_UPDATES,
        "terminal selected update changed",
    )
    materialized = _mapping(terminal.get("materialized_candidate"), "materialized candidate")
    require(
        materialized.get("kind") == "c236_terminal_selected_materialized_candidate_v1"
        and materialized.get("id") == "c236-c214-native-db-separator-v1"
        and materialized.get("selected_update") == selected_update
        and materialized.get("validation_opened_before_evaluation") is False
        and materialized.get("promotion_performed") is False
        and materialized.get("protected_audio_used") is False
        and materialized.get("c191_payload_sha256") == EXPECTED_C191_PAYLOAD_SHA256
        and materialized.get("c191_runtime_head_state_sha256")
        == EXPECTED_C191_HEAD_SHA256,
        "materialized candidate authority changed",
    )
    chain = _mapping(
        terminal.get("terminal_recovery_chain"), "terminal recovery chain"
    )
    require(
        chain.get("schema_version") == 1
        and chain.get("kind")
        == "hs_tasnet_c236_terminal_recovery_export_chain_authority_v1"
        and chain.get("status") == "pass"
        and chain.get("selected_update") == selected_update
        and chain.get("protected_audio_used_during_training") is False
        and chain.get("validation_opened_only_by_qualification_v2") is True
        and chain.get("promotion_performed") is False,
        "terminal recovery-chain authority changed",
    )
    terminal_run = _mapping(terminal.get("terminal_run"), "terminal run")
    recovery = _mapping(terminal_run.get("terminal_recovery"), "terminal recovery")
    recovery_contract = _mapping(recovery.get("recovery_contract"), "recovery contract")
    recovery_receipt = _mapping(recovery.get("recovery_receipt"), "recovery receipt")
    require(
        recovery_contract.get("sha256") == EXPECTED_RECOVERY_CONTRACT_SHA256
        and recovery.get("semantic_terminal_assertion_correction_count") == 1
        and recovery.get("replay_parity", {}).get("compared_update_count") == 390
        and recovery.get("replay_parity", {}).get("mismatch_count") == 0
        and recovery.get("protected_audio_used") is False
        and recovery.get("validation_opened") is False
        and recovery.get("promotion_performed") is False,
        "terminal recovery proof changed",
    )
    chain_bundle = _mapping(chain.get("final_chain_bundle"), "final chain bundle")
    chain_receipt_record = _mapping(chain_bundle.get("receipt"), "final chain receipt")
    chain_receipt_path = Path(str(chain_receipt_record.get("path")))
    chain_receipt_path = _absolute_real(
        chain_receipt_path, "final chain receipt", directory=False
    )
    chain_root = chain_receipt_path.parent.parent
    if not fixture:
        require(
            chain_root == CANONICAL_CHAIN_ROOT
            and chain_receipt_path == CANONICAL_CHAIN_ROOT / "chain-complete/receipt.json",
            "final-chain path changed",
        )
    else:
        require(
            chain_root.is_relative_to(workspace)
            and chain_receipt_path.name == "receipt.json"
            and chain_receipt_path.parent.name == "chain-complete",
            "synthetic final-chain path changed",
        )
    _directory_inventory(
        chain_receipt_path.parent,
        {"receipt.json", "receipt.json.sha256"},
        "final chain bundle",
    )
    final_file = stack.enter_context(_retained(chain_receipt_path, "final chain receipt"))
    final_sidecar = stack.enter_context(
        _retained(chain_receipt_path.with_name("receipt.json.sha256"), "final chain sidecar")
    )
    _verify_sidecar(final_file, final_sidecar, "final chain receipt")
    require(
        final_file.sha256 == expected_final_chain_receipt_sha256,
        "final-chain receipt differs from independently reviewed hash",
    )
    final_identity = final_file.identity(sidecar=final_sidecar)
    _identity_matches(chain_receipt_record, final_identity, "export final chain")
    _identity_matches(
        completion.get("chain_final_receipt"), final_identity, "completion final chain"
    )
    final = strict_json(final_file.read(), "final chain receipt")
    require(
        final.get("schema_version") == 1
        and final.get("kind")
        == "hs_tasnet_c236_terminal_recovery_post_training_chain_v1"
        and final.get("status") == "complete_stopped_before_export"
        and final.get("selected_update") == selected_update
        and final.get("run_uuid") == materialized.get("run_uuid")
        and final.get("protected_audio_used_during_training") is False
        and final.get("validation_opened_only_by_qualification_v2") is True
        and final.get("exclusive_gpu_lock", {}).get(
            "held_through_materialization_and_qualification"
        )
        is True,
        "final-chain receipt changed",
    )
    stop = _mapping(final.get("deliberate_stop"), "final-chain stop boundary")
    require(
        all(
            stop.get(key) is False
            for key in (
                "onnx_export_started",
                "onnx_artifact_published",
                "plugin_repository_touched",
                "model_promoted",
                "branch_pushed",
                "listening_approval_claimed",
            )
        ),
        "final chain crossed its deliberate stop boundary",
    )
    final_qualification_record = _mapping(
        final.get("qualification_v2"), "final qualification record"
    )
    qualification_path = Path(str(final_qualification_record.get("path")))
    qualification_path = _under(qualification_path, chain_root, "qualification-v2")
    require(
        ATTEMPT_NAME.fullmatch(qualification_path.parent.name) is not None
        and QUALIFICATION_NAME.fullmatch(qualification_path.name) is not None,
        "qualification-v2 path layout changed",
    )
    qualification_file = stack.enter_context(
        _retained(qualification_path, "qualification-v2 result")
    )
    qualification_sidecar = stack.enter_context(
        _retained(
            qualification_path.with_name(qualification_path.name + ".sha256"),
            "qualification-v2 sidecar",
        )
    )
    _verify_sidecar(qualification_file, qualification_sidecar, "qualification-v2")
    qualification_identity = qualification_file.identity(sidecar=qualification_sidecar)
    chain_qualification = _mapping(chain.get("qualification_v2"), "chain qualification")
    _identity_matches(
        chain_qualification, qualification_identity, "export chain qualification"
    )
    _identity_matches(
        completion.get("qualification_v2"), qualification_identity, "completion qualification"
    )
    qualification = strict_json(qualification_file.read(), "qualification-v2 result")
    _validate_qualification(
        qualification,
        final_record=final_qualification_record,
        qualification_identity=qualification_identity,
        materialized=materialized,
    )
    evaluation = _mapping(receipt.get("evaluator_coordination"), "evaluator coordination")
    runtime = _mapping(receipt.get("runtime_identity"), "runtime identity")
    candidate_artifact = _mapping(
        materialized.get("candidate_artifact"), "materialized artifact"
    )
    materialization_receipt = _mapping(
        materialized.get("materialization_receipt"), "materialization receipt"
    )
    binding: dict[str, Any] = {
        "model_sha256": model.sha256,
        "model_byte_size": model.size,
        "export_receipt_sha256": receipt_file.sha256,
        "selected_update": selected_update,
        "run_uuid": materialized.get("run_uuid"),
        "contract_identity_sha256": materialized.get("contract_identity_sha256"),
        "static_identity_sha256": materialized.get("static_identity_sha256"),
        "source_checkpoint_sha256": materialized.get("source_checkpoint_sha256"),
        "materialized_artifact_sha256": candidate_artifact.get("sha256"),
        "materialization_receipt_sha256": materialization_receipt.get("sha256"),
        "model_state_sha256": runtime.get("model_state_sha256"),
        "composite_runtime_state_sha256": runtime.get(
            "composite_runtime_state_sha256"
        ),
        "recovery_receipt_sha256": recovery_receipt.get("sha256"),
        "final_chain_receipt_sha256": final_file.sha256,
        "qualification_v2_sha256": qualification_file.sha256,
        "qualification_v2_candidate_status": QUALIFIED_STATUS,
        "evaluation_result_bound": "true",
    }
    require(
        isinstance(binding["run_uuid"], str)
        and binding["run_uuid"]
        and not binding["run_uuid"].startswith("PENDING_"),
        "selected run UUID is absent",
    )
    for key in (
        "contract_identity_sha256",
        "static_identity_sha256",
        "source_checkpoint_sha256",
        "materialized_artifact_sha256",
        "materialization_receipt_sha256",
        "model_state_sha256",
        "composite_runtime_state_sha256",
        "recovery_receipt_sha256",
        "final_chain_receipt_sha256",
        "qualification_v2_sha256",
    ):
        _sha_record(binding[key], key)
    require(
        runtime.get("materialized_file_sha256") == binding["materialized_artifact_sha256"]
        and runtime.get("c191_payload_sha256") == EXPECTED_C191_PAYLOAD_SHA256
        and runtime.get("c191_head_state_sha256") == EXPECTED_C191_HEAD_SHA256
        and runtime.get("c191_payload_restricted_single_read_sha256")
        == EXPECTED_C191_PAYLOAD_SHA256,
        "runtime identity changed",
    )
    expected_evaluation = {
        "shared_publication_authority_sha256": EXPECTED_SHARED_AUTHORITY_SHA256,
        "recovery_receipt_authority_sha256": EXPECTED_RECOVERY_AUTHORITY_SHA256,
        "terminal_chain_authority_sha256": EXPECTED_CHAIN_AUTHORITY_SHA256,
        "materialization_receipt_sha256": binding["materialization_receipt_sha256"],
        "candidate_file_sha256": binding["materialized_artifact_sha256"],
        "candidate_model_state_sha256": binding["model_state_sha256"],
        "composite_runtime_state_sha256": binding["composite_runtime_state_sha256"],
        "recovery_contract_sha256": EXPECTED_RECOVERY_CONTRACT_SHA256,
        "recovery_receipt_sha256": binding["recovery_receipt_sha256"],
        "semantic_terminal_assertion_correction_count": 1,
        "replay_compared_update_count": 390,
        "replay_mismatch_count": 0,
        "final_chain_receipt_sha256": binding["final_chain_receipt_sha256"],
        "qualification_v2_sha256": binding["qualification_v2_sha256"],
        "qualification_v2_candidate_status": QUALIFIED_STATUS,
        "post_training_evaluation_result_bound": True,
        "production_promotion_requires_target_mac_evidence": True,
    }
    require(evaluation == expected_evaluation, "evaluator coordination changed")
    require(
        publication.get("chain_final_receipt_sha256")
        == binding["final_chain_receipt_sha256"]
        and publication.get("qualification_v2_sha256")
        == binding["qualification_v2_sha256"]
        and completion.get("selected_update") == selected_update,
        "completion lineage binding changed",
    )
    metadata = _mapping(export.get("metadata"), "export metadata")
    require(
        "hs_tasnet.c236.export_receipt_sha256" not in metadata,
        "export-receipt SHA must remain evidence-only",
    )
    require(
        dict(metadata) == _expected_metadata(binding),
        "checked ONNX metadata contract changed",
    )
    for value in (
        model,
        model_sidecar,
        receipt_file,
        receipt_sidecar,
        final_file,
        final_sidecar,
        qualification_file,
        qualification_sidecar,
        completion_file,
        completion_sidecar,
    ):
        value.revalidate()
    return AuthenticatedArtifacts(
        completion=completion,
        completion_identity=completion_file.identity(sidecar=completion_sidecar),
        model=model,
        model_sidecar=model_sidecar,
        export_receipt_file=receipt_file,
        export_receipt_sidecar=receipt_sidecar,
        export_receipt=receipt,
        final_chain_file=final_file,
        final_chain_sidecar=final_sidecar,
        final_chain=final,
        qualification_file=qualification_file,
        qualification_sidecar=qualification_sidecar,
        qualification=qualification,
        binding=binding,
    )


CONTRACT_FIELDS: dict[str, tuple[str, bool]] = {
    "STEMGENRT_QUALIFIED_MODEL_SHA256": ("model_sha256", True),
    "STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE": ("model_byte_size", False),
    "STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SHA256": ("export_receipt_sha256", True),
    "STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE": ("selected_update", False),
    "STEMGENRT_QUALIFIED_C236_RUN_UUID": ("run_uuid", True),
    "STEMGENRT_QUALIFIED_C236_CONTRACT_IDENTITY_SHA256": (
        "contract_identity_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_STATIC_IDENTITY_SHA256": (
        "static_identity_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_SOURCE_CHECKPOINT_SHA256": (
        "source_checkpoint_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_MATERIALIZED_ARTIFACT_SHA256": (
        "materialized_artifact_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_MATERIALIZATION_RECEIPT_SHA256": (
        "materialization_receipt_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_MODEL_STATE_SHA256": ("model_state_sha256", True),
    "STEMGENRT_QUALIFIED_C236_COMPOSITE_RUNTIME_STATE_SHA256": (
        "composite_runtime_state_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_SHA256": (
        "recovery_receipt_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_FINAL_CHAIN_RECEIPT_SHA256": (
        "final_chain_receipt_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_SHA256": (
        "qualification_v2_sha256",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_CANDIDATE_STATUS": (
        "qualification_v2_candidate_status",
        True,
    ),
    "STEMGENRT_QUALIFIED_C236_EVALUATION_RESULT_BOUND": (
        "evaluation_result_bound",
        True,
    ),
}


def _render_final_contract(pending: bytes, binding: Mapping[str, Any]) -> bytes:
    try:
        text = pending.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SubstitutionError("pending contract is not UTF-8") from error
    rendered = text
    for variable, (binding_key, quoted) in CONTRACT_FIELDS.items():
        value = binding[binding_key]
        replacement_value = f'"{value}"' if quoted else str(value)
        pattern = re.compile(
            rf"set\({re.escape(variable)}\s+(?:\"[^\"]*\"|[0-9]+)\)",
            re.MULTILINE,
        )
        rendered, count = pattern.subn(
            f"set({variable} {replacement_value})", rendered, count=1
        )
        require(count == 1, f"pending contract field layout changed: {variable}")
    final = rendered.encode("utf-8")
    require(ZERO_SHA256.encode("ascii") not in final, "final contract retains a SHA sentinel")
    require(b"PENDING_C236" not in final, "final contract retains a pending marker")
    require(
        b'STEMGENRT_QUALIFIED_C236_EVALUATION_RESULT_BOUND "true"' in final,
        "final contract does not bind evaluation",
    )
    return final


AT_EMPTY_PATH = 0x1000
AT_FDCWD = -100
AT_SYMLINK_FOLLOW = 0x400


def _open_unnamed_file(directory_fd: int) -> int:
    require(hasattr(os, "O_TMPFILE"), "Linux O_TMPFILE is unavailable")
    return os.open(
        ".",
        os.O_WRONLY | os.O_TMPFILE | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )


def _publish_unnamed_file(directory_fd: int, descriptor: int, name: str) -> None:
    require(
        Path(name).name == name and name not in {"", ".", ".."},
        f"unsafe publication name: {name}",
    )
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    require(linkat is not None, "Linux linkat is unavailable")
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = linkat(
        descriptor,
        b"",
        directory_fd,
        os.fsencode(name),
        AT_EMPTY_PATH,
    )
    if result != 0 and ctypes.get_errno() in {errno.ENOENT, errno.EPERM}:
        # Unprivileged linkat(AT_EMPTY_PATH) can be denied even though linking
        # the caller-owned O_TMPFILE inode is safe. Linux documents this
        # /proc/self/fd form as the capability-free equivalent.
        ctypes.set_errno(0)
        result = linkat(
            AT_FDCWD,
            os.fsencode(f"/proc/self/fd/{descriptor}"),
            directory_fd,
            os.fsencode(name),
            AT_SYMLINK_FOLLOW,
        )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise SubstitutionError(f"refusing to replace staged path: {name}")
        raise OSError(number, os.strerror(number), name)
    opened = os.fstat(descriptor)
    published = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    require(
        stat.S_ISREG(published.st_mode)
        and _moved_stat_tuple(opened) == _moved_stat_tuple(published),
        f"published file changed during link: {name}",
    )
    _fsync_directory_fd(directory_fd)


def _write_exclusive(directory_fd: int, name: str, payload: bytes, mode: int) -> None:
    descriptor = _open_unnamed_file(directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        require(
            os.fstat(descriptor).st_size == len(payload),
            f"short staged write: {name}",
        )
        _publish_unnamed_file(directory_fd, descriptor, name)
    finally:
        os.close(descriptor)


def _fsync_directory_fd(descriptor: int) -> None:
    os.fsync(descriptor)


def _renameat2(
    source_fd: int,
    source: str,
    destination_fd: int,
    destination: str,
    flags: int,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "Linux renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        source_fd,
        os.fsencode(source),
        destination_fd,
        os.fsencode(destination),
        flags,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise SubstitutionError(f"refusing to replace live path: {destination}")
        raise OSError(number, os.strerror(number), destination)


RENAME_NOREPLACE = 1
RENAME_EXCHANGE = 2


def _path_identity(path: Path, label: str) -> dict[str, Any]:
    with _retained(path, label) as value:
        return value.identity()


def _retain_baseline(
    workspace: Path, *, fixture: bool, stack: contextlib.ExitStack
) -> RetainedBaseline:
    _directory_inventory(
        workspace / "model",
        PENDING_MODEL_INVENTORY,
        "pending model evidence",
    )
    contract = stack.enter_context(
        _retained(workspace / CONTRACT_RELATIVE, "pending contract")
    )
    model = stack.enter_context(
        _retained(workspace / MODEL_RELATIVE, "baseline model")
    )
    legacy = {
        name: stack.enter_context(
            _retained(workspace / "model" / name, f"legacy {name}")
        )
        for name in LEGACY_EVIDENCE
    }
    require(
        contract.sha256 == PENDING_CONTRACT_SHA256
        and contract.size == PENDING_CONTRACT_BYTES,
        "live contract is not the exact prepared sentinel",
    )
    if not fixture:
        require(
            model.sha256 == BASELINE_MODEL["sha256"]
            and model.size == BASELINE_MODEL["bytes"],
            "baseline model identity changed",
        )
        for name, expected in LEGACY_EVIDENCE.items():
            require(
                legacy[name].sha256 == expected["sha256"]
                and legacy[name].size == expected["bytes"],
                f"legacy 512 evidence changed: {name}",
            )
    retained = RetainedBaseline(contract=contract, model=model, legacy=legacy)
    retained.revalidate()
    return retained


def _baseline_snapshot(workspace: Path, *, fixture: bool) -> dict[str, Any]:
    with contextlib.ExitStack() as stack:
        return _retain_baseline(workspace, fixture=fixture, stack=stack).snapshot()


def _substitution_receipt(
    *,
    workspace: Path,
    artifacts: AuthenticatedArtifacts,
    baseline: Mapping[str, Any],
    final_contract_sha256: str,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    binding = dict(artifacts.binding)
    binding_hash = sha256_bytes(canonical_json_bytes(binding))
    return {
        "schema_version": 1,
        "kind": "stemgenrt_c236_audition_substitution_authority_v1",
        "status": "authorized_for_contract_last_activation",
        "created_at_utc": (
            dt.datetime.now(dt.timezone.utc).isoformat()
            if created_at_utc is None
            else created_at_utc
        ),
        "workspace": {
            "path": str(workspace),
            "base_commit": BASE_COMMIT,
            "pending_contract": baseline["contract"],
        },
        "trust_anchors": {
            "expected_export_receipt_sha256": artifacts.export_receipt_file.sha256,
            "expected_final_chain_receipt_sha256": artifacts.final_chain_file.sha256,
            "independently_supplied": True,
        },
        "checked_export": {
            "completion": artifacts.completion_identity,
            "model": artifacts.model.identity(sidecar=artifacts.model_sidecar),
            "receipt": artifacts.export_receipt_file.identity(
                sidecar=artifacts.export_receipt_sidecar
            ),
            "schema_version": 3,
            "kind": "hs_tasnet_c236_recovery_checked_onnx_export_v3",
            "status": "pass",
            "deployment_status": "qualified_unpromoted_listening_candidate",
        },
        "qualification": {
            "final_chain_receipt": artifacts.final_chain_file.identity(
                sidecar=artifacts.final_chain_sidecar
            ),
            "qualification_v2": artifacts.qualification_file.identity(
                sidecar=artifacts.qualification_sidecar
            ),
            "candidate_status": QUALIFIED_STATUS,
            "required_scopes": ["full14", "electronic"],
        },
        "contract_binding": {
            "values": binding,
            "canonical_map_sha256": binding_hash,
            "final_contract_sha256": final_contract_sha256,
            "export_receipt_sha_is_onnx_metadata": False,
        },
        "fixed_authorities": {
            "materializer_sha256": EXPECTED_MATERIALIZER_SHA256,
            "shared_publication_authority_sha256": EXPECTED_SHARED_AUTHORITY_SHA256,
            "recovery_receipt_authority_sha256": EXPECTED_RECOVERY_AUTHORITY_SHA256,
            "terminal_chain_authority_sha256": EXPECTED_CHAIN_AUTHORITY_SHA256,
            "recovery_contract_sha256": EXPECTED_RECOVERY_CONTRACT_SHA256,
            "c191_payload_sha256": EXPECTED_C191_PAYLOAD_SHA256,
            "c191_head_state_sha256": EXPECTED_C191_HEAD_SHA256,
        },
        "retired_legacy_512_evidence": [
            {**dict(baseline["legacy"][name]), "disposition": "removed_at_activation"}
            for name in LEGACY_EVIDENCE
        ],
        "activation": {
            "protocol": "durable_dependencies_then_contract_last_atomic_exchange_v1",
            "complete_if_live_contract_sha256": final_contract_sha256,
            "pending_contract_sha256": PENDING_CONTRACT_SHA256,
            "model_copied_not_linked": True,
            "linux_renameat2_exchange": True,
            "linux_renameat2_noreplace": True,
            "directory_fsync": True,
            "symlink_and_hardlink_rejection": True,
            "ordinary_error_byte_exact_rollback": True,
            "crash_recovery_hash_inferred": True,
        },
        "scope": {
            "build_performed": False,
            "model_executed": False,
            "audio_accessed": False,
            "gpu_used": False,
            "commit_performed": False,
            "push_performed": False,
            "production_checkout_touched": False,
            "listening_approval_required": True,
            "target_mac_qualification_required": True,
        },
    }


def _validate_existing_substitution(
    document: Mapping[str, Any],
    *,
    workspace: Path,
    artifacts: AuthenticatedArtifacts,
    final_contract_sha256: str,
    fixture: bool,
) -> None:
    require(
        set(document)
        == {
            "schema_version",
            "kind",
            "status",
            "created_at_utc",
            "workspace",
            "trust_anchors",
            "checked_export",
            "qualification",
            "contract_binding",
            "fixed_authorities",
            "retired_legacy_512_evidence",
            "activation",
            "scope",
        },
        "existing substitution receipt inventory changed",
    )
    created_at_utc = document.get("created_at_utc")
    require(isinstance(created_at_utc, str), "substitution timestamp is absent")
    try:
        created = dt.datetime.fromisoformat(created_at_utc)
    except ValueError as error:
        raise SubstitutionError("substitution timestamp is malformed") from error
    require(created.tzinfo is not None, "substitution timestamp has no timezone")
    pending_contract = {
        "path": str(workspace / CONTRACT_RELATIVE),
        "bytes": PENDING_CONTRACT_BYTES,
        "sha256": PENDING_CONTRACT_SHA256,
    }
    retired = document.get("retired_legacy_512_evidence")
    require(
        isinstance(retired, list) and len(retired) == len(LEGACY_EVIDENCE),
        "retired legacy evidence inventory changed",
    )
    legacy: dict[str, Mapping[str, Any]] = {}
    for name, record in zip(LEGACY_EVIDENCE, retired, strict=True):
        require(
            isinstance(record, Mapping)
            and set(record) == {"path", "bytes", "sha256", "disposition"}
            and record.get("path") == str(workspace / "model" / name)
            and type(record.get("bytes")) is int
            and record.get("bytes", 0) > 0
            and isinstance(record.get("sha256"), str)
            and HEX_SHA256.fullmatch(str(record.get("sha256"))) is not None
            and record.get("disposition") == "removed_at_activation",
            f"retired legacy evidence changed: {name}",
        )
        if not fixture:
            expected = LEGACY_EVIDENCE[name]
            require(
                record.get("bytes") == expected["bytes"]
                and record.get("sha256") == expected["sha256"],
                f"retired legacy identity changed: {name}",
            )
        legacy[name] = {
            "path": record["path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
    expected = _substitution_receipt(
        workspace=workspace,
        artifacts=artifacts,
        baseline={
            "contract": pending_contract,
            "model": {},
            "legacy": legacy,
        },
        final_contract_sha256=final_contract_sha256,
        created_at_utc=created_at_utc,
    )
    require(document == expected, "existing substitution receipt changed")


def _verify_fixture_mode(workspace: Path, allowed: bool) -> bool:
    requested = Path(os.path.abspath(workspace.expanduser()))
    if not allowed:
        require(
            requested == CANONICAL_WORKSPACE,
            "real substitution is restricted to the detached c236 audition workspace",
        )
        return False
    require(
        os.environ.get("STEMGENRT_C236_SUBSTITUTION_TEST_MODE") == "1",
        "synthetic fixture mode requires the explicit test environment",
    )
    require(
        requested != CANONICAL_WORKSPACE,
        "synthetic fixture mode is forbidden in the canonical workspace",
    )
    marker = requested / SYNTHETIC_MARKER
    with _retained(marker, "synthetic fixture marker") as retained:
        require(retained.read() == SYNTHETIC_MARKER_BYTES, "fixture marker changed")
    return True


def _test_hook(workspace: Path, boundary: str, *, fixture: bool) -> None:
    require(boundary in MUTATION_BOUNDARIES, f"unknown mutation boundary: {boundary}")
    if not fixture:
        return
    if (
        os.environ.get("STEMGENRT_C236_FAIL_AFTER") == boundary
        and boundary not in _FIRED_SYNTHETIC_FAILURES
    ):
        _FIRED_SYNTHETIC_FAILURES.add(boundary)
        raise SyntheticFault(f"synthetic failure after {boundary}")
    if os.environ.get("STEMGENRT_C236_KILL_AFTER") == boundary:
        os.kill(os.getpid(), signal.SIGKILL)
    if os.environ.get("STEMGENRT_C236_PAUSE_AFTER") == boundary:
        paused = workspace / ".c236-substitution-test-paused"
        resume = workspace / ".c236-substitution-test-resume"
        paused.write_text(f"{boundary}\n", encoding="utf-8")
        deadline = time.monotonic() + 10.0
        while not resume.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        require(resume.exists(), f"synthetic pause timed out after {boundary}")
        resume.unlink()
        paused.unlink()


def _run_static_contract_test(
    workspace: Path,
    *,
    contract: Path,
    model: Path,
    export_receipt: Path,
    binding: Mapping[str, Any],
    skip: bool,
) -> None:
    if skip:
        return
    command = [
        "cmake",
        f"-DC236_CONTRACT_FILE={contract}",
        "-DC236_CONTRACT_PHASE=final",
        f"-DC236_MODEL_FILE={model}",
        f"-DC236_EXPORT_RECEIPT_FILE={export_receipt}",
        f"-DC236_EXPECTED_MODEL_SHA256={binding['model_sha256']}",
        f"-DC236_EXPECTED_MODEL_BYTE_SIZE={binding['model_byte_size']}",
        f"-DC236_EXPECTED_EXPORT_RECEIPT_SHA256={binding['export_receipt_sha256']}",
        f"-DC236_EXPECTED_SELECTED_UPDATE={binding['selected_update']}",
        f"-DC236_EXPECTED_RUN_UUID={binding['run_uuid']}",
        f"-DC236_EXPECTED_CONTRACT_IDENTITY_SHA256={binding['contract_identity_sha256']}",
        f"-DC236_EXPECTED_STATIC_IDENTITY_SHA256={binding['static_identity_sha256']}",
        f"-DC236_EXPECTED_SOURCE_CHECKPOINT_SHA256={binding['source_checkpoint_sha256']}",
        f"-DC236_EXPECTED_MATERIALIZED_ARTIFACT_SHA256={binding['materialized_artifact_sha256']}",
        f"-DC236_EXPECTED_MATERIALIZATION_RECEIPT_SHA256={binding['materialization_receipt_sha256']}",
        f"-DC236_EXPECTED_MODEL_STATE_SHA256={binding['model_state_sha256']}",
        f"-DC236_EXPECTED_COMPOSITE_RUNTIME_STATE_SHA256={binding['composite_runtime_state_sha256']}",
        f"-DC236_EXPECTED_RECOVERY_RECEIPT_SHA256={binding['recovery_receipt_sha256']}",
        f"-DC236_EXPECTED_FINAL_CHAIN_RECEIPT_SHA256={binding['final_chain_receipt_sha256']}",
        f"-DC236_EXPECTED_QUALIFICATION_V2_SHA256={binding['qualification_v2_sha256']}",
        f"-DC236_EXPECTED_QUALIFICATION_V2_CANDIDATE_STATUS={binding['qualification_v2_candidate_status']}",
        "-P",
        str(workspace / "test/cmake/C236DraftContractTest.cmake"),
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    require(
        completed.returncode == 0,
        f"staged final contract test failed:\n{completed.stdout}",
    )


def _journal_identity_record(value: Any, label: str) -> Mapping[str, Any]:
    record = _mapping(value, label)
    require(
        set(record) == {"path", "bytes", "sha256"}
        and isinstance(record.get("path"), str)
        and type(record.get("bytes")) is int
        and record.get("bytes", 0) > 0,
        f"{label} identity changed",
    )
    _sha_record(record.get("sha256"), f"{label} SHA-256")
    return record


def _validate_journal(journal: Mapping[str, Any], workspace: Path) -> None:
    require(
        set(journal)
        == {
            "schema_version",
            "kind",
            "workspace",
            "pending_contract_sha256",
            "created_at_utc",
            "baseline",
            "final",
            "binding",
            "recovery",
        }
        and journal.get("schema_version") == 1
        and journal.get("kind") == "stemgenrt_c236_substitution_transaction_v1"
        and journal.get("workspace") == str(workspace)
        and journal.get("pending_contract_sha256") == PENDING_CONTRACT_SHA256
        and journal.get("recovery")
        == "infer_from_exact_live_and_backup_hashes_never_phase_label",
        "substitution journal changed",
    )
    created_at_utc = journal.get("created_at_utc")
    require(isinstance(created_at_utc, str), "journal timestamp is absent")
    try:
        created = dt.datetime.fromisoformat(created_at_utc)
    except ValueError as error:
        raise SubstitutionError("journal timestamp is malformed") from error
    require(created.tzinfo is not None, "journal timestamp has no timezone")
    baseline = _mapping(journal.get("baseline"), "journal baseline")
    require(set(baseline) == {"contract", "model", "legacy"}, "baseline inventory changed")
    contract = _journal_identity_record(baseline.get("contract"), "baseline contract")
    model = _journal_identity_record(baseline.get("model"), "baseline model")
    require(
        contract.get("path") == str(workspace / CONTRACT_RELATIVE)
        and contract.get("sha256") == PENDING_CONTRACT_SHA256
        and contract.get("bytes") == PENDING_CONTRACT_BYTES
        and model.get("path") == str(workspace / MODEL_RELATIVE),
        "journal baseline paths changed",
    )
    legacy = _mapping(baseline.get("legacy"), "journal legacy")
    require(set(legacy) == set(LEGACY_EVIDENCE), "journal legacy inventory changed")
    for name in LEGACY_EVIDENCE:
        record = _journal_identity_record(legacy.get(name), f"journal legacy {name}")
        require(record.get("path") == str(workspace / "model" / name), f"legacy path changed: {name}")
    if workspace == CANONICAL_WORKSPACE:
        require(
            model.get("sha256") == BASELINE_MODEL["sha256"]
            and model.get("bytes") == BASELINE_MODEL["bytes"],
            "journal baseline model differs from the sealed preimage",
        )
        for name, expected in LEGACY_EVIDENCE.items():
            record = _mapping(legacy.get(name), f"journal legacy {name}")
            require(
                record.get("sha256") == expected["sha256"]
                and record.get("bytes") == expected["bytes"],
                f"journal legacy differs from the sealed preimage: {name}",
            )
    final = _mapping(journal.get("final"), "journal final")
    require(
        set(final)
        == {
            "contract_sha256",
            "contract_bytes",
            "model",
            "substitution_receipt",
            "substitution_sidecar",
        }
        and type(final.get("contract_bytes")) is int
        and final.get("contract_bytes", 0) > 1,
        "journal final inventory changed",
    )
    _sha_record(final.get("contract_sha256"), "final contract SHA-256")
    for key in ("model", "substitution_receipt", "substitution_sidecar"):
        record = _mapping(final.get(key), f"final {key}")
        require(
            set(record) == {"sha256", "bytes"}
            and type(record.get("bytes")) is int
            and record.get("bytes", 0) > 0,
            f"final {key} identity changed",
        )
        _sha_record(record.get("sha256"), f"final {key} SHA-256")
    binding = _mapping(journal.get("binding"), "journal binding")
    require(
        set(binding) == {field[0] for field in CONTRACT_FIELDS.values()},
        "journal binding inventory changed",
    )


def _read_journal(
    workspace: Path, *, root_name: str = TXN_NAME
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, set[str]]]:
    txn = workspace / root_name
    _absolute_real(txn, "substitution transaction", directory=True)
    top_inventory = _directory_names(txn, "substitution transaction")
    allowed_top = {STAGE_NAME, BACKUP_NAME, JOURNAL_NAME, f"{JOURNAL_NAME}.sha256"}
    require(top_inventory <= allowed_top, "substitution transaction inventory changed")
    require(JOURNAL_NAME in top_inventory, "substitution journal is absent")
    journal_only_creation = (
        root_name == TXN_NAME and top_inventory == {JOURNAL_NAME}
    )
    require(
        f"{JOURNAL_NAME}.sha256" in top_inventory
        or root_name == RETIRED_TXN_NAME
        or journal_only_creation,
        "active substitution journal sidecar is absent",
    )
    stage = txn / STAGE_NAME
    backup = txn / BACKUP_NAME
    stage_inventory = (
        _directory_names(stage, "transaction stage") if STAGE_NAME in top_inventory else set()
    )
    backup_inventory = (
        _directory_names(backup, "transaction backup") if BACKUP_NAME in top_inventory else set()
    )
    require(
        stage_inventory
        <= {
            "model.onnx.final",
            "QualifiedModelContract.cmake.final",
            SUBSTITUTION_RECEIPT_NAME,
            SUBSTITUTION_SIDECAR_NAME,
        },
        "transaction stage inventory changed",
    )
    require(
        backup_inventory <= set(LEGACY_EVIDENCE),
        "transaction backup inventory changed",
    )
    journal_path = txn / JOURNAL_NAME
    sidecar_path = txn / f"{JOURNAL_NAME}.sha256"
    with _retained(journal_path, "substitution journal") as journal_file:
        journal = strict_json(journal_file.read(), "substitution journal")
        if os.path.lexists(sidecar_path):
            with _retained(sidecar_path, "substitution journal sidecar") as sidecar:
                _verify_sidecar(journal_file, sidecar, "substitution journal")
                identity = journal_file.identity(sidecar=sidecar)
        else:
            identity = journal_file.identity()
    _validate_journal(journal, workspace)
    return journal, identity, {"stage": stage_inventory, "backup": backup_inventory}


def _existing_hash(path: Path, label: str) -> str | None:
    if not os.path.lexists(path):
        return None
    with _retained(path, label) as retained:
        return retained.sha256


def _path_matches(path: Path, expected: Mapping[str, Any], label: str) -> bool:
    try:
        identity = _path_identity(path, label)
    except (SubstitutionError, OSError):
        return False
    return (
        identity["sha256"] == expected.get("sha256")
        and identity["bytes"] == expected.get("bytes")
    )


def _terminal_tree_state(
    workspace: Path, journal: Mapping[str, Any]
) -> str | None:
    baseline = _mapping(journal.get("baseline"), "journal baseline")
    final = _mapping(journal.get("final"), "journal final")
    baseline_legacy = _mapping(baseline.get("legacy"), "baseline legacy")
    contract_path = workspace / CONTRACT_RELATIVE
    model_path = workspace / MODEL_RELATIVE
    receipt_path = workspace / "model" / SUBSTITUTION_RECEIPT_NAME
    sidecar_path = workspace / "model" / SUBSTITUTION_SIDECAR_NAME
    model_inventory = _directory_names(
        workspace / "model", "live model evidence"
    )
    pending = (
        model_inventory == PENDING_MODEL_INVENTORY
        and
        _path_matches(
            contract_path, _mapping(baseline.get("contract"), "baseline contract"), "pending contract"
        )
        and _path_matches(
            model_path, _mapping(baseline.get("model"), "baseline model"), "baseline model"
        )
        and all(
            _path_matches(
                workspace / "model" / name,
                _mapping(baseline_legacy.get(name), f"baseline {name}"),
                f"baseline {name}",
            )
            for name in LEGACY_EVIDENCE
        )
        and not os.path.lexists(receipt_path)
        and not os.path.lexists(sidecar_path)
    )
    final_contract = {
        "sha256": final.get("contract_sha256"),
        "bytes": final.get("contract_bytes"),
    }
    final_tree = (
        model_inventory
        == {"model.onnx", SUBSTITUTION_RECEIPT_NAME, SUBSTITUTION_SIDECAR_NAME}
        and
        _path_matches(contract_path, final_contract, "final contract")
        and _path_matches(
            model_path, _mapping(final.get("model"), "final model"), "final model"
        )
        and _path_matches(
            receipt_path,
            _mapping(final.get("substitution_receipt"), "final receipt"),
            "final substitution receipt",
        )
        and _path_matches(
            sidecar_path,
            _mapping(final.get("substitution_sidecar"), "final sidecar"),
            "final substitution sidecar",
        )
        and all(
            not os.path.lexists(workspace / "model" / name)
            for name in LEGACY_EVIDENCE
        )
    )
    require(not (pending and final_tree), "pending and final tree identities overlap")
    if pending:
        return "pending"
    if final_tree:
        return "final"
    return None


def _unlink_expected(
    path: Path,
    alternatives: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    with _retained(path, label) as retained:
        require(
            any(
                retained.sha256 == expected.get("sha256")
                and retained.size == expected.get("bytes")
                for expected in alternatives
            ),
            f"unknown cleanup member: {path}",
        )
        retained.revalidate()
        os.unlink(path)
        retained.revalidate_descriptor_after_move()


def _remove_empty_directory(path: Path, label: str) -> None:
    require(_directory_names(path, label) == set(), f"{label} is not empty")
    os.rmdir(path)


def _retire_transaction(
    workspace: Path,
    journal: Mapping[str, Any],
    *,
    fixture: bool,
) -> Mapping[str, Any]:
    state = _terminal_tree_state(workspace, journal)
    require(state in {"pending", "final"}, "cleanup requires an exact terminal tree")
    txn = workspace / TXN_NAME
    retired = workspace / RETIRED_TXN_NAME
    require(not os.path.lexists(retired), "retired transaction already exists")
    _current, _identity, inventory = _read_journal(workspace)
    baseline = _mapping(journal.get("baseline"), "journal baseline")
    final = _mapping(journal.get("final"), "journal final")
    stage = txn / STAGE_NAME
    backup = txn / BACKUP_NAME
    stage_expectations: dict[str, Sequence[Mapping[str, Any]]] = {
        "model.onnx.final": (
            _mapping(baseline.get("model"), "baseline model"),
            _mapping(final.get("model"), "final model"),
        ),
        "QualifiedModelContract.cmake.final": (
            _mapping(baseline.get("contract"), "baseline contract"),
            {
                "sha256": final.get("contract_sha256"),
                "bytes": final.get("contract_bytes"),
            },
        ),
        SUBSTITUTION_RECEIPT_NAME: (
            _mapping(final.get("substitution_receipt"), "final receipt"),
        ),
        SUBSTITUTION_SIDECAR_NAME: (
            _mapping(final.get("substitution_sidecar"), "final sidecar"),
        ),
    }
    for name in sorted(inventory["stage"]):
        _unlink_expected(stage / name, stage_expectations[name], f"staged cleanup {name}")
    if os.path.lexists(stage):
        _remove_empty_directory(stage, "transaction stage")
    baseline_legacy = _mapping(baseline.get("legacy"), "baseline legacy")
    for name in sorted(inventory["backup"]):
        _unlink_expected(
            backup / name,
            (_mapping(baseline_legacy.get(name), f"baseline {name}"),),
            f"backup cleanup {name}",
        )
    if os.path.lexists(backup):
        _remove_empty_directory(backup, "transaction backup")
    txn_fd = _open_directory(txn, "trimmed transaction")
    try:
        _fsync_directory_fd(txn_fd)
    finally:
        os.close(txn_fd)
    require(
        _directory_names(txn, "trimmed transaction")
        in (
            {JOURNAL_NAME, f"{JOURNAL_NAME}.sha256"},
            {JOURNAL_NAME},
        ),
        "trimmed transaction inventory changed",
    )
    _test_hook(workspace, "cleanup_payload_removed", fixture=fixture)
    workspace_fd = _open_directory(workspace, "workspace cleanup root")
    try:
        _renameat2(
            workspace_fd, TXN_NAME, workspace_fd, RETIRED_TXN_NAME, RENAME_NOREPLACE
        )
        _fsync_directory_fd(workspace_fd)
    finally:
        os.close(workspace_fd)
    _test_hook(workspace, "cleanup_retired", fixture=fixture)
    return _resume_retired_cleanup(workspace, fixture=fixture, expected_state=state)


def _resume_retired_cleanup(
    workspace: Path,
    *,
    fixture: bool,
    expected_state: str | None = None,
) -> Mapping[str, Any]:
    retired = workspace / RETIRED_TXN_NAME
    marker = workspace / CLEANUP_JOURNAL_NAME
    require(
        os.path.lexists(retired) or os.path.lexists(marker),
        "no retired c236 transaction exists",
    )
    require(
        not os.path.lexists(workspace / TXN_NAME),
        "active transaction coexists with cleanup state",
    )
    state = expected_state
    journal_identity: Mapping[str, Any] | None = None
    if not os.path.lexists(marker):
        _absolute_real(retired, "retired transaction", directory=True)
        inventory = _directory_names(retired, "retired transaction")
        require(
            inventory
            in (
                {JOURNAL_NAME, f"{JOURNAL_NAME}.sha256"},
                {JOURNAL_NAME},
            ),
            "retired transaction inventory changed",
        )
        journal, journal_identity, nested = _read_journal(
            workspace, root_name=RETIRED_TXN_NAME
        )
        require(not nested["stage"] and not nested["backup"], "retired payload remains")
        observed_state = _terminal_tree_state(workspace, journal)
        require(
            observed_state in {"pending", "final"},
            "retired cleanup sees unknown live tree",
        )
        if state is not None:
            require(observed_state == state, "retired cleanup terminal state changed")
        state = observed_state
        retired_fd = _open_directory(retired, "retired transaction")
        workspace_fd = _open_directory(workspace, "workspace cleanup root")
        try:
            if f"{JOURNAL_NAME}.sha256" in inventory:
                os.unlink(f"{JOURNAL_NAME}.sha256", dir_fd=retired_fd)
                _fsync_directory_fd(retired_fd)
                _test_hook(workspace, "cleanup_sidecar_removed", fixture=fixture)
            _renameat2(
                retired_fd,
                JOURNAL_NAME,
                workspace_fd,
                CLEANUP_JOURNAL_NAME,
                RENAME_NOREPLACE,
            )
            _fsync_directory_fd(retired_fd)
            _fsync_directory_fd(workspace_fd)
        finally:
            os.close(workspace_fd)
            os.close(retired_fd)
        _test_hook(workspace, "cleanup_journal_moved", fixture=fixture)

    with _retained(marker, "cleanup journal marker") as marker_file:
        journal = strict_json(marker_file.read(), "cleanup journal marker")
        journal_identity = marker_file.identity()
        _validate_journal(journal, workspace)
        observed_state = _terminal_tree_state(workspace, journal)
        require(
            observed_state in {"pending", "final"},
            "cleanup marker sees unknown live tree",
        )
        if state is not None:
            require(observed_state == state, "cleanup marker terminal state changed")
        state = observed_state
        if os.path.lexists(retired):
            _remove_empty_directory(retired, "retired transaction")
            workspace_fd = _open_directory(workspace, "workspace cleanup root")
            try:
                _fsync_directory_fd(workspace_fd)
            finally:
                os.close(workspace_fd)
            _test_hook(workspace, "cleanup_retired_removed", fixture=fixture)
        require(
            not os.path.lexists(retired),
            "retired transaction reappeared during cleanup",
        )
        require(
            _terminal_tree_state(workspace, journal) == state,
            "terminal tree changed during cleanup",
        )
        marker_file.revalidate()
        os.unlink(marker)
        marker_file.revalidate_descriptor_after_move()
    workspace_fd = _open_directory(workspace, "workspace cleanup root")
    try:
        _fsync_directory_fd(workspace_fd)
    finally:
        os.close(workspace_fd)
    _test_hook(workspace, "cleanup_complete", fixture=fixture)
    return {
        "status": f"recovered_{state}",
        "journal": journal_identity,
    }


def _recover_locked(
    workspace: Path, *, fixture: bool, prefer_pending: bool = False
) -> Mapping[str, Any]:
    txn = workspace / TXN_NAME
    require(os.path.lexists(txn), "no c236 substitution transaction exists")
    _absolute_real(txn, "substitution transaction", directory=True)
    if not _directory_names(txn, "substitution transaction"):
        # The only journal-free crash point is immediately after durable mkdir,
        # before any live pathname mutation. Re-authenticate the complete real
        # baseline (the fixture path is deliberately not a production trust
        # boundary) and require that no final receipt appeared before removing
        # the empty marker.
        _baseline_snapshot(workspace, fixture=fixture)
        require(
            not os.path.lexists(workspace / "model" / SUBSTITUTION_RECEIPT_NAME)
            and not os.path.lexists(
                workspace / "model" / SUBSTITUTION_SIDECAR_NAME
            ),
            "empty creation transaction coexists with final evidence",
        )
        _remove_empty_directory(txn, "empty creation transaction")
        workspace_fd = _open_directory(workspace, "workspace recovery root")
        try:
            _fsync_directory_fd(workspace_fd)
        finally:
            os.close(workspace_fd)
        return {"status": "recovered_pending", "journal": None}
    journal, journal_identity, _inventory = _read_journal(workspace)
    baseline = _mapping(journal.get("baseline"), "journal baseline")
    final = _mapping(journal.get("final"), "journal final")
    contract_path = workspace / CONTRACT_RELATIVE
    model_path = workspace / MODEL_RELATIVE
    receipt_path = workspace / "model" / SUBSTITUTION_RECEIPT_NAME
    receipt_sidecar_path = workspace / "model" / SUBSTITUTION_SIDECAR_NAME
    stage = txn / STAGE_NAME
    backup = txn / BACKUP_NAME
    contract_hash = _existing_hash(contract_path, "live contract")
    pending_hash = str(journal["pending_contract_sha256"])
    final_contract_hash = str(final["contract_sha256"])

    final_dependencies = (
        _path_matches(model_path, _mapping(final.get("model"), "final model"), "final model")
        and _path_matches(
            receipt_path,
            _mapping(final.get("substitution_receipt"), "final substitution receipt"),
            "final substitution receipt",
        )
        and _path_matches(
            receipt_sidecar_path,
            _mapping(final.get("substitution_sidecar"), "final substitution sidecar"),
            "final substitution sidecar",
        )
        and all(not os.path.lexists(workspace / "model" / name) for name in LEGACY_EVIDENCE)
    )
    if contract_hash == final_contract_hash and final_dependencies and not prefer_pending:
        return _retire_transaction(workspace, journal, fixture=fixture)

    if _terminal_tree_state(workspace, journal) == "pending":
        return _retire_transaction(workspace, journal, fixture=fixture)

    stage_contract = stage / "QualifiedModelContract.cmake.final"
    stage_model = stage / "model.onnx.final"
    baseline_contract = _mapping(baseline.get("contract"), "baseline contract")
    baseline_model = _mapping(baseline.get("model"), "baseline model")
    baseline_legacy = _mapping(baseline.get("legacy"), "baseline legacy")

    if contract_hash == final_contract_hash:
        backups_complete = (
            _path_matches(stage_contract, baseline_contract, "sentinel contract backup")
            and _path_matches(stage_model, baseline_model, "baseline model backup")
            and all(
                _path_matches(backup / name, _mapping(baseline_legacy[name], name), f"legacy backup {name}")
                for name in LEGACY_EVIDENCE
            )
        )
        require(
            backups_complete,
            "final contract has mismatched dependencies and exact rollback backups are unavailable",
        )
        cmake_fd = _open_directory(workspace / "cmake", "workspace cmake")
        stage_fd = _open_directory(stage, "transaction stage")
        try:
            _renameat2(
                stage_fd,
                stage_contract.name,
                cmake_fd,
                contract_path.name,
                RENAME_EXCHANGE,
            )
            _fsync_directory_fd(cmake_fd)
            _fsync_directory_fd(stage_fd)
        finally:
            os.close(stage_fd)
            os.close(cmake_fd)
        contract_hash = _existing_hash(contract_path, "restored sentinel contract")

    require(
        contract_hash == pending_hash == baseline_contract.get("sha256"),
        "unknown live contract hash; recovery stopped without cleanup",
    )
    model_fd = _open_directory(workspace / "model", "workspace model")
    stage_fd = _open_directory(stage, "transaction stage")
    backup_fd = _open_directory(backup, "transaction backup")
    try:
        live_model_hash = _existing_hash(model_path, "live recovery model")
        if live_model_hash == final.get("model", {}).get("sha256"):
            require(
                _path_matches(stage_model, baseline_model, "baseline model backup"),
                "baseline model backup is unavailable",
            )
            _renameat2(
                stage_fd,
                stage_model.name,
                model_fd,
                model_path.name,
                RENAME_EXCHANGE,
            )
            _fsync_directory_fd(model_fd)
            _fsync_directory_fd(stage_fd)
        else:
            require(
                live_model_hash == baseline_model.get("sha256"),
                "unknown live model hash; recovery stopped",
            )
        for live_path, staged_name, expected_key in (
            (receipt_path, SUBSTITUTION_RECEIPT_NAME, "substitution_receipt"),
            (receipt_sidecar_path, SUBSTITUTION_SIDECAR_NAME, "substitution_sidecar"),
        ):
            if os.path.lexists(live_path):
                require(
                    _path_matches(live_path, _mapping(final[expected_key], expected_key), expected_key),
                    f"unknown live {expected_key}; recovery stopped",
                )
                require(
                    not os.path.lexists(stage / staged_name),
                    f"staged {expected_key} collision",
                )
                _renameat2(
                    model_fd, live_path.name, stage_fd, staged_name, RENAME_NOREPLACE
                )
        for name in LEGACY_EVIDENCE:
            live = workspace / "model" / name
            expected = _mapping(baseline_legacy[name], f"baseline {name}")
            if os.path.lexists(live):
                require(_path_matches(live, expected, f"live legacy {name}"), f"legacy {name} changed")
            else:
                require(
                    _path_matches(backup / name, expected, f"backup legacy {name}"),
                    f"legacy backup is unavailable: {name}",
                )
                _renameat2(backup_fd, name, model_fd, name, RENAME_NOREPLACE)
        _fsync_directory_fd(model_fd)
        _fsync_directory_fd(stage_fd)
        _fsync_directory_fd(backup_fd)
    finally:
        os.close(backup_fd)
        os.close(stage_fd)
        os.close(model_fd)
    restored = _baseline_snapshot(workspace, fixture=fixture)
    require(restored == baseline, "byte-exact baseline rollback verification failed")
    return _retire_transaction(workspace, journal, fixture=fixture)


def _recover_any_locked(
    workspace: Path, *, fixture: bool, prefer_pending: bool = False
) -> Mapping[str, Any]:
    active = os.path.lexists(workspace / TXN_NAME)
    retired = os.path.lexists(workspace / RETIRED_TXN_NAME)
    marker = os.path.lexists(workspace / CLEANUP_JOURNAL_NAME)
    require(
        not (active and (retired or marker)),
        "active transaction coexists with cleanup state",
    )
    if retired or marker:
        return _resume_retired_cleanup(workspace, fixture=fixture)
    if active:
        return _recover_locked(
            workspace, fixture=fixture, prefer_pending=prefer_pending
        )
    raise SubstitutionError("no c236 substitution transaction exists")


def _create_transaction(
    workspace_fd: int,
    workspace: Path,
    *,
    artifacts: AuthenticatedArtifacts,
    baseline: RetainedBaseline,
    final_contract: bytes,
    fixture: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    require(
        not os.path.lexists(workspace / TXN_NAME)
        and not os.path.lexists(workspace / RETIRED_TXN_NAME)
        and not os.path.lexists(workspace / CLEANUP_JOURNAL_NAME),
        "existing transaction requires --recover",
    )
    baseline.revalidate()
    baseline_snapshot = baseline.snapshot()
    final_contract_hash = sha256_bytes(final_contract)
    substitution = _substitution_receipt(
        workspace=workspace,
        artifacts=artifacts,
        baseline=baseline_snapshot,
        final_contract_sha256=final_contract_hash,
    )
    substitution_bytes = canonical_json_bytes(substitution)
    substitution_hash = sha256_bytes(substitution_bytes)
    sidecar_bytes = (
        f"{substitution_hash}  {SUBSTITUTION_RECEIPT_NAME}\n".encode("ascii")
    )
    final: dict[str, Any] = {
        "contract_sha256": final_contract_hash,
        "contract_bytes": len(final_contract),
        "model": {
            "sha256": artifacts.model.sha256,
            "bytes": artifacts.model.size,
        },
        "substitution_receipt": {
            "sha256": substitution_hash,
            "bytes": len(substitution_bytes),
        },
        "substitution_sidecar": {
            "sha256": sha256_bytes(sidecar_bytes),
            "bytes": len(sidecar_bytes),
        },
    }
    journal = {
        "schema_version": 1,
        "kind": "stemgenrt_c236_substitution_transaction_v1",
        "workspace": str(workspace),
        "pending_contract_sha256": PENDING_CONTRACT_SHA256,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline": baseline_snapshot,
        "final": final,
        "binding": dict(artifacts.binding),
        "recovery": "infer_from_exact_live_and_backup_hashes_never_phase_label",
    }
    journal_bytes = canonical_json_bytes(journal)
    journal_hash = sha256_bytes(journal_bytes)
    os.mkdir(TXN_NAME, mode=0o700, dir_fd=workspace_fd)
    _fsync_directory_fd(workspace_fd)
    _test_hook(workspace, "transaction_directory_created", fixture=fixture)
    txn_fd = _open_directory(workspace / TXN_NAME, "transaction")
    try:
        _write_exclusive(txn_fd, JOURNAL_NAME, journal_bytes, 0o444)
        _test_hook(workspace, "journal_body_published", fixture=fixture)
        _write_exclusive(
            txn_fd,
            f"{JOURNAL_NAME}.sha256",
            f"{journal_hash}  {JOURNAL_NAME}\n".encode("ascii"),
            0o444,
        )
        _fsync_directory_fd(txn_fd)
        _fsync_directory_fd(workspace_fd)
        baseline.revalidate()
        _test_hook(workspace, "journal_published", fixture=fixture)
        os.mkdir(STAGE_NAME, mode=0o700, dir_fd=txn_fd)
        os.mkdir(BACKUP_NAME, mode=0o700, dir_fd=txn_fd)
        _fsync_directory_fd(txn_fd)
        _test_hook(workspace, "stage_directories_created", fixture=fixture)
    finally:
        os.close(txn_fd)
    stage_fd = _open_directory(workspace / TXN_NAME / STAGE_NAME, "transaction stage")
    try:
        baseline.revalidate()
        artifacts.model.copy_to(stage_fd, "model.onnx.final")
        _fsync_directory_fd(stage_fd)
        _test_hook(workspace, "stage_model_published", fixture=fixture)
        _write_exclusive(
            stage_fd,
            "QualifiedModelContract.cmake.final",
            final_contract,
            0o644,
        )
        _fsync_directory_fd(stage_fd)
        _test_hook(workspace, "stage_contract_published", fixture=fixture)
        _write_exclusive(
            stage_fd, SUBSTITUTION_SIDECAR_NAME, sidecar_bytes, 0o444
        )
        _fsync_directory_fd(stage_fd)
        _test_hook(
            workspace, "stage_receipt_sidecar_published", fixture=fixture
        )
        _write_exclusive(
            stage_fd, SUBSTITUTION_RECEIPT_NAME, substitution_bytes, 0o444
        )
        _fsync_directory_fd(stage_fd)
        _test_hook(workspace, "stage_receipt_published", fixture=fixture)
    finally:
        os.close(stage_fd)
    baseline.revalidate()
    return journal, substitution


def _verify_final_tree(
    workspace: Path,
    *,
    final_contract_sha256: str,
    artifacts: AuthenticatedArtifacts,
    substitution: Mapping[str, Any],
) -> Mapping[str, Any]:
    with _retained(workspace / CONTRACT_RELATIVE, "final contract") as contract_file:
        final_contract = contract_file.read()
        contract = contract_file.identity()
    model = _path_identity(workspace / MODEL_RELATIVE, "final model")
    receipt_path = workspace / "model" / SUBSTITUTION_RECEIPT_NAME
    sidecar_path = workspace / "model" / SUBSTITUTION_SIDECAR_NAME
    with _retained(receipt_path, "substitution receipt") as receipt, _retained(
        sidecar_path, "substitution receipt sidecar"
    ) as sidecar:
        _verify_sidecar(receipt, sidecar, "substitution receipt")
        require(
            strict_json(receipt.read(), "substitution receipt") == substitution,
            "substitution receipt replay changed",
        )
        receipt_identity = receipt.identity(sidecar=sidecar)
    require(
        contract["sha256"] == final_contract_sha256
        and model["sha256"] == artifacts.model.sha256
        and model["bytes"] == artifacts.model.size,
        "final contract/model identity changed",
    )
    require(
        all(
            not os.path.lexists(workspace / "model" / name)
            for name in LEGACY_EVIDENCE
        ),
        "stale 512-hop evidence remains after substitution",
    )
    require(
        ZERO_SHA256.encode("ascii") not in final_contract
        and b"PENDING_C236" not in final_contract,
        "recovery sentinels remain in final contract",
    )
    _directory_inventory(
        workspace / "model",
        {"model.onnx", SUBSTITUTION_RECEIPT_NAME, SUBSTITUTION_SIDECAR_NAME},
        "final model evidence",
    )
    return {
        "contract": contract,
        "model": model,
        "substitution_receipt": receipt_identity,
    }


def apply_substitution(args: argparse.Namespace) -> Mapping[str, Any]:
    workspace = _absolute_real(args.workspace, "plugin audition workspace", directory=True)
    fixture = _verify_fixture_mode(workspace, args.allow_test_fixtures)
    require(
        isinstance(args.expected_export_receipt_sha256, str)
        and HEX_SHA256.fullmatch(args.expected_export_receipt_sha256)
        and args.expected_export_receipt_sha256 != ZERO_SHA256,
        "--expected-export-receipt-sha256 is mandatory and must be nonzero",
    )
    require(
        isinstance(args.expected_final_chain_receipt_sha256, str)
        and HEX_SHA256.fullmatch(args.expected_final_chain_receipt_sha256)
        and args.expected_final_chain_receipt_sha256 != ZERO_SHA256,
        "--expected-final-chain-receipt-sha256 is mandatory and must be nonzero",
    )
    require(args.export_completion is not None, "--export-completion is mandatory")
    require(
        not args.skip_static_contract_test or fixture,
        "static contract checks may only be skipped in synthetic fixtures",
    )
    workspace_fd = _open_directory(workspace, "plugin audition workspace")
    try:
        try:
            fcntl.flock(workspace_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SubstitutionError("another c236 substitution owns the workspace lock") from error
        require(
            not os.path.lexists(workspace / TXN_NAME)
            and not os.path.lexists(workspace / RETIRED_TXN_NAME)
            and not os.path.lexists(workspace / CLEANUP_JOURNAL_NAME),
            "existing transaction requires explicit --recover",
        )
        with contextlib.ExitStack() as stack:
            artifacts = _authenticate_artifacts(
                args.export_completion,
                args.expected_export_receipt_sha256,
                args.expected_final_chain_receipt_sha256,
                fixture=fixture,
                workspace=workspace,
                stack=stack,
            )
            contract_path = workspace / CONTRACT_RELATIVE
            with _retained(contract_path, "live pending contract") as pending_contract:
                current_contract_hash = pending_contract.sha256
                pending_bytes = pending_contract.read()
            final_contract = _render_final_contract(pending_bytes, artifacts.binding)
            final_contract_hash = sha256_bytes(final_contract)
            if current_contract_hash == final_contract_hash:
                receipt_path = workspace / "model" / SUBSTITUTION_RECEIPT_NAME
                require(os.path.lexists(receipt_path), "final contract lacks substitution receipt")
                with _retained(
                    receipt_path, "existing substitution receipt"
                ) as receipt_file:
                    substitution = strict_json(
                        receipt_file.read(), "existing substitution receipt"
                    )
                _validate_existing_substitution(
                    substitution,
                    workspace=workspace,
                    artifacts=artifacts,
                    final_contract_sha256=final_contract_hash,
                    fixture=fixture,
                )
                _run_static_contract_test(
                    workspace,
                    contract=contract_path,
                    model=workspace / MODEL_RELATIVE,
                    export_receipt=artifacts.export_receipt_file.path,
                    binding=artifacts.binding,
                    skip=args.skip_static_contract_test,
                )
                final = _verify_final_tree(
                    workspace,
                    final_contract_sha256=final_contract_hash,
                    artifacts=artifacts,
                    substitution=substitution,
                )
                return {"status": "already_applied", "final": final}
            require(
                current_contract_hash == PENDING_CONTRACT_SHA256,
                "live contract is neither the prepared sentinel nor this exact final contract",
            )
            baseline = _retain_baseline(workspace, fixture=fixture, stack=stack)
            try:
                journal, substitution = _create_transaction(
                    workspace_fd,
                    workspace,
                    artifacts=artifacts,
                    baseline=baseline,
                    final_contract=final_contract,
                    fixture=fixture,
                )
            except BaseException:
                if any(
                    os.path.lexists(workspace / name)
                    for name in (TXN_NAME, RETIRED_TXN_NAME, CLEANUP_JOURNAL_NAME)
                ):
                    try:
                        recovered = _recover_any_locked(
                            workspace, fixture=fixture, prefer_pending=True
                        )
                        require(
                            recovered["status"] == "recovered_pending",
                            "creation failure did not restore the pending tree",
                        )
                    except BaseException as recovery_error:
                        raise SubstitutionError(
                            "transaction creation failed and exact recovery could not "
                            "be proven; the tree requires --recover"
                        ) from recovery_error
                raise
            stage = workspace / TXN_NAME / STAGE_NAME
            try:
                baseline.revalidate()
                _test_hook(workspace, "journal_prepared", fixture=fixture)
                _run_static_contract_test(
                    workspace,
                    contract=stage / "QualifiedModelContract.cmake.final",
                    model=stage / "model.onnx.final",
                    export_receipt=artifacts.export_receipt_file.path,
                    binding=artifacts.binding,
                    skip=args.skip_static_contract_test,
                )
            except BaseException:
                # The journal is already durable, so even preparation-stage
                # failures take the same proven byte-exact rollback path.
                try:
                    recovered = _recover_any_locked(
                        workspace, fixture=fixture, prefer_pending=True
                    )
                    require(
                        recovered["status"] == "recovered_pending",
                        "preparation failure did not restore the pending tree",
                    )
                except BaseException as recovery_error:
                    raise SubstitutionError(
                        "substitution preparation failed and safe automatic recovery "
                        "could not be proven; the tree requires --recover"
                    ) from recovery_error
                raise
            model_fd: int | None = None
            cmake_fd: int | None = None
            stage_fd: int | None = None
            backup_fd: int | None = None
            try:
                # Open every mutation root under the already-held workspace
                # lock inside the recovery envelope. A caught open failure is
                # therefore as byte-exact and residue-free as a later failure.
                model_fd = _open_directory(workspace / "model", "workspace model")
                cmake_fd = _open_directory(workspace / "cmake", "workspace cmake")
                stage_fd = _open_directory(stage, "transaction stage")
                backup_fd = _open_directory(
                    workspace / TXN_NAME / BACKUP_NAME, "transaction backup"
                )
                baseline.revalidate()
                _renameat2(
                    stage_fd,
                    "model.onnx.final",
                    model_fd,
                    "model.onnx",
                    RENAME_EXCHANGE,
                )
                _fsync_directory_fd(model_fd)
                _fsync_directory_fd(stage_fd)
                baseline.model.revalidate_at(
                    stage / "model.onnx.final", "baseline model backup"
                )
                _test_hook(workspace, "model_exchanged", fixture=fixture)
                _renameat2(
                    stage_fd,
                    SUBSTITUTION_SIDECAR_NAME,
                    model_fd,
                    SUBSTITUTION_SIDECAR_NAME,
                    RENAME_NOREPLACE,
                )
                _fsync_directory_fd(model_fd)
                _fsync_directory_fd(stage_fd)
                _test_hook(workspace, "receipt_sidecar_published", fixture=fixture)
                _renameat2(
                    stage_fd,
                    SUBSTITUTION_RECEIPT_NAME,
                    model_fd,
                    SUBSTITUTION_RECEIPT_NAME,
                    RENAME_NOREPLACE,
                )
                _fsync_directory_fd(model_fd)
                _fsync_directory_fd(stage_fd)
                _test_hook(workspace, "receipt_published", fixture=fixture)
                boundary_by_name = {
                    "FULL_CORRECTION_CANDIDATE.json": "legacy_candidate_retired",
                    "FULL_CORRECTION_CANDIDATE.json.sha256": "legacy_sidecar_retired",
                    "FULL_CORRECTION_QUALIFICATION.md": "legacy_qualification_retired",
                }
                for name in LEGACY_EVIDENCE:
                    baseline.legacy[name].revalidate()
                    _renameat2(model_fd, name, backup_fd, name, RENAME_NOREPLACE)
                    _fsync_directory_fd(model_fd)
                    _fsync_directory_fd(backup_fd)
                    baseline.legacy[name].revalidate_at(
                        workspace / TXN_NAME / BACKUP_NAME / name,
                        f"legacy backup {name}",
                    )
                    _test_hook(workspace, boundary_by_name[name], fixture=fixture)
                require(
                    _existing_hash(workspace / MODEL_RELATIVE, "installed model")
                    == artifacts.model.sha256,
                    "installed model changed before activation",
                )
                baseline.contract.revalidate()
                _renameat2(
                    stage_fd,
                    "QualifiedModelContract.cmake.final",
                    cmake_fd,
                    "QualifiedModelContract.cmake",
                    RENAME_EXCHANGE,
                )
                _fsync_directory_fd(cmake_fd)
                _fsync_directory_fd(stage_fd)
                _fsync_directory_fd(workspace_fd)
                baseline.contract.revalidate_at(
                    stage / "QualifiedModelContract.cmake.final",
                    "pending contract backup",
                )
                _test_hook(workspace, "contract_activated", fixture=fixture)
                final = _verify_final_tree(
                    workspace,
                    final_contract_sha256=str(journal["final"]["contract_sha256"]),
                    artifacts=artifacts,
                    substitution=substitution,
                )
                _test_hook(workspace, "final_verified", fixture=fixture)
            except BaseException:
                # Ordinary failures are resolved immediately.  SIGKILL/power
                # loss instead leaves the immutable journal for --recover.
                try:
                    recovered = _recover_any_locked(
                        workspace, fixture=fixture, prefer_pending=True
                    )
                    require(
                        recovered["status"] == "recovered_pending",
                        "ordinary failure did not restore the pending tree",
                    )
                except BaseException as recovery_error:
                    raise SubstitutionError(
                        "substitution failed and safe automatic recovery could not be proven; "
                        "the tree remains fail-closed and requires --recover"
                    ) from recovery_error
                raise
            finally:
                for descriptor in (backup_fd, stage_fd, cmake_fd, model_fd):
                    if descriptor is not None:
                        os.close(descriptor)
            try:
                cleanup = _retire_transaction(workspace, journal, fixture=fixture)
                require(
                    cleanup["status"] == "recovered_final",
                    "final cleanup changed state",
                )
            except BaseException:
                # Contract-last activation is already committed. Resume an
                # interrupted cleanup to the same exact final tree; never try
                # to roll back after its verified recovery payload is retired.
                if any(
                    os.path.lexists(workspace / name)
                    for name in (TXN_NAME, RETIRED_TXN_NAME, CLEANUP_JOURNAL_NAME)
                ):
                    try:
                        recovered = _recover_any_locked(
                            workspace, fixture=fixture, prefer_pending=False
                        )
                        require(
                            recovered["status"]
                            == "recovered_final",
                            "cleanup recovery changed the committed tree",
                        )
                    except BaseException as recovery_error:
                        raise SubstitutionError(
                            "final tree is committed but deterministic cleanup "
                            "requires explicit --recover"
                        ) from recovery_error
                raise
            return {
                "status": "substituted_unpromoted_audition_candidate",
                "final": final,
                "next_gate": "build/listen/qualify on the target Mac; no promotion or push was performed",
            }
    finally:
        try:
            fcntl.flock(workspace_fd, fcntl.LOCK_UN)
        finally:
            os.close(workspace_fd)


def recover_substitution(args: argparse.Namespace) -> Mapping[str, Any]:
    workspace = _absolute_real(args.workspace, "plugin audition workspace", directory=True)
    fixture = _verify_fixture_mode(workspace, args.allow_test_fixtures)
    workspace_fd = _open_directory(workspace, "plugin audition workspace")
    try:
        try:
            fcntl.flock(workspace_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SubstitutionError("another c236 substitution owns the workspace lock") from error
        return _recover_any_locked(workspace, fixture=fixture)
    finally:
        try:
            fcntl.flock(workspace_fd, fcntl.LOCK_UN)
        finally:
            os.close(workspace_fd)


def source_plan() -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "kind": "stemgenrt_c236_audition_substitution_plan_v1",
        "status": "prepared_source_only_inert",
        "canonical_workspace": str(CANONICAL_WORKSPACE),
        "canonical_export_completion": str(CANONICAL_EXPORT_COMPLETION),
        "real_apply_requires": [
            "--apply",
            "--export-completion",
            "--expected-export-receipt-sha256",
            "--expected-final-chain-receipt-sha256",
        ],
        "activation": "durable dependencies then contract-last renameat2 exchange",
        "recovery": "explicit --recover inferred from sealed before/after hashes",
        "mutation_boundaries": list(MUTATION_BOUNDARIES),
        "default_opened_real_artifacts": False,
        "default_mutated_workspace": False,
        "build_performed": False,
        "model_executed": False,
        "gpu_or_audio_used": False,
        "git_operation_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source-check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--recover", action="store_true")
    parser.add_argument("--workspace", type=Path, default=CANONICAL_WORKSPACE)
    parser.add_argument("--export-completion", type=Path)
    parser.add_argument("--expected-export-receipt-sha256")
    parser.add_argument("--expected-final-chain-receipt-sha256")
    parser.add_argument("--allow-test-fixtures", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-static-contract-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.apply:
            result = apply_substitution(args)
        elif args.recover:
            require(args.export_completion is None, "--recover does not accept export paths")
            require(
                args.expected_export_receipt_sha256 is None
                and args.expected_final_chain_receipt_sha256 is None,
                "--recover infers authority from the immutable transaction journal",
            )
            result = recover_substitution(args)
        else:
            require(
                args.export_completion is None
                and args.expected_export_receipt_sha256 is None
                and args.expected_final_chain_receipt_sha256 is None
                and not args.allow_test_fixtures
                and not args.skip_static_contract_test,
                "source-check/default mode accepts no artifact or mutation arguments",
            )
            result = source_plan()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (SubstitutionError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"c236 audition substitution stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
