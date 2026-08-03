#!/usr/bin/env python3
"""Seal the listened-to full-c191 graph for lightweight qualification.

The input is the exact audition ONNX.  This tool authenticates the completed
c213 electronic-holdout terminal, replaces audition-only model metadata, and
publishes a deterministic candidate whose computational GraphProto is byte
identical to the audition graph.  Target-Mac evidence is intentionally kept in
an external receipt so the Mac-tested ONNX bytes never need to change again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping

import onnx
from onnx import numpy_helper


AUDITION_MODEL_SHA256 = (
    "6e817d7a09832072f0d7df4a3cbec0a798804efa4a88fbaabea14a1f70fdcf50"
)
AUDITION_MODEL_BYTES = 114_646_323
SOURCE_MODEL_SHA256 = (
    "07557c7756815c0a84960c02faed4becd31413b53e1e4bb5b28177f2d6a97159"
)
C191_STEP128_PAYLOAD_SHA256 = (
    "a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b"
)
FULL_HEAD_STATE_SHA256 = (
    "238991eced221734b5e931bbc4256412dd6ed361656f2cbe4864abccedd22bcf"
)
FULL_RUNTIME_STATE_SHA256 = (
    "6315b6670f0313ad554e8e3088ced42be6c0cbc285bccb8f5f553c4d8fa367f2"
)
FULL_PROJECTION_SHA256 = (
    "7f3b594e1ea06b6b5f4fe9f188181e908be249b1c918dc782d204cb7bc2f338b"
)
AUDITION_GRAPH_SHA256 = (
    "ddb6f3733f9e65ff60fa0121bbaade071805fbbb4874babffa11789ef27ca6ec"
)
AUDITION_METADATA_CLEARED_MODEL_SHA256 = (
    "0ce19a4d07b8e1de7dc3efeca812e548b6ae709911c7c0f973d5aa4ab3e07f2a"
)
AUDITION_INITIALIZER_MANIFEST_SHA256 = (
    "97a114269bdac7be692e83966914020a7fca8fa70bdee23dd2d02304d43c0eba"
)
SEALED_SCALED_EVALUATION_SHA256 = (
    "94324ca23049cc7dd378ca648292bbb0db7adc9f51f2dfdba522db76ee807edb"
)
PROJECTION_NAME = "runtime.head.output_projection.weight"
CANDIDATE_ID = "c191-step128-full-correction-128-over-128"
DEPLOYMENT_TRACK = "full_correction_lightweight_qualification_track_v1"
QUALITY_POLICY = "c213_electronic_holdout_and_target_macos_10k_paced_v1"
EXPECTED_CHECKS = {
    "aggregate_si": -0.5,
    "aggregate_low": -0.5,
    "per_stem_drums_si": -1.0,
    "per_stem_bass_si": -1.0,
    "per_stem_drums_low": -1.0,
    "per_stem_bass_low": -1.0,
}


class SealError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SealError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def verify_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n"
    require(sidecar.read_text(encoding="ascii") == expected, f"bad sidecar: {sidecar}")


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def authenticate_holdout(terminal: Path) -> dict[str, Any]:
    require(terminal.is_dir() and not terminal.is_symlink(), "holdout terminal missing")
    evaluation_path = terminal / "evaluation.json"
    gate_path = terminal / "gate.json"
    seal_path = terminal / "SEAL.json"
    require(
        {path.name for path in terminal.iterdir()}
        == {
            "evaluation.json",
            "evaluation.json.sha256",
            "gate.json",
            "gate.json.sha256",
            "SEAL.json",
        },
        "holdout terminal inventory changed",
    )
    evaluation_sha, evaluation_bytes = sha256_file(evaluation_path)
    gate_sha, gate_bytes = sha256_file(gate_path)
    seal_sha, seal_bytes = sha256_file(seal_path)
    verify_sidecar(evaluation_path, evaluation_sha)
    verify_sidecar(gate_path, gate_sha)
    evaluation = load_json(evaluation_path)
    gate = load_json(gate_path)
    seal = load_json(seal_path)
    require(
        seal
        == {
            "schema_version": 1,
            "kind": "hs_tasnet_c213_full_c191_lightweight_holdout_terminal_seal_v1",
            "attempt_id": "c213-full-c191-lightweight-electronic-holdout-v1",
            "evaluation_sha256": evaluation_sha,
            "gate_sha256": gate_sha,
            "gate_all_pass": True,
            "file_count": 5,
            "atomic_directory_rename": True,
        },
        "holdout terminal seal is not exact",
    )
    require(
        evaluation.get("kind")
        == "hs_tasnet_c213_full_c191_lightweight_holdout_evaluation_v1"
        and evaluation.get("status") == "complete"
        and evaluation.get("attempt_id")
        == "c213-full-c191-lightweight-electronic-holdout-v1",
        "holdout evaluation identity changed",
    )
    per_track = evaluation.get("per_track")
    require(
        isinstance(per_track, list)
        and len(per_track) == 12
        and all(isinstance(record, Mapping) for record in per_track)
        and len({record.get("track_id") for record in per_track}) == 12,
        "holdout is not the exact 12-track unique inventory",
    )
    require(
        gate.get("kind")
        == "hs_tasnet_c213_full_c191_lightweight_production_gate_v1"
        and gate.get("schema_version") == 1
        and gate.get("all_pass") is True
        and evaluation.get("lightweight_gate") == gate,
        "holdout gate is absent, failed, or not bound into the evaluation",
    )
    checks = gate.get("checks")
    require(
        isinstance(checks, Mapping) and set(checks) == set(EXPECTED_CHECKS),
        "gate inventory changed",
    )
    for name, minimum in EXPECTED_CHECKS.items():
        check = checks[name]
        require(isinstance(check, Mapping), f"gate check is not an object: {name}")
        baseline = finite_number(check.get("baseline_db"), f"{name}.baseline_db")
        candidate = finite_number(check.get("candidate_db"), f"{name}.candidate_db")
        delta = finite_number(check.get("delta_db"), f"{name}.delta_db")
        declared_minimum = finite_number(
            check.get("minimum_delta_db"), f"{name}.minimum_delta_db"
        )
        require(
            abs((candidate - baseline) - delta) <= 1.0e-12
            and declared_minimum == minimum
            and delta >= minimum
            and check.get("pass") is True,
            f"gate check failed exact recomputation: {name}",
        )
    models = evaluation.get("models")
    require(
        isinstance(models, Mapping) and set(models) == {"c91", "full_c191"},
        "model inventory changed",
    )
    full = models["full_c191"]
    require(
        isinstance(full, Mapping)
        and full.get("candidate_id") == "c191-step128-full-correction-128-over-128"
        and full.get("selected_head_sha256") == C191_STEP128_PAYLOAD_SHA256
        and full.get("runtime_model_state_sha256") == FULL_RUNTIME_STATE_SHA256
        and full.get("head_state_sha256") == FULL_HEAD_STATE_SHA256
        and full.get("output_projection_sha256") == FULL_PROJECTION_SHA256,
        "evaluated full-c191 identity changed",
    )
    scaled = evaluation.get("sealed_scaled_c193_evidence")
    reproduction_delta = finite_number(
        scaled.get("fresh_c91_aggregate_max_abs_delta_db")
        if isinstance(scaled, Mapping)
        else None,
        "fresh c91 reproduction delta",
    )
    require(
        isinstance(scaled, Mapping)
        and scaled.get("sha256") == SEALED_SCALED_EVALUATION_SHA256
        and 0.0 <= reproduction_delta <= 1.0e-9,
        "sealed scaled-c193 comparison or fresh c91 reproduction changed",
    )
    comparisons = evaluation.get("comparisons")
    require(
        isinstance(comparisons, Mapping)
        and set(comparisons)
        == {
            "full_c191_vs_fresh_c91",
            "full_c191_vs_sealed_scaled_c193",
            "fresh_c91_vs_sealed_c197_c91",
        },
        "holdout comparison inventory changed",
    )
    return {
        "evaluation_sha256": evaluation_sha,
        "evaluation_bytes": evaluation_bytes,
        "gate_sha256": gate_sha,
        "gate_bytes": gate_bytes,
        "terminal_seal_sha256": seal_sha,
        "terminal_seal_bytes": seal_bytes,
        "track_count": len(per_track),
        "all_six_checks_pass": True,
        "checks": checks,
        "comparisons": comparisons,
    }


def deterministic_proto_bytes(value: Any) -> bytes:
    return value.SerializeToString(deterministic=True)


def metadata_cleared_model_sha256(model: onnx.ModelProto) -> str:
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    del clone.metadata_props[:]
    return sha256_bytes(deterministic_proto_bytes(clone))


def initializer_manifest_sha256(model: onnx.ModelProto) -> str:
    digest = hashlib.sha256()
    for tensor in model.graph.initializer:
        name = tensor.name.encode("utf-8")
        payload = deterministic_proto_bytes(tensor)
        digest.update(struct.pack(">Q", len(name)))
        digest.update(name)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def seal_model(
    input_model: Path, output_model: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    source_sha, source_bytes = sha256_file(input_model)
    require(
        source_sha == AUDITION_MODEL_SHA256 and source_bytes == AUDITION_MODEL_BYTES,
        "input is not the exact listened-to audition ONNX",
    )
    model = onnx.load_model(str(input_model), load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    require(
        all(
            initializer.data_location == onnx.TensorProto.DEFAULT
            and len(initializer.external_data) == 0
            for initializer in model.graph.initializer
        ),
        "input model unexpectedly uses external initializer data",
    )
    graph_before = deterministic_proto_bytes(model.graph)
    graph_sha = sha256_bytes(graph_before)
    metadata_cleared_sha = metadata_cleared_model_sha256(model)
    initializer_manifest_sha = initializer_manifest_sha256(model)
    require(
        graph_sha == AUDITION_GRAPH_SHA256
        and metadata_cleared_sha == AUDITION_METADATA_CLEARED_MODEL_SHA256
        and initializer_manifest_sha == AUDITION_INITIALIZER_MANIFEST_SHA256,
        "audition computational identity anchors changed",
    )
    initializer_manifest = {
        value.name: sha256_bytes(deterministic_proto_bytes(value))
        for value in model.graph.initializer
    }
    require(len(initializer_manifest) == len(model.graph.initializer), "duplicate initializer name")
    require(PROJECTION_NAME in initializer_manifest, "full projection initializer missing")
    projection = next(
        value for value in model.graph.initializer if value.name == PROJECTION_NAME
    )
    projection_array = numpy_helper.to_array(projection)
    require(
        tuple(projection_array.shape) == (4, 8, 1)
        and str(projection_array.dtype) == "float32"
        and sha256_bytes(projection_array.tobytes(order="C"))
        == FULL_PROJECTION_SHA256,
        "full output projection bits changed",
    )

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    require(
        metadata.get("hs_tasnet.c193.candidate_id")
        == "c191-step128-full-correction-audition"
        and metadata.get("hs_tasnet.deployment.status")
        == "audition_only_unqualified_full_c191_correction"
        and metadata.get("hs_tasnet.audition.source_model_sha256")
        == SOURCE_MODEL_SHA256
        and metadata.get("hs_tasnet.audition.source_checkpoint_sha256")
        == C191_STEP128_PAYLOAD_SHA256
        and metadata.get("hs_tasnet.audition.output_projection")
        == "drums128_bass128_over128"
        and metadata.get("hs_tasnet.audition.changed_initializer") == PROJECTION_NAME,
        "audition metadata changed",
    )
    for key in tuple(metadata):
        if key.startswith("hs_tasnet.audition."):
            del metadata[key]
    metadata.update(
        {
            "hs_tasnet.c193.candidate_id": CANDIDATE_ID,
            "hs_tasnet.deployment.status": DEPLOYMENT_TRACK,
            "hs_tasnet.deployment.quality_guards": QUALITY_POLICY,
            "hs_tasnet.full_correction.source_model_sha256": SOURCE_MODEL_SHA256,
            "hs_tasnet.full_correction.source_checkpoint_sha256": C191_STEP128_PAYLOAD_SHA256,
            "hs_tasnet.full_correction.output_projection": "drums128_bass128_over128",
            "hs_tasnet.full_correction.changed_initializer": PROJECTION_NAME,
            "hs_tasnet.full_correction.output_projection_sha256": FULL_PROJECTION_SHA256,
            "hs_tasnet.c213.evaluation_sha256": str(evidence["evaluation_sha256"]),
            "hs_tasnet.c213.gate_sha256": str(evidence["gate_sha256"]),
            "hs_tasnet.c213.terminal_seal_sha256": str(evidence["terminal_seal_sha256"]),
        }
    )
    del model.metadata_props[:]
    for key in sorted(metadata):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = metadata[key]
    require(deterministic_proto_bytes(model.graph) == graph_before, "graph changed in memory")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_model.name}.", suffix=".tmp", dir=output_model.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        onnx.save_model(model, str(temporary), save_as_external_data=False)
        published = onnx.load_model(str(temporary), load_external_data=False)
        onnx.checker.check_model(published, full_check=True)
        require(
            deterministic_proto_bytes(published.graph) == graph_before,
            "serialized computational graph changed",
        )
        require(
            metadata_cleared_model_sha256(published) == metadata_cleared_sha
            and initializer_manifest_sha256(published) == initializer_manifest_sha,
            "serialized compute-only model or ordered initializer manifest changed",
        )
        published_initializers = {
            value.name: sha256_bytes(deterministic_proto_bytes(value))
            for value in published.graph.initializer
        }
        require(published_initializers == initializer_manifest, "initializer bits changed")
        require(
            {entry.key: entry.value for entry in published.metadata_props} == metadata,
            "published metadata changed",
        )
        candidate_sha, candidate_bytes = sha256_file(temporary)
        os.replace(temporary, output_model)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_version": 1,
        "kind": "stemgenrt_c191_full_correction_lightweight_candidate_seal_v1",
        "candidate_id": CANDIDATE_ID,
        "deployment_track": DEPLOYMENT_TRACK,
        "quality_policy": QUALITY_POLICY,
        "source_model_sha256": source_sha,
        "source_model_bytes": source_bytes,
        "candidate_model_sha256": candidate_sha,
        "candidate_model_bytes": candidate_bytes,
        "computational_graph_sha256": graph_sha,
        "computational_graph_byte_identical": True,
        "metadata_cleared_model_sha256": metadata_cleared_sha,
        "initializer_count": len(initializer_manifest),
        "ordered_initializer_manifest_sha256": initializer_manifest_sha,
        "all_initializers_bit_identical": True,
        "output_projection_sha256": FULL_PROJECTION_SHA256,
        "holdout": dict(evidence),
        "target_mac_10k_paced_test": "required_on_exact_candidate_model_sha256",
    }


def write_bytes_atomic(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    digest = sha256_bytes(encoded)
    write_bytes_atomic(path, encoded)
    write_bytes_atomic(
        path.with_name(path.name + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-terminal", type=Path, required=True)
    parser.add_argument("--input-model", type=Path, default=Path("model/model.onnx"))
    parser.add_argument("--output-model", type=Path, default=Path("model/model.onnx"))
    parser.add_argument(
        "--candidate-receipt",
        type=Path,
        default=Path("model/FULL_CORRECTION_CANDIDATE.json"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    evidence = authenticate_holdout(arguments.holdout_terminal.resolve())
    receipt = seal_model(arguments.input_model, arguments.output_model, evidence)
    receipt_sha256 = write_json_atomic(arguments.candidate_receipt, receipt)
    print(
        json.dumps(
            {
                "candidate_receipt": receipt,
                "candidate_receipt_sha256": receipt_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
