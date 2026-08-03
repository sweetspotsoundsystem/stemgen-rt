#!/usr/bin/env python3
"""Create and verify the full-strength c191 correction audition ONNX.

This intentionally changes one initializer in the sealed c212 graph.  The
result is ABI-compatible with the qualified model but is audition-only: the
full c191 correction did not pass the later c193 production guard screen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper


SOURCE_MODEL_SHA256 = (
    "07557c7756815c0a84960c02faed4becd31413b53e1e4bb5b28177f2d6a97159"
)
C191_STEP128_PAYLOAD_SHA256 = (
    "a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b"
)
C193_SELECTED_HEAD_SHA256 = (
    "e353bc8cdd2534c4ebbeaf658dfb5a167f574e4f2998adc4e02f609aa04b4ca3"
)
CURRENT_PROJECTION_SHA256 = (
    "ea9686f34466d37cd275c974bc7c33e82b40a5524d0b0ba1c2affb1b55583801"
)
FULL_PROJECTION_SHA256 = (
    "7f3b594e1ea06b6b5f4fe9f188181e908be249b1c918dc782d204cb7bc2f338b"
)
FULL_HEAD_STATE_SHA256 = (
    "238991eced221734b5e931bbc4256412dd6ed361656f2cbe4864abccedd22bcf"
)
FULL_RUNTIME_STATE_SHA256 = (
    "6315b6670f0313ad554e8e3088ced42be6c0cbc285bccb8f5f553c4d8fa367f2"
)
PARENT_STATE_SHA256 = (
    "7bfd9291ee036ee9cc517483bb8c1237f2ef6ed4ede0e5447c20607716075039"
)
PROJECTION_NAME = "runtime.head.output_projection.weight"
INPUT_NAMES = (
    "audio_chunk",
    "past_audio",
    "fusion_hidden",
    "c130_history",
    "previous_hidden",
    "adapter_valid",
    "raw_parent_history",
    "emitted_db_history",
)
OUTPUT_NAMES = (
    "separated_chunk",
    "next_past_audio",
    "next_fusion_hidden",
    "next_c130_history",
    "next_previous_hidden",
    "next_adapter_valid",
    "next_raw_parent_history",
    "next_emitted_db_history",
)
STATE_SHAPES = (
    (1, 2, 512),
    (2, 1, 1000),
    (1, 20, 128),
    (1, 32, 512),
    (1, 1),
    (1, 4, 2048),
    (1, 4, 2048),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_shapes(values: list[Any]) -> list[list[int]]:
    return [list(value.shape) for value in values]


def import_materializer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("c194_materializer_for_audition", path)
    require(spec is not None and spec.loader is not None, "materializer import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def value_info_shape(value: Any) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def render_ort(session: ort.InferenceSession, audio: np.ndarray) -> list[np.ndarray]:
    states = [np.zeros(shape, dtype=np.float32) for shape in STATE_SHAPES]
    collected: list[list[np.ndarray]] = [[] for _ in OUTPUT_NAMES]
    for current in audio:
        outputs = session.run(
            list(OUTPUT_NAMES),
            dict(zip(INPUT_NAMES, [current, *states], strict=True)),
        )
        require(len(outputs) == 8, "ORT output count changed")
        require(all(np.isfinite(value).all() for value in outputs), "non-finite ORT output")
        for index, value in enumerate(outputs):
            collected[index].append(np.asarray(value, dtype=np.float32).copy())
        states = [np.asarray(value, dtype=np.float32).copy() for value in outputs[1:]]
    return [np.stack(values, axis=0) for values in collected]


def render_eager(runtime: Any, audio: np.ndarray) -> list[np.ndarray]:
    states = tuple(runtime.initial_state(1, device="cpu", dtype=torch.float32))
    collected: list[list[np.ndarray]] = [[] for _ in OUTPUT_NAMES]
    with torch.inference_mode():
        for current_np in audio:
            current = torch.from_numpy(np.ascontiguousarray(current_np))
            outputs = runtime(current, *states)
            separated = outputs[0]
            other = current - torch.cat(
                (separated[:, 0:1], separated[:, 1:2], separated[:, 2:3]), dim=1
            ).sum(dim=1)
            outputs = (
                torch.cat(
                    (
                        separated[:, 0:1],
                        separated[:, 1:2],
                        separated[:, 2:3],
                        other.unsqueeze(1),
                    ),
                    dim=1,
                ),
                *outputs[1:],
            )
            require(len(outputs) == 8, "eager output count changed")
            require(all(bool(torch.isfinite(value).all()) for value in outputs), "non-finite eager output")
            for index, value in enumerate(outputs):
                collected[index].append(value.detach().cpu().numpy().copy())
            states = tuple(value.detach().clone() for value in outputs[1:])
    return [np.stack(values, axis=0) for values in collected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selected-head", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    args = parser.parse_args()

    require(file_sha256(args.source_model) == SOURCE_MODEL_SHA256, "source c212 SHA changed")
    require(
        file_sha256(args.checkpoint) == C191_STEP128_PAYLOAD_SHA256,
        "c191 step-128 payload SHA changed",
    )
    require(
        file_sha256(args.selected_head) == C193_SELECTED_HEAD_SHA256,
        "c193 selected-head SHA changed",
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    selected = torch.load(args.selected_head, map_location="cpu", weights_only=True)
    full_state = checkpoint["head_state"]
    selected_state = selected["head_state"]
    require(full_state.keys() == selected_state.keys(), "head tensor inventory changed")
    require(
        all(
            torch.equal(full_state[name], selected_state[name])
            for name in full_state
            if name != "output_projection.weight"
        ),
        "a non-projection c193 head tensor differs from c191 step 128",
    )
    full_projection = (
        full_state["output_projection.weight"].detach().cpu().contiguous().numpy()
    )
    selected_projection = (
        selected_state["output_projection.weight"].detach().cpu().contiguous().numpy()
    )
    require(full_projection.shape == (4, 8, 1), "full projection shape changed")
    require(full_projection.dtype == np.float32, "full projection dtype changed")
    require(array_sha256(full_projection) == FULL_PROJECTION_SHA256, "full projection SHA changed")
    require(
        array_sha256(selected_projection) == CURRENT_PROJECTION_SHA256,
        "selected projection SHA changed",
    )
    expected_selected = full_projection.copy()
    expected_selected[0:2] *= np.float32(7.0 / 128.0)
    expected_selected[2:4] *= np.float32(16.0 / 128.0)
    require(
        np.array_equal(expected_selected.view(np.uint32), selected_projection.view(np.uint32)),
        "selected projection is not the exact d7/b16 fold of W128",
    )

    model = onnx.load(args.source_model, load_external_data=False)
    source_inputs = [(value.name, value_info_shape(value)) for value in model.graph.input]
    source_outputs = [(value.name, value_info_shape(value)) for value in model.graph.output]
    require(tuple(name for name, _ in source_inputs) == INPUT_NAMES, "source input ABI changed")
    require(tuple(name for name, _ in source_outputs) == OUTPUT_NAMES, "source output ABI changed")
    require(
        all(
            initializer.data_location == onnx.TensorProto.DEFAULT
            and len(initializer.external_data) == 0
            for initializer in model.graph.initializer
        ),
        "source model unexpectedly uses external initializer data",
    )
    matches = [
        (index, initializer)
        for index, initializer in enumerate(model.graph.initializer)
        if initializer.name == PROJECTION_NAME
    ]
    require(len(matches) == 1, "projection initializer is missing or duplicated")
    projection_index, projection = matches[0]
    current_projection = np.asarray(numpy_helper.to_array(projection))
    require(
        np.array_equal(current_projection.view(np.uint32), selected_projection.view(np.uint32)),
        "source ONNX does not contain the exact c193 selected projection",
    )
    model.graph.initializer[projection_index].CopyFrom(
        numpy_helper.from_array(full_projection, name=PROJECTION_NAME)
    )

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    require(
        metadata.get("hs_tasnet.c193.candidate_id")
        == "c193-selected-drums7-bass16-over128-c191-step128",
        "source candidate metadata changed",
    )
    metadata.update(
        {
            "hs_tasnet.c193.candidate_id": "c191-step128-full-correction-audition",
            "hs_tasnet.c193.selected_head_sha256": C191_STEP128_PAYLOAD_SHA256,
            "hs_tasnet.c193.selected_head_state_sha256": FULL_HEAD_STATE_SHA256,
            "hs_tasnet.c193.full_runtime_state_sha256": FULL_RUNTIME_STATE_SHA256,
            "hs_tasnet.deployment.quality_guards": "none_full_c191_correction_audition_only",
            "hs_tasnet.deployment.status": "audition_only_unqualified_full_c191_correction",
            "hs_tasnet.audition.source_model_sha256": SOURCE_MODEL_SHA256,
            "hs_tasnet.audition.source_checkpoint_sha256": C191_STEP128_PAYLOAD_SHA256,
            "hs_tasnet.audition.output_projection": "drums128_bass128_over128",
            "hs_tasnet.audition.changed_initializer": PROJECTION_NAME,
        }
    )
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output_model, save_as_external_data=False)

    published = onnx.load(args.output_model, load_external_data=False)
    onnx.checker.check_model(published, full_check=True)
    require(
        [(value.name, value_info_shape(value)) for value in published.graph.input]
        == source_inputs,
        "published input ABI changed",
    )
    require(
        [(value.name, value_info_shape(value)) for value in published.graph.output]
        == source_outputs,
        "published output ABI changed",
    )
    published_projection = np.asarray(
        numpy_helper.to_array(
            next(value for value in published.graph.initializer if value.name == PROJECTION_NAME)
        )
    )
    require(
        np.array_equal(published_projection.view(np.uint32), full_projection.view(np.uint32)),
        "published full projection bits changed",
    )
    require(
        all(
            initializer.data_location == onnx.TensorProto.DEFAULT
            and len(initializer.external_data) == 0
            for initializer in published.graph.initializer
        ),
        "published model unexpectedly uses external initializer data",
    )

    materializer = import_materializer(args.materializer)
    chain = materializer._validate_payloads(materializer.DEFAULT_PATHS)
    runtime, _ = chain.runtime_module.build_exact_runtime(device=torch.device("cpu"))
    chain.checkpoint.restore_head_state(runtime.head, chain.step128_payload)
    runtime.eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    require(
        chain.runtime_module.head_state_sha256(runtime) == FULL_HEAD_STATE_SHA256,
        "full eager head identity changed",
    )
    require(
        chain.runtime_module.tensor_state_sha256(runtime.state_dict())
        == FULL_RUNTIME_STATE_SHA256,
        "full eager runtime identity changed",
    )
    require(
        chain.runtime_module.parent_state_sha256(runtime) == PARENT_STATE_SHA256,
        "parent runtime identity changed",
    )

    rng = np.random.default_rng(0xC191)
    random_audio = rng.normal(0.0, 0.08, size=(12, 1, 2, 512)).astype(np.float32)
    time = np.arange(12 * 512, dtype=np.float32) / np.float32(44100.0)
    low_audio = (
        np.float32(0.125)
        * np.sin(np.float32(2.0 * np.pi * 50.0) * time)
    ).reshape(12, 1, 1, 512)
    low_audio = np.repeat(low_audio, 2, axis=2).astype(np.float32)
    audio = np.concatenate((random_audio, low_audio), axis=0)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.output_model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    require(
        [(value.name, list(value.shape)) for value in session.get_inputs()]
        == source_inputs,
        "ORT input ABI changed",
    )
    require(
        [(value.name, list(value.shape)) for value in session.get_outputs()]
        == source_outputs,
        "ORT output ABI changed",
    )
    eager = render_eager(runtime, audio)
    observed = render_ort(session, audio)
    replay = render_ort(session, audio)
    max_errors = [float(np.max(np.abs(lhs - rhs))) for lhs, rhs in zip(eager, observed)]
    require(max(max_errors) <= 2.0e-5, "eager/ORT parity exceeded 2e-5")
    require(
        all(np.array_equal(lhs.view(np.uint32), rhs.view(np.uint32)) for lhs, rhs in zip(observed, replay)),
        "zero-state ORT replay was not bit-exact",
    )
    mixture = audio
    separated = observed[0]
    reconstruction = separated.sum(axis=2)
    residual_max_abs = float(np.max(np.abs(reconstruction - mixture)))
    require(residual_max_abs <= 2.0e-6, "mixture residual exceeded 2e-6")
    state_liveness = [
        bool(np.any(value.view(np.uint32) != 0)) for value in observed[1:]
    ]
    require(all(state_liveness), "one or more recurrent state families remained dead")

    result = {
        "schema_version": 1,
        "kind": "stemgenrt_c191_full_correction_audition_materialization_v1",
        "deployment_status": "audition_only_unqualified_full_c191_correction",
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "output_model_sha256": file_sha256(args.output_model),
        "output_model_bytes": args.output_model.stat().st_size,
        "c191_step128_payload_sha256": C191_STEP128_PAYLOAD_SHA256,
        "changed_initializer": PROJECTION_NAME,
        "changed_initializer_shape": list(full_projection.shape),
        "source_projection_sha256": CURRENT_PROJECTION_SHA256,
        "output_projection_sha256": FULL_PROJECTION_SHA256,
        "drums_scale_numerator": 128,
        "bass_scale_numerator": 128,
        "scale_denominator": 128,
        "all_other_head_tensors_bit_exact": True,
        "graph_abi_unchanged": True,
        "external_data": False,
        "parity_hops": int(audio.shape[0]),
        "all_eight_outputs_eager_ort_max_abs": max(max_errors),
        "per_output_eager_ort_max_abs": max_errors,
        "reset_replay_bit_exact": True,
        "all_seven_states_live": True,
        "mixture_reconstruction_max_abs": residual_max_abs,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
