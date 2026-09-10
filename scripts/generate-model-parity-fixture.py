#!/usr/bin/env python3
"""Regenerate the synthetic oracle from the authenticated research checkpoint.

Requires the HS-TasNet research checkout with its retained checkpoint/artifacts.
Run on CPU with numpy and PyTorch; this never imports or executes ONNX Runtime.
The checked-in fixture allows ordinary plugin CI to validate without PyTorch.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.research_root.resolve()
    export_plan = root / "research/direct/runs/latency58/leader-cleanup-250-onnx-001/plan.json"
    expected_plan_sha = "59730a0fff133937142cc267b31a6d872a2842c167c9768e472defd50807b124"
    if digest(export_plan) != expected_plan_sha:
        raise ValueError("Authenticated export plan differs")
    plan = json.loads(export_plan.read_text())
    # The original plan binds every research source used to construct/load the
    # native model, including its parent initialization and checkpoint bytes.
    bindings = {}
    for absolute, expected in plan["source_bindings"].items():
        original = Path(absolute)
        if original.suffix not in (".py", ".pt"):
            continue
        original_root = Path("/home/axel/autoresearch/codex/HS-TasNet")
        relative = original.relative_to(original_root) if original.is_relative_to(original_root) else original
        if digest(root / relative) != expected:
            raise ValueError(f"Authenticated input differs: {relative}")
        bindings[str(relative)] = (str(relative), expected)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from research.direct.evaluate_latency58_leader_cleanup import load_evaluation_model
    from research.direct.train_latency58 import state_sha256

    # Use the same authenticated native-model loader as the scored candidate.
    # It verifies the training plan and retained inference generation, including
    # the model's complete ancestry. Export-copy and ORT code are not executed.
    model, receipt = load_evaluation_model(plan)
    model_state_sha = state_sha256(model.state_dict())
    if (model_state_sha != plan["model_state_sha256"]
            or receipt["files"]["model.pt"]["sha256"] != plan["checkpoint"]["sha256"]):
        raise ValueError("Loaded model fingerprint differs")

    lengths = (1, 127, 128, 129, 255, 256, 257, 16521)
    t = np.arange(max(lengths), dtype=np.float64) / 44100.0
    # Distinct channels, low bass, higher partials, chirp and short transients.
    # The initial peak opens the existing writer confidence envelope fully.
    left = 0.19 * np.cos(2 * np.pi * 30 * t) + 0.08 * np.sin(2 * np.pi * 731 * t)
    right = -0.14 * np.cos(2 * np.pi * 43 * t) + 0.07 * np.sin(2 * np.pi * (190 * t + 260 * t*t))
    left[2047:2051] += (0.2, -0.3, 0.15, -0.1)
    right[8191:8195] += (-0.13, 0.27, -0.19, 0.09)
    audio = np.stack((left, right)).astype("<f4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream, torch.inference_mode():
        stream.write(b"SGRTG001" + struct.pack("<I", len(lengths)))
        for length in lengths:
            state = model.initial_state(1)
            hops = []
            padded = torch.zeros(1, 2, ((length + 127) // 128) * 128)
            padded[..., :length] = torch.from_numpy(audio[:, :length].copy())
            for offset in range(0, padded.shape[-1], 128):
                deployed, state = model.forward_chunk(padded[..., offset:offset+128], state)
                if offset:
                    hops.append(deployed)
            deployed, state = model.flush(state)
            hops.append(deployed)
            expected = torch.cat(hops, dim=-1)[0, ..., :length].numpy().astype("<f4")
            stream.write(struct.pack("<I", length))
            stream.write(audio[:, :length].tobytes(order="C"))
            stream.write(expected.tobytes(order="C"))
    provenance = {
        "format": "SGRTG001: LE uint32 case count, then per case LE uint32 frames, planar input[2,T] and deployed[4,2,T] LE float32",
        "fixture_sha256": digest(args.output),
        "generator_sha256": digest(Path(__file__)),
        "export_plan_sha256": expected_plan_sha,
        "sources": {name: {"research_path": rel, "sha256": sha} for name, (rel, sha) in bindings.items()},
        "model_state_sha256": model_state_sha,
        "torch": torch.__version__, "numpy": np.__version__,
        "sample_rate": 44100, "hop": 128, "frames": lengths,
        "source_order": ["drums", "bass", "vocals", "other"],
        "reference": "Original CPU FP32 PyTorch deployed outputs, zero state, one zero flush, cropped to real length. No ONNX execution.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
