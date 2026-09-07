#!/usr/bin/env python3
"""Regenerate the synthetic oracle from the authenticated research checkpoint.

Requires the HS-TasNet research checkout with its retained checkpoint/artifacts.
Run on CPU with numpy and PyTorch; this never imports or executes ONNX Runtime.
The checked-in fixture allows ordinary plugin CI to validate without PyTorch.
"""

import argparse
import hashlib
import importlib.util
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
    bindings = {
        "checkpoint": (
            "research/direct/runs/latency11/cropped1024-matched-raw4_control-b4-bf16-lr3e-5/checkpoints/step-000250/model.pt",
            "ac46729e5e4d379b09914a6e40ae927e09089b43fd4eef219ae7e034f355da65"),
        "family": (
            "research/direct/runs/latency11/smoke/cropped1024-ola-prep/cropped1024_ola.py",
            "5d5359e1b25749a4d84db0a30671154378bbf4c91e0b2efaae50894b676d7b27"),
        "base": (
            "research/direct/latency_ola512.py",
            "90cbc93a3ab6c5e37b39442c066e82de7d1012b83b570b55acf57d026f775368"),
    }
    for name, (relative, expected) in bindings.items():
        if digest(root / relative) != expected:
            raise ValueError(f"Authenticated {name} differs")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location("fixture_family", root / bindings["family"][0])
    family = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(family)
    payload = torch.load(root / bindings["checkpoint"][0], map_location="cpu", weights_only=True)
    model = family.Cropped1024OLAModel().eval().requires_grad_(False)
    model.load_state_dict(payload["model"], strict=True)
    if payload["architecture"] != model.architecture_metadata:
        raise ValueError("Saved architecture differs")

    lengths = (1, 255, 256, 257, 511, 512, 513, 16521)
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
            padded = torch.zeros(1, 2, ((length + 255) // 256) * 256)
            padded[..., :length] = torch.from_numpy(audio[:, :length].copy())
            for offset in range(0, padded.shape[-1], 256):
                deployed, state = model.forward_chunk(padded[..., offset:offset+256], state)
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
        "sources": {name: {"research_path": rel, "sha256": sha} for name, (rel, sha) in bindings.items()},
        "model_state_sha256": payload["model_state_sha256"],
        "torch": torch.__version__, "numpy": np.__version__,
        "sample_rate": 44100, "hop": 256, "frames": lengths,
        "source_order": ["drums", "bass", "vocals", "other"],
        "reference": "Original CPU FP32 PyTorch deployed outputs, zero state, one zero flush, cropped to real length. No ONNX execution.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
