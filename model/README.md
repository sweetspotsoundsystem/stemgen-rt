# Experimental independent DBV + residual Other

`model.onnx` is one complete FP32 graph containing independent Drums, Bass and
Vocals models, each using its authenticated 40,000-update EMA checkpoint.
Other is computed inside the graph as the previous input hop minus DBV.
There are no shared learned parameters and no extra audio lookahead.

- ONNX SHA256: `0a49388dd2a47986ab9a161c8764c22853b4d601c188ac052eb530f076d513c7`
- Size: 31,946,241 bytes; self-contained, no external tensor file.
- Input: stereo float32 `[1,2,128]`, 44.1 kHz; unchanged input level.
- Output: float32 `[1,4,2,128]`, Drums/Bass/Vocals/Other.
- States: 5 Drums + 4 Bass + 4 Vocals; all zero initially.
- Delay: one graph hop; one invalid pre-roll output and exactly one EOF flush.
- Host: ORT 1.26.0 CPU, sequential, one intra/inter-op thread, spinning disabled,
  KleidiAI disabled, preallocated buffers.

The authoritative identity and shape lock is
`cmake/QualifiedModelContract.cmake`. Its historical QUALIFIED variable prefix
means identity checking; this candidate is experimental and has no physical M4
qualification. Regenerate it using `scripts/generate-independent-contract.py`
with this ONNX file and checkpoint manifest. Loading fails closed on changed
hash, size, names, shapes, tensor types or metadata.

`independent-model.json` records each child's architecture, training provenance,
state shapes and model-state hash. `independent-checkpoints.json` binds all three
checkpoint files and their EMA parameters. Matching training code is included at `training-source/stemgenrt-0.6.2.tar.gz`.

Native PyTorch source-model evaluation improved the combined development mean,
but three Other preservation/leakage guards failed. That source-model quality
result is separate from deployment-graph quality. The independent synthetic
fixtures test every stem, alignment, reset and partial EOF; they do not measure
musical quality or qualify real-time performance. M4/M4 Pro timing remains to be
measured on the actual hardware, including the complete plug-in worker path.
