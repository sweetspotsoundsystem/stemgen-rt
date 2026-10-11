# StemgenRT 0.7.0 model

The bundled FP32 graph contains independent Drums, Bass and Vocals models,
each using its final 40,000-update parameter EMA. Other is the previous input
hop minus `((Drums + Bass) + Vocals)`. There are no shared learned parameters.

- SHA-256: `0a49388dd2a47986ab9a161c8764c22853b4d601c188ac052eb530f076d513c7`.
- Size: 31,946,241 bytes; self-contained with no external tensor file.
- Input/output: stereo `[1,2,128]` / `[1,4,2,128]` float32 at 44.1 kHz.
- Graph source order: Drums, Bass, Vocals, Other.
- States: five Drums, four Bass and four Vocals, all initially zero.
- Lifecycle: one invalid pre-roll output, previous-hop alignment, one EOF flush.
- Runtime: ORT 1.26.0 CPU, one worker, sequential execution, one intra/inter-op
  thread, spinning and KleidiAI disabled, preallocated buffers.

`cmake/QualifiedModelContract.cmake` locks the graph identity, geometry and
metadata. Configuration verifies size/hash; loading verifies names, shapes,
types and metadata. Its historical QUALIFIED prefix describes identity checks,
not physical M4 timing certification. The frozen graph's original experimental
schema/key remains unchanged to preserve its verified bytes.

`independent-model.json` records architecture and state shapes;
`independent-checkpoints.json` identifies the three EMA endpoints. The matching
model/training/data/loss/recovery/evaluation/export source is bundled at
`training-source/stemgenrt-0.7.0.tar.gz` with its checksum.

Independent PyTorch fixtures cover all four stems, reset, alignment and partial
EOF. Full plugin tests cover queue epochs, fallback, offline variable callbacks
and real-time allocation rules. See `release-validation.json` for release checks.
Physical M4/M4 Pro sustained timing and DAW deadline measurements remain pending.
Other can regress because it inherits the net error from the three learned stems.
