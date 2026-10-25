# StemgenRT model identity (introduced in 0.7.0)

The bundled FP32 graph contains independent Drums, Bass and Vocals models,
each using its final 40,000-update parameter EMA. Other is the previous input
hop minus `((Drums + Bass) + Vocals)`. There are no shared learned parameters.

- SHA-256: `0a49388dd2a47986ab9a161c8764c22853b4d601c188ac052eb530f076d513c7`.
- Size: 31,946,241 bytes; self-contained with no external tensor file.
- Input/output: stereo `[1,2,128]` / `[1,4,2,128]` float32 at 44.1 kHz.
- Graph source order: Drums, Bass, Vocals, Other.
- States: five Drums, four Bass and four Vocals, all initially zero.
- Lifecycle: one invalid pre-roll output, previous-hop alignment, one EOF flush.
- Runtime: ORT 1.26.0 CPU, three workers by default in plugin 0.8.0, sequential
  execution and one intra/inter-op thread per session, spinning and KleidiAI
  disabled, preallocated buffers. The parent graph remains bundled for identity
  validation and single-worker diagnostics; parallel playback uses its exact
  three partitions.

`cmake/QualifiedModelContract.cmake` locks the graph identity, geometry and
metadata. Configuration verifies size/hash; loading verifies names, shapes,
types and metadata. Its historical QUALIFIED prefix describes identity checks,
not physical M4 timing certification. The frozen graph's original experimental
schema/key remains unchanged to preserve its verified bytes.

`independent-model.json` records architecture and state shapes;
`independent-checkpoints.json` identifies the three EMA endpoints. The matching
model/training/data/loss/recovery/evaluation/export source is bundled at
`training-source/stemgenrt-0.8.0.tar.gz` with its checksum and source commit,
including the matching partition exporter. The model weights are unchanged.

Independent PyTorch fixtures cover all four stems, reset, alignment and partial
EOF. Full plugin tests cover queue epochs, fallback, offline variable callbacks
and real-time allocation rules. See `release-validation.json` for historical
0.7.0 checks and [parallel-worker evidence](../docs/parallel-workers.md) for the
current runtime. M4 sustained standalone playback passed across 60 measured
minutes per mode, and M1 Pro quick playback passed. Windows Ryzen playback still
had underruns at both tested buffers. The M4 Ableton result is a user report.

Earlier single-worker evidence is retained for comparison:
The returned M4 worker probe missed 7,575 / 105,000 deadlines (7.21%) during its
five-minute released-graph run. A later plugin-path comparison with real-time
synthetic callback and worker observed zero boundary misses across four short
trials and one five-minute trial (145,000 measured callbacks), versus seven with
the same callback and QoS worker. These are different metrics from the earlier
native probe. The newer sustained result concerns the runtime documented above;
independently inspected installed-DAW traces and fresh M4 Pro evidence remain
separate. See [M4 testing](../M4_TESTING.md).
Other can regress because it inherits the net error from the three learned stems.
