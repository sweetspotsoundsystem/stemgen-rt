# Experimental three-worker execution

Build with `-DSTEMGENRT_DEFAULT_INDEPENDENT_WORKERS=3` to try one model per
persistent inference worker. The source build default remains 1. Before model
loading, the paced harness can select 1 or 3 with
`STEMGENRT_PACED_INDEPENDENT_WORKERS`; it records the actual configured count.
The experiment preserves plugin identity, controls, raw levels, 13-state
streaming behavior and the 256-sample latency at 128 samples / 44.1 kHz.

The existing queue worker runs Bass. Two persistent helper threads run Drums
and Vocals. Each has a separate CPU ORT session, sequential execution, one
intra/inter-op thread, spinning off and KleidiAI disabled. The parent combined
session is released after its identity/ABI validation. There are three live
sessions during parallel playback, with no fourth coordinator thread.

The queue worker publishes a task using release/acquire atomics and gathers
both completed helpers before advancing state or publishing the output hop.
Helpers poll at 50 microseconds while idle; the gathering worker sleeps for
10 microseconds between completion checks. Only worker threads do this. The
audio callback neither waits nor wakes them. OS scheduling can extend these
intervals, so they are not hard latency bounds. Epoch changes are handled only
after a complete in-flight task is gathered; old-epoch publication is discarded.
A failed branch invalidates the whole hop and resets all thirteen states.

The graph partitions retain original node operations and initializer bytes.
Each uses its own state subset (Drums 5, Bass 4, Vocals 4). The calling worker
forms `Other = previous_input - ((Drums + Bass) + Vocals)` in the parent's
addition order. Partial EOF, one final graph flush and invalid first pre-roll
are inherited from the existing queue and runtime.

On macOS, the existing worker retains its 2.7 ms computation request within the
128/44100 s period. Each helper requests 1.5 ms within that same period, with
initial/final readbacks; offline preparation uses QoS. AU workgroup updates
fan out through three bounded mailboxes. Each worker owns its membership token
and leaves before restoring its scheduling policy. As before, JUCE telemetry
means join requested, not proof of native membership. More threads compete for
cores and memory; multiple plugin instances require separate physical testing.

The schema-2 per-worker trace measures a single `OrtApi::Run` and one calling
thread's CPU. It is deliberately rejected for three-worker execution rather
than mislabeling dispatch/join time or omitting helper CPU. The first A/B
comparison is uninstrumented; external Instruments captures remain possible
using the named `StemgenRT bass`, `StemgenRT drums`, `StemgenRT vocals` threads.

To reproduce partitions using the matching full model/training package:

```sh
python -m stemgenrt.partition_independent model/model.onnx model/parallel-new
```

Review waveform and state parity before replacing `model/parallel`, then run
`scripts/generate-parallel-contract.py`. CMake, bundle sealing and runtime load
verify partition identities. This is a candidate experiment; CPU/CI correctness
and a short benchmark do not qualify physical M4/M4 Pro or installed DAW timing.
