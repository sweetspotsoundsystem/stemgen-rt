# Three-worker execution

Source builds and CI artifacts default to one model per persistent inference
worker (three total). `-DSTEMGENRT_DEFAULT_INDEPENDENT_WORKERS=1` builds the
single-worker diagnostic control. Before model
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
On macOS and Linux, helpers poll at 50 microseconds while idle and the gathering
worker sleeps for 10 microseconds between completion checks. On Windows,
helpers block on their atomic pending flag; the primary worker publishes and
notifies each task, then waits on each helper's atomic completion flag. A task
published before a helper begins waiting is retained by the flag. Shutdown
changes the pending value before notifying, then joins both helpers before a
restart clears their flags. These waits and notifications occur on worker or
lifecycle threads. The audio callback still publishes with atomics only.
Epoch changes are handled only
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

On Windows, each inference worker first requests `THREAD_PRIORITY_HIGHEST`
and checks it. Playback workers then register with the system's
[MMCSS Pro Audio task](https://learn.microsoft.com/en-us/windows/win32/api/avrt/nf-avrt-avsetmmthreadcharacteristicsw)
and request `AVRT_PRIORITY_HIGH`. Offline workers skip MMCSS. If registration
or relative priority is unavailable, the worker keeps its verified ordinary
priority and reports `Unsupported` instead of claiming MMCSS acceptance. Each
worker unregisters and restores its original priority on its own thread.
Status `Applied` means the requested API calls succeeded; it is not a readback
of MMCSS's dynamic scheduling decisions.

A primary worker with no queued input uses its own
[high-resolution waitable timer](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw)
for a 100-microsecond idle interval. The timer is created and destroyed on that
worker; creation failure rejects startup and a wait failure stops processing.
This path requires Windows 10 version 1803 or later and the x64 Visual C++
runtime, including `MSVCP140_ATOMIC_WAIT.dll`. It requests no global timer-period
change. Timer expiry and accepted priority requests cannot guarantee a deadline.

The Windows standalone comparison host also uses a high-resolution timer to
pace its original absolute callback schedule. Its synthetic callback requests
MMCSS Pro Audio with `AVRT_PRIORITY_CRITICAL`; the production plugin leaves
its host's callback scheduling to the host. Reports identify
`host_clock_wait=high_resolution_timer` and
`policy_backend=windows_mmcss_pro_audio`. Initial status records acceptance;
final status also requires successful cleanup. Timing qualification requires
MMCSS acceptance for playback workers and the synthetic callback. Every
numerical, reconstruction, missed-block, underrun, zero-wait, late-start and
catch-up gate remains required. An ordinary-priority fallback can process
audio but does not qualify the multimedia scheduling experiment.
The Mac standalone host holds a user-initiated, latency-critical activity for
the duration of a trial, outside the measured callback.

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
verify partition identities. The exporter verifies the exact residual arithmetic
and history update before replacing them with the host formula. CPU/CI correctness
and a short benchmark do not qualify other machines or installed DAW timing.

## Physical-machine evidence (2026-10-24)

All comparisons retain the same FP32 graph/partition weights, thirteen states,
one ORT thread per session, 44.1 kHz and a 128-sample host callback. Separate
machines and operating systems are not a controlled CPU ranking.

| Machine / protocol | One worker, mean complete hop | Three workers, mean complete hop | Actual plugin path |
| --- | ---: | ---: | --- |
| M4 quick, 40,000 measured hops/callbacks per mode | 2.304 ms | 0.694 ms | Zero missed results in either mode |
| M4 sustained, two 30-minute trials per mode, ABBA | — | — | Zero misses/underruns across 1,240,314 measured callbacks per mode |
| M1 Pro quick, 40,000 measured hops/callbacks per mode | 2.244 ms | 1.060 ms | Zero missed results in either mode; no matching endurance run |
| Ryzen 5 5500 / Windows 11, four balanced pairs per stage | 3.246 ms | 1.525 ms | Three workers: 224 missed results/underrun blocks across 40,400 callbacks including warmup; one worker: 40,396 missed results |

The Windows candidate **does not pass the zero-miss playback gate at 128 samples**.
Its parallel native calls all completed within the 2.902 ms hop budget
(maximum 2.584 ms), but 16/40,000 completions missed the original arrival deadline
once start lateness was included. Plugin callbacks themselves never waited or
overran their deadline; no late-start/catch-up events or queue-full drops were
recorded in the parallel trials. Retained stems were present and distinct,
outputs were finite, and reconstruction passed. The remaining 224 missed
results still fail qualification. This synthetic host supplies no audio driver
or installed DAW evidence.

An additional Windows diagnostic used a 256-sample host buffer and the same
validated production library (model hop 128, PDC 384 samples / 8.707 ms).
Across two trials totaling 20,200 callbacks including warmup, the due-boundary
counter was zero but six underrun blocks occurred during measured playback.
This setting also fails the zero-underrun gate; it is not a clean-buffer
recommendation.

The earlier Windows three-worker runtime averaged 3.753 ms and missed 40,396
results across the same callback exposure. The intermediate atomic-wait/timer
candidate averaged 1.537 ms and missed 307. The MMCSS candidate above averaged
1.525 ms and missed 224. The original-to-final comparison changes both runtime
waits and standalone host pacing; it does not isolate their individual effects.
The final within-build one-versus-three comparison uses the same corrected
host for both modes. Windows summary computation/constraint fields retained
from the Mac harness are not MMCSS CPU reservations.

Windows correctness: 206 passed, eight timing/future-rate tests disabled during
ordinary CTest and the selected timing tests run separately. Linux regression:
197 passed, one expected platform skip, eight disabled. Parent waveform/state
parity and callback allocation checks pass. Training/model source is unchanged
from the retained 357-test research and 418-test public CPU suites. The public
source additionally limits a libc++ declaration workaround to its affected
test configuration; timed Windows production source is unchanged.

The M4 Ableton stability report is a user observation; its settings, duration
and host trace were not independently inspected. Windows installed-DAW timing,
Windows endurance, multiple-instance load, and physical Mac timing after this
Windows-only runtime patch remain distinct from CPU/CI correctness.
