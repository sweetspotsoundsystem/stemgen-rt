# Test StemgenRT 0.7.0 on M4

Use native arm64 macOS 14+, 44.1 kHz and the released independent-model graph
(SHA-256 starts `0a49388dd2a4`). It has 13 states and a 128-sample hop. With a
128-sample host buffer the graph-plus-worker latency is 256 samples. Previous
model measurements do not establish this model's real-time performance.

## Returned worker evidence

One physical Apple M4 run on October 20, 2026 used this exact released graph,
ORT 1.26.0, one worker and user-interactive QoS at 128 samples / 44.1 kHz. Its
returned checksums, independent PyTorch waveform/state preflight and raw timing
statistics were verified on October 22. The hop budget is 2.902494 ms.

| Released graph | Mean loop | p99 completion from arrival | Deadline misses |
| --- | ---: | ---: | ---: |
| Six 10,000-hop segments | 2.3053 ms | 4.2715 ms | 4,636 / 60,000 (7.73%) |
| 105,000 continuous hops | 2.3173 ms | 3.9520 ms | 7,575 / 105,000 (7.21%) |

Of the endurance misses, 6,579 occurred with loop compute inside one hop. This
motivates separate scheduling and compute investigations. A layout-only graph
candidate had slightly lower mean cost but more misses and a long late burst;
it was not promoted. This single run does not establish causation, plugin
underruns or DAW behavior. Fresh M4 Pro evidence is still needed.

## Returned plugin scheduling comparison

The October 23, 2026 packet ran the complete plugin callback path on one physical
Apple M4. Its input identities, result checksums, twenty trial logs, failure
phases and both raw timing traces were independently checked. The released graph,
ORT settings and 256-sample latency were identical across conditions.

| Synthetic callback | Worker | Misses / 40,000 measured short callbacks | Short warmup misses | Misses / 105,000 continuous callbacks |
| --- | --- | ---: | ---: | ---: |
| QoS | QoS | 5 | 1 | Not run |
| QoS | Real-time | 2 | 0 | Not run |
| Real-time | QoS | 2 | 0 | 5 |
| Real-time | Real-time | 0 | 0 | 0 |

Each short condition comprises four separate 10,000-callback trials and four
100-callback warmups. The continuous trial lasts about 305 seconds. Both
real-time threads passed every original gate in all four short trials and the
continuous trial: 145,000 measured callbacks altogether, not one continuous
145,000-callback run. This supports further qualification of the real-time
worker; it does not establish longer playback or installed-DAW reliability.

In the separate traced QoS-worker diagnostic, the first of two misses combined
a 2.122–2.131 ms dispatch delay with a 2.434 ms inference call. The real-time
worker's maximum dispatch upper bound in its traced diagnostic was 0.113 ms.
These instrumented observations explain individual events; they are excluded
from the table's uninstrumented timing evidence.

The native probe stopped at a QoS error in its real-time preflight, so it supplied
no completed scheduling comparison, workgroup measurements or operator profile.
Its QoS waveform/all-state preflight passed. Earlier native completion misses
and the table's plugin due-boundary misses are different metrics. Preserve both
records; do not describe this as a measured reduction from 7.21% to zero.
Portable aggregates and input identities are in
[`model/m4-scheduling-20261023.json`](model/m4-scheduling-20261023.json).

## Controlled scheduling comparison

After downloading the pinned SDK, run `scripts/compare-paced-scheduling-macos.py`
as shown in the README. Each short-trial round contains all four combinations
of callback QoS/real-time and worker QoS/real-time. Every condition occupies
every trial position once. The two five-minute endurance trials hold callback
policy at real-time and compare worker QoS with worker real-time. They run
sequentially, so time and thermal effects may influence their difference.

The worker requests 2.7 ms compute in each 2.902494 ms period. The synthetic
callback requests 0.5 ms compute within a 1 ms constraint at the same period;
XNU requires computation of at least half the constraint. Both threads verify
their requested policy at the start and end. These are diagnostic controls;
the released plugin continues to request its existing QoS policy.

Main comparisons have worker tracing disabled. Two separate traced diagnostics
run last and save callback/worker timestamps to CSV after joining the worker.
These instrumented runs help explain dispatch and compute delays and are kept
separate from uninstrumented deadline measurements. The trace's CPU clock covers
all process threads, not just inference. Every original zero-miss, zero-wait,
finite-output and reconstruction gate remains required. A completed comparison
can contain failed timing gates; inspect `results.json` and each retained log.

The standalone native probe in the matching model repository also compares a
standalone audio workgroup and records an independent ORT operator profile.
Neither harness joins an AU host's workgroup or measures an installed audio
driver. Apple's [Mach scheduling guide](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/KernelProgramming/scheduler/scheduler.html)
describes soft real-time scheduling; the [audio workgroups documentation](https://developer.apple.com/documentation/audiotoolbox/understanding-audio-workgroups)
describes collaboration with the host.

For sustained qualification of the leading candidate, add `--protocol long-soak`
to the comparison command. This builds and checks the same plugin, then runs a
30-minute real-time-worker trial, a 30-minute QoS-worker control, and another
30-minute real-time-worker trial. Every trial keeps the synthetic callback on
real-time policy, uses 620,157 measured callbacks plus 100 warmups, and disables
tracing. Allow roughly 90 minutes after compilation. The result distinguishes
candidate gates from control gates, preserves failed controls, and records
power/thermal snapshots around each trial. Sequential ordering still leaves
possible time and thermal effects. A passing result does not install or promote
a plugin; real host testing remains necessary.

## Correctness and callback timing

From a clean checkout:

```sh
git lfs pull
./scripts/qualify-macos.sh "$HOME/Desktop/stemgenrt-070-qualification"
bash scripts/diagnose-runtime-parity.sh . "$HOME/Desktop/stemgenrt-070-parity"
```

Use new result directories. Qualification builds Release with official ORT
1.26.0, runs native correctness tests, checks AU/VST3 bundles and measures the
paced callback/worker path. The diagnostic compares independent PyTorch output
with three ORT configurations. The deployed `kleidiai_disabled` mode must pass
every case; a zero script exit alone is not a per-case numerical verdict.

## Sustained playback

```sh
bash scripts/extended-soak-macos.sh . "$HOME/Desktop/stemgenrt-070-soak"
```

This runs two untraced 30-minute synthetic-host measurements and retains source,
model, machine, exit status and fallback counts. Use a separate directory with
`--trace` for diagnostics; tracing changes measurement overhead.

Install with `./scripts/install-plugins.sh --release`, restart your DAW and test
at 44.1 kHz with a documented buffer size and workload. Record startup/reset
fallback increments separately from steady playback. Check at least 30 minutes
of playback, seeks, loops, start/stop and all output buses. Retain the previous
release outside the plug-in directories if you need to roll back.
