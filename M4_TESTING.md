# Model tests on M4

This candidate exports the frozen StemgenRT-5.8 teacher004 EMA checkpoint.
The graph SHA-256 begins `77164d6a581f`. It retains seventeen integer products,
`mlas.disable_kleidiai=1`, one inference worker, eight states and 128-sample
hops at 44.1 kHz. Graph plus host delay remains 256 samples with a 128-sample
host buffer. The teacher is not present in the plugin.

Read [the model report](model/README.md) for this graph's measured quality and
Linux correctness. Physical M4 parity, sustained fallback counts and listening
acceptance must be measured for these weights. Historical M4 evidence is
retained under [the v0.6.1 baseline](model/baseline-v0.6.1/README.md).

## Correctness and identity

Use a clean checkout of this candidate on native arm64 macOS 14 or newer.
From the repository root, fetch its LFS model and run the existing full
qualification script with a new evidence directory outside the source tree:

```bash
git lfs pull
./scripts/qualify-macos.sh "$HOME/Desktop/stemgen-teacher004-qualification-001"
bash scripts/diagnose-runtime-parity.sh . "$HOME/Desktop/stemgen-teacher004-parity-001"
```

The qualification script makes a fresh Release build with official ONNX
Runtime 1.26.0, checks the complete native suite and AU/VST3 bundles, and runs
the short paced callback/worker measurement. The model contract and independent
fixture must match this checkout. Both ordinary PyTorch parity tests must pass.

The diagnostic retains three modes: ORT defaults, `kleidiai_disabled`, and
graph optimizations disabled. The plugin uses `kleidiai_disabled`; every case
in that mode must pass. The `default` label describes ORT's backend defaults.
`diagnostic-exit.txt = 0` means that measurements completed; inspect each
`parity_pass` in `results.jsonl`. Keep `identity.txt`, `stderr.log`, actual
compiler/process exits and all per-stem errors. These records bind the graph,
fixture, runtime path/hash/build and CPU capabilities.

## Sustained timing

From the same committed checkout and verified Release build, run two untraced
30-minute synthetic-host measurements:

```bash
bash scripts/extended-soak-macos.sh . "$HOME/Desktop/stemgen-teacher004-soak-001"
```

The runner retains both raw logs, actual exits, machine/model/source identities,
and startup and measured fallback counts. Keep its strict pass/fail result.
For a separate diagnostic of acquisition, inference and publication delays,
use another directory and enable the complete-duration trace:

```bash
bash scripts/extended-soak-macos.sh . "$HOME/Desktop/stemgen-teacher004-trace-001" --trace
```

The trace allocates about 60 MB for 30 minutes and rejects requests beyond a
128 MiB bound before starting the worker. It reports omitted timing samples,
matched and missing requests, and bounded failure-event omissions. Worker
statistics cover matched requests; total process CPU time includes other
threads. Tracing adds clock reads and memory traffic, so retain the untraced
runs as the timing measurements.

Measure this graph with its production runtime setting. Timing from the previous
checkpoint does not qualify the candidate.

## Installed AU in Ableton

After correctness passes, preserve the prior complete bundle, install with
`./scripts/install-plugins.sh --release`, and restart Ableton. Use the confirmed
**M4 / 44.1 kHz / 128-sample buffer** setup with one inference worker and a
documented DAW load. Record the candidate revision, graph identity, duration,
and starting/ending cumulative fallback counters. Repeat at least 30 minutes
of steady playback and require **zero additional steady-playback fallback**.
Keep startup/reset increments separately and retain all raw counts.

Exercise start/stop, seeks and loops, then listen to instrumental passages,
quiet real vocals and Other. Synthetic-host success alone leaves this installed
AU test outstanding. Preserve the complete shipped v0.6.1 bundle outside the plugin folder for
comparison and rollback.
