# Model tests on M4

StemgenRT 0.5.0 fuses the attention query, key and value products into one
signed integer projection. The graph SHA-256 begins `08424ca91fea`; it has
seventeen integer products and uses `mlas.disable_kleidiai=1` in the plugin.
It retains one inference worker, eight persistent states, 44.1 kHz,
128-sample model hops and 256 samples of graph-plus-host delay with a
128-sample host buffer.

The graph passed independent short and long numerical checks, 164 Linux
native correctness tests, and all 24 Linux backend diagnostic cases. The
[physical M4 Pro suite](model/macos-validation.json) also passed 164 tests,
including both independent PyTorch parity tests at the unchanged `1e-5` limit.
Each native suite has one platform-specific skip and seven disabled tests.
The tested PR #19 source is `35b5330`; the release preparation changes version
metadata and documentation. Read the [model report](model/README.md) for exact
deployment quality, all regressions and the scope of these measurements.
Sustained timing measurements remain outstanding.

The user reports that the installed PR #19 plugin works well, without a
recorded duration or fallback-counter trace. The earlier PR #17 baseline was
**1,920 fallback samples after ten minutes on M4 in Ableton at 44.1 kHz /
128 samples**. The
retained [M4 Pro diagnostic](model/macos-runtime-parity-diagnostic.json) belongs
to that parent graph. Default ORT settings failed all eight cases, while
KleidiAI disabled passed all eight at the unchanged `1e-5` limit. That evidence
motivates the new plugin setting; it does not measure this graph's timing.

## Correctness and identity

Use a clean checkout of this candidate on native arm64 macOS 14 or newer.
From the repository root, fetch its LFS model and run the existing full
qualification script with a new evidence directory outside the source tree:

```bash
git lfs pull
./scripts/qualify-macos.sh "$HOME/Desktop/stemgen-qkv-qualification-001"
bash scripts/diagnose-runtime-parity.sh . "$HOME/Desktop/stemgen-qkv-parity-001"
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
bash scripts/extended-soak-macos.sh . "$HOME/Desktop/stemgen-qkv-soak-001"
```

The runner retains both raw logs, actual exits, machine/model/source identities,
and startup and measured fallback counts. Keep its strict pass/fail result.
For a separate diagnostic of acquisition, inference and publication delays,
use another directory and enable the complete-duration trace:

```bash
bash scripts/extended-soak-macos.sh . "$HOME/Desktop/stemgen-qkv-trace-001" --trace
```

The trace allocates about 60 MB for 30 minutes and rejects requests beyond a
128 MiB bound before starting the worker. It reports omitted timing samples,
matched and missing requests, and bounded failure-event omissions. Worker
statistics cover matched requests; total process CPU time includes other
threads. Tracing adds clock reads and memory traffic, so retain the untraced
runs as the timing measurements.

The graph alone reduced local Linux median block p50 by 5.14%. The parent M4
Pro diagnostic averaged 0.932 ms with defaults and 1.047 ms with KleidiAI
disabled on its longest short clip. These are separate measurements on
different platforms. Measure this candidate with its actual plugin setting;
its net M4 performance has not been established.

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
AU test outstanding. Preserve v0.4.0 and the PR #17 bundle outside the plugin
folder for comparison and rollback.
