# Test StemgenRT 0.7.0 on M4

Use native arm64 macOS 14+, 44.1 kHz and the released independent-model graph
(SHA-256 starts `0a49388dd2a4`). It has 13 states and a 128-sample hop. With a
128-sample host buffer the graph-plus-worker latency is 256 samples. Previous
model measurements do not establish this model's real-time performance.

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
