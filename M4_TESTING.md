# Model tests on M4

This checkout tests the sixteen-projection graph (SHA-256 prefix `c7ea50ac67bf`).
It retains one inference worker, 44.1 kHz, 128-sample model hops and 256 samples
of graph-plus-host delay with a 128-sample host buffer. The two branch-memory
output projections now use integer products. Short and long independent
numerical checks passed, and the Linux native suite passed 164 tests with one
platform-specific skip and seven performance tests disabled. Full-panel quality
review is complete: 4.455153 dB SDR versus PR #15's 4.455150 dB. Instrumental
vocal leakage remains essentially unchanged.
The local native comparison measured 9.4% lower median block p50 than PR #15
under concurrent training. Physical M4 testing remains pending.
See [the model report](model/README.md) for the evidence and limitations.

The earlier PR #15 candidate produced a user-reported 3,072 cumulative fallback
samples after ten minutes and failed two local independent PyTorch parity
tests. Retain its graph/runtime evidence when comparing this candidate.

To investigate the two independent PyTorch parity failures, run:

```bash
bash scripts/diagnose-runtime-parity.sh . "$HOME/Desktop/stemgen-parity-evidence-001"
```

Use a new evidence directory for each run. This builds a small diagnostic
against the checkout's official ORT 1.26.0 SDK. Before inference it verifies
the model against `cmake/QualifiedModelContract.cmake` and checks the independent
fixture's hash and declared graph identity. The normal test build now verifies
these identities too, and both parity tests reject fixtures from a different
compiled model.

The diagnostic compares the same eight clips and all four stems with default
settings, `mlas.disable_kleidiai=1`, and graph optimizations disabled. It records
runtime path/build/hash, checkout and fixture identities, CPU/SME capabilities,
compiler and execution exits, and per-stem errors. `diagnostic-exit.txt = 0`
means all measurements completed; each `parity_pass` reports the unchanged
`1e-5` accuracy check. Inspect `results.jsonl`, `stderr.log` and `identity.txt`.

If the default session fails and disabling KleidiAI passes, that isolates the
backend choice as the cause of the numerical mismatch. This remains an M4
hypothesis: [ORT's SME dispatch](https://github.com/microsoft/onnxruntime/blob/v1.26.0/onnxruntime/core/mlas/lib/platform.cpp)
can select a [different input quantizer](https://github.com/ARM-software/kleidiai/blob/v1.20.0/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c).
The diagnostic does not change production kernel settings. Its short Run
averages do not qualify sustained plugin timing. Retain the evidence from the
failing checkout when comparing a later graph or runtime.

After authenticating this checkout and building its native arm64 Release
binary with official ONNX Runtime 1.26.0, run the existing correctness and
paced qualification, then run `scripts/extended-soak-macos.sh` with the source
directory and a new evidence directory. The extended runner repeats 30-minute
paced tests and retains machine/model/source identities, raw logs and exits.
It does not qualify installed-DAW playback by itself.

For a separate diagnostic of late worker acquisition, inference or publication,
append `--trace` and use another evidence directory. Worker and callback timing
storage covers the full requested 30 minutes (about 60 MB on 64-bit platforms).
The test rejects traced runs exceeding its 128 MiB collection budget before
starting the worker. It reports omitted samples, matched and missing measured
requests, and allocated collection bytes. Worker statistics describe matched
requests; missing requests can also result from a queue gap. Detailed failure
events retain their existing 64-entry limit and report omissions. Total process
CPU time includes other threads and is not worker CPU time alone.

Tracing changes memory use and adds clock reads. Keep both default untraced
soaks as the timing evidence; traced diagnostics help investigate their failures.
The extended trace only affects the test harness and runner. No physical M4
execution of that trace has been verified yet.

In the DAW, record sample rate, buffer, workload, duration and cumulative
fallback counters. Separate startup/reset increments from steady playback;
require zero new steady-playback fallback. Include start/stop, seeks and loops,
instrumental passages, quiet real vocals, and Other-stem listening. Retain the
complete v0.4.0 bundle outside the plugin directory for rollback.
