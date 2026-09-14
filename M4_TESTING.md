# Model tests on M4

This development checkout tests the fourteen-projection graph (SHA-256 prefix
`878c74694fa4`). It retains one inference worker, 44.1 kHz, 128-sample model hops
and 256 samples of graph-plus-host delay with a 128-sample host buffer.
The Linux Release correctness suite passed 162 tests. Complete quality review
measures 4.455150 dB full-band SDR versus v0.4.0's 4.455188 dB. Instrumental
vocal leakage is essentially unchanged; Skelpolu Other SIR loses 0.196152 dB.
Normal hosted macOS and Windows correctness runs passed. The local Mac run
passed 160 tests, failed both independent PyTorch parity checks and skipped
one Windows-only test. The user reports that the installed PR #15 candidate
works; formal zero-fallback M4 acceptance remains pending. A separate hosted
Apple M1 Virtual paced run failed with substantial scheduling irregularity.
Version v0.4.1-rc.1 is a speed and playback testing prerelease.
See [the model report](model/README.md) for current evidence.

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
This change only affects the test harness and runner. No physical M4 execution
of the extended trace has been verified yet.

In the DAW, record sample rate, buffer, workload, duration and cumulative
fallback counters. Separate startup/reset increments from steady playback;
require zero new steady-playback fallback. Include start/stop, seeks and loops,
instrumental passages, quiet real vocals, and Other-stem listening. Retain the
complete v0.4.0 bundle outside the plugin directory for rollback.
