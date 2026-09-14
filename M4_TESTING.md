# Model tests on M4

This development checkout tests the fourteen-projection graph (SHA-256 prefix
`878c74694fa4`). It retains one inference worker, 44.1 kHz, 128-sample model hops
and 256 samples of graph-plus-host delay with a 128-sample host buffer.
The Linux Release correctness suite passed 162 tests. Complete quality review
measures 4.455150 dB full-band SDR versus v0.4.0's 4.455188 dB. Instrumental
vocal leakage is essentially unchanged; Skelpolu Other SIR loses 0.196152 dB.
Native Mac correctness and target-Mac playback are pending. This is a speed
test candidate, with no quality or release selection.
See [the model report](model/README.md) for current evidence.

After authenticating this checkout and building its native arm64 Release
binary with official ONNX Runtime 1.26.0, run the existing correctness and
paced qualification, then run `scripts/extended-soak-macos.sh` with the source
directory and a new evidence directory. The extended runner repeats 30-minute
paced tests and retains machine/model/source identities, raw logs and exits.
It does not qualify installed-DAW playback by itself.

In the DAW, record sample rate, buffer, workload, duration and cumulative
fallback counters. Separate startup/reset increments from steady playback;
require zero new steady-playback fallback. Include start/stop, seeks and loops,
instrumental passages, quiet real vocals, and Other-stem listening. Retain the
complete v0.4.0 bundle outside the plugin directory for rollback.
