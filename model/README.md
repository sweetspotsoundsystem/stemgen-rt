# Streaming separation model

`model.onnx` separates Drums, Bass, Vocals and Other at 44.1 kHz. It accepts
128-sample stereo hops, uses a 1024-sample asymmetric analysis window and a
256-sample synthesis frame, and carries eight FP32 states. The graph delay
is 128 samples; the asynchronous host queue adds 128 samples at the supported
128-sample host buffer. Total graph-plus-host delay is 256 samples.

The authoritative identity and interface are in
`cmake/QualifiedModelContract.cmake`. This self-contained graph has SHA-256
`08424ca91feae8d4746442a35ebf70489dea70ea6e81401b39483cf02d497748` and is 37,532,574 bytes.
It retains the source checkpoint, 39,250 training updates, raw input levels,
four outputs, residual policy and streaming/reset/EOF contract. The confidence
envelope and output routing are unchanged.

## Inference changes

The three attention input products are packed into one dynamic U8/S8 product
with per-column signed weight scales. The query's last-frame slice follows
the packed projection, preserving its time selection. Seventeen products now
use reduced signed weights in [-64,64]. The other graph nodes and initializers
retain their definitions. No extra audio buffering or persistent state is added.

The plugin also sets `mlas.disable_kleidiai=1`. The retained
[M4 Pro parent diagnostic](macos-runtime-parity-diagnostic.json) failed all
eight independent reference cases with ORT defaults and passed all eight with
KleidiAI disabled. The setting and graph optimization are separate commits;
the setting-only parent is `af06fac`. The complete production candidate now
passes both independent PyTorch parity tests on the physical M4 Pro, as
recorded in [the local validation report](macos-validation.json). Sustained
timing measurements remain outstanding.

## Deployment quality

The exact graph scores **4.455172594 dB full-band SDR**, a **+0.000019666 dB**
change from PR #17's sixteen-product graph on the unchanged 14-track,
28-excerpt development panel. The source FP32 checkpoint scores 4.465157422 dB;
v0.4.0's deployment graph scores 4.455188055 dB. These are separate endpoints.
The 5.0 dB target remains unmet.

| Stem | PR #17 SDR (dB) | Candidate SDR (dB) | Change (dB) |
| --- | ---: | ---: | ---: |
| Drums | 4.403038 | 4.403057 | +0.000019 |
| Bass | 4.995898 | 4.995904 | +0.000006 |
| Vocals | 5.307888 | 5.307862 | -0.000026 |
| Other | 3.113788 | 3.113867 | +0.000080 |

Full-band SDR decreases in 29/56 track/stem cells;
the worst change is -0.000927550 dB. The complete
[deployment report](quality-deployment.json) retains all track/stem/band/absence
regressions, paired bootstrap summaries, and all 840 source-view windows.
The worst SIR change is -0.049846 dB for other
on Skelpolu - Human Mistakes. The largest natural-absence output increase is
+0.005724 dB for vocals on
Young Griffo - Pennies.

On the exact instrumental remixes, mean unwanted vocal output is
-47.254567 dBFS, a +0.003535 dB change from PR #17
(positive means more leakage). The largest one-second instrumental-window
increase is +0.030907 dB. On isolated vocals, desired-vocal
SDR changes by +0.001391 dB and signed desired projection gain
changes by +0.000019911. Read these with Other quality and the retained
worst windows. They do not establish improved instrumental listening.
The small instrumental-vocal increase occurs on all fourteen tracks; this
runtime experiment does not solve the reported vocal-leakage problem.

Scoring uses the original continuous input, physical intervals, alignment,
residual reconstruction and metric code. CPU ORT 1.26.0 runtime binaries match
those used for the parent measurements. There is no new confirmation panel;
source-view references can contain recording bleed. Listening acceptance and
representative real instrumental material remain outstanding.

## Numerical and native checks

[Streaming validation](streaming-validation.json) retains independent PyTorch
reconstruction of all seventeen integer products, short and long carried-state
cases, nonzero initial states, partial EOF and exact reset replay. Expected
fixture outputs are generated without importing ONNX Runtime.

The [Linux native suite](linux-validation.json) passed 164 tests, with one
platform-specific skip and seven disabled performance tests. Both four-stem
parity tests passed at the unchanged 1e-5 waveform limit. The suite also checks
state/reset/EOF behavior, timestamp admission, queue recovery, alignment,
reconstruction, variable offline callbacks and callback heap traffic. Its
record includes 50 plugin/test/contract/fixture input hashes.

All 24 [Linux backend diagnostic cases](runtime-parity-diagnostic.json) passed
with maximum error 1.63912773132e-7. The diagnostic retains ORT-default,
KleidiAI-disabled and optimization-disabled sessions. The plugin uses the
KleidiAI-disabled setting. Linux correctness does not establish physical M4
correctness for this new graph by itself.

The [physical M4 Pro Release suite](macos-validation.json) passed **164 tests**,
with one platform-specific skip and seven disabled tests, at PR #19 commit
`35b533017b32099f417b6c965e37214b72a8ccea`. Both independent PyTorch parity tests
passed at the unchanged `1e-5` waveform limit with KleidiAI disabled in
production. AU and VST3 bundles passed strict signature checks and matched the
built candidate byte for byte. The user subsequently reported zero fallback
on M4 with the installed PR #19 plugin; playback duration and a raw
fallback-counter trace were not supplied.
The 0.5.0 release preparation changes version metadata and documentation.

## Timing and M4 acceptance

An eight-block preallocated native ORT 1.26.0 comparison on an AMD Ryzen 5 5500
under WSL2 measured median block p50 of 3.154038 ms for PR #17 and 2.991973 ms
for this graph: **5.14% lower**, with all four paired medians faster. Each block
used 256 warmup and 2,048 measured hops, one ORT thread and disabled spinning.
Timing tails varied under concurrent training. This is a relative local
graph measurement; the two legacy `linux-*-timing.log` files are historical.

The parent M4 Pro diagnostic's longest short clip averaged 0.932 ms with ORT
defaults and 1.047 ms with KleidiAI disabled. Those averages and the Linux graph
comparison cannot establish the combined candidate's M4 performance.

The earlier user report recorded 1,920 fallback samples after ten minutes
with PR #17 in Ableton on M4 at 44.1 kHz / 128 samples. The latest PR #19
playback report explicitly reports zero fallback on M4. This is user-reported
playback evidence; the run duration and raw counter trace are not recorded. Follow
[the M4 test instructions](../M4_TESTING.md) for reproducible numerical checks,
repeated 30-minute untraced soaks,
complete worker traces when needed, and installed-AU playback. Retain raw
startup/reset counters and require zero additional steady-playback fallback.
Exercise transport changes and listen to instrumental passages, quiet real
vocals and Other. Version 0.5.0 ships this graph and runtime setting. Sustained
zero-fallback M4 timing and the broader quality goal remain unqualified.

## Host sample-rate conversion

The resampling branch admits 48, 88.2, 96, 176.4 and 192 kHz hosts through the
existing streaming converters. The model, weights and 44.1 kHz inference clock
are unchanged. Reported host latency includes converter group delay and the
reserve for host/model clock alignment. Native Main remains a delayed copy of
the input; Other reconstructs its residual after Drums, Bass and Vocals.

The model scores and previous M4 playback report above do not measure this
converted path. Sustained CPU/deadline behavior and listening at these rates
require the [resampling checks](../RESAMPLING_TESTING.md) on the target machine.
