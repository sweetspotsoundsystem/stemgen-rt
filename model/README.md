# Streaming separation model

`model.onnx` is the self-contained stereo HS-TasNet graph used by StemgenRT.
It separates **Drums, Bass, Vocals and Other** at **44.1 kHz**, using a 1024-sample
asymmetric analysis window, 256-sample synthesis frame and 128-sample hop.
The graph delay is **128 samples**; one asynchronous scheduling hop makes the
plugin delay **256 samples / 5.80 ms** with a 128-sample host buffer.

## Artifact and state

The model is tracked with Git LFS and needs no external weights file.
SHA-256: `878c74694fa4c558de1c5a75837893a0afeadcf57f6e3b860d5904cab04e9fc9`.
Size: **39,789,914 bytes**. Runtime: **ONNX Runtime 1.26.0 CPU**.

| Input | Shape | Output |
| --- | --- | --- |
| `audio_chunk` | `[1,2,128]` | `separated_chunk`: `[1,4,2,128]` |
| `audio_history` | `[1,2,896]` | `next_audio_history`: same shape |
| `fusion_hidden` | `[2,1,1000]` | `next_fusion_hidden`: same shape |
| `spectral_numerator_tail` | `[1,4,2,128]` | `next_spectral_numerator_tail`: same shape |
| `waveform_tail` | `[1,4,2,128]` | `next_waveform_tail`: same shape |
| `attention_keys` | `[1,31,64]` | `next_attention_keys`: same shape |
| `attention_values` | `[1,31,128]` | `next_attention_values`: same shape |
| `spec_memory_hidden` | `[1,1,500]` | `next_spec_memory_hidden`: same shape |
| `waveform_memory_hidden` | `[1,1,500]` | `next_waveform_memory_hidden`: same shape |

All public tensors are float32. Initialize all eight states to zero and carry
every returned state unchanged. Fusion and branch GRU states use the public
scale 2^-18. Discard the first output after reset; each subsequent call returns
the preceding input hop. Pad a partial final hop once, submit exactly one zero
hop, then drain the plugin queue. Reset every state after a discontinuity.

The [CMake contract](../cmake/QualifiedModelContract.cmake) owns identity,
geometry and metadata. Configuration verifies the graph hash and size; loading
checks nine inputs, nine outputs and 88 metadata entries. Its `QUALIFIED`
variable prefix denotes this identity lock, without a timing guarantee.

The attention cache and two branch memories use only received features. The
new branch states add 4,000 bytes per stream and no audio lookahead or queue.
The graph also includes magnitude features, quadrature spectral correction and
a nonlinear fused-feature map.

## Output behavior

The graph receives raw stereo input levels and returns all four stems. Plugin
buses are Main, Drums, Bass, Other, Vocals; graph order is Drums, Bass, Vocals,
Other. The writer applies the existing linked confidence and recovery fades to
Drums/Bass/Vocals, then computes `Other = Main - Drums - Bass - Vocals`.
Complete fallback routes Main to Other.

The confidence envelope opens immediately, holds for 50 ms, releases by
60 dB per 100 ms, and smoothly opens between -96 and -72 dBFS peak.

## Quality and current scope

This development graph adds signed integer weights and dynamic unsigned
activations to four branch-memory GRU matrix products. Fourteen projections
now use integer products; GRU biases, nonlinearities and output projections
retain their previous floating arithmetic. The source EMA checkpoint still
scores 4.465157 dB full-band SDR. That is not this changed graph's score.

The exact graph scores **4.455150 dB full-band SDR** on the unchanged 14-track /
28-excerpt panel, versus **4.455188 dB** for v0.4.0. The difference is
-0.000038 dB, with a paired-track 95% interval of -0.000244 to +0.000132 dB.
This is a runtime cost experiment; the 5 dB target remains unmet.

| Full-band SDR, dB | Drums | Bass | Vocals | Other |
| --- | ---: | ---: | ---: | ---: |
| Candidate | 4.403024 | 4.995858 | 5.307915 | 3.113803 |
| Change from v0.4.0 | +0.000070 | -0.000126 | +0.000046 | -0.000143 |

Twenty-five of 56 track/stem SDR cells regress; the largest loss is
0.003876 dB on Triviul Other. SIR shows a larger local change: Skelpolu Other
loses **0.196152 dB**, from -5.469816 to -5.665968 dB. Eight of 23 eligible
natural-absence cells have higher unwanted output. All cells and raw source-view
windows are retained in [deployment quality](quality-deployment.json).
Embedded graph metadata retains its export-time quality status; completed
deployment measurements are recorded in that report against the exact graph hash.

Instrumental vocal output averages -47.257578 dBFS, or -26.661028 dB relative
to the mix. Its 0.001991 dB decrease from v0.4.0 leaves the reported leakage
problem essentially unchanged. The same 17/420 active instrumental windows
remain within 10 dB of the mix; Rockshow at 80–81 seconds remains only
0.590990 dB below it. Isolated-vocal SDR is 22.502774 dB, signed gain 0.922522,
and instrumental Other SDR is 5.484724 dB. Version v0.4.1-rc.1 is for M4 speed
and playback testing. User listening and representative instrumental material
remain necessary alongside these development-panel measurements.

## Numerical and native checks

The saved graph passed ten short cases and 8,216 longer graph calls per
implementation, with exact reset replay and maximum waveform error 8.381903e-8
against its independent fourteen-projection reference. Existing tolerances
remain fixed. The portable fixture covers eight clip lengths, including
partial EOF, with expectations derived in PyTorch and runtime imports blocked.
See [streaming evidence](streaming-validation.json).

The native Release suite passed 162 tests, with one Windows-only test skipped
and seven disabled by default. All-four-stem parity, partial EOF, queue/reset
races, Main/PDC alignment, fallback reconstruction and callback allocation
checks passed. [Linux evidence](linux-validation.json) retains actual exits
and source/model/binary identities. The separate local preallocated inference
comparison measured about 40% lower median block p50
than v0.4.0 under concurrent load; every measured call still exceeded the local
hop budget. That x86 comparison establishes no M4 or complete-plugin timing.

Normal hosted macOS and Windows correctness builds of PR #15 passed. The
local Mac Release run of `95a5e6c` passed 160 tests, failed the two independent
PyTorch waveform parity tests and skipped the Windows-only case. Maximum
absolute stem error was 0.00123772398 against the unchanged 0.00001 tolerance.
Both direct model inference and offline plugin rendering exhibited the
mismatch; its cause remains unestablished. Model and fixture hashes matched
the declared identities. A successful playback report does not clear these
numerical failures.

## M4 playback

The user reported 1,408 fallback samples after a few minutes on M4 with v0.4.0.
The user subsequently reported that the installed PR #15 candidate works.
This is a user-reported playback result without a recorded soak duration or
counter trace. The target remains zero additional fallback during repeated
30-minute steady-playback tests and the installed DAW workload, with
startup/reset counters retained separately. Follow [M4 testing](../M4_TESTING.md).

The separate [hosted qualification run](https://github.com/sweetspotsoundsystem/stemgen-rt/actions/runs/34795259568)
on Apple M1 Virtual failed: the paced run recorded 1,074,816 measured fallback
samples, 6,956 callback-start deadline misses and 7,247 catch-up intervals.
Those scheduling irregularities do not establish physical M4 behavior or
isolate the cause of fallback. Formal target-Mac qualification remains open.
The original v0.4.0, earlier attention model and C204 remain rollback baselines.
