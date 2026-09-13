# Streaming separation model

`model.onnx` is the self-contained stereo HS-TasNet graph used by StemgenRT.
It separates **Drums, Bass, Vocals and Other** at **44.1 kHz**, using a 1024-sample
asymmetric analysis window, 256-sample synthesis frame and 128-sample hop.
The graph delay is **128 samples**; one asynchronous scheduling hop makes the
plugin delay **256 samples / 5.80 ms** with a 128-sample host buffer.

## Artifact and state

The model is tracked with Git LFS and needs no external weights file.
SHA-256: `d2945742d27fe23469614aef4f5b79e46fb1a11696ee2c8e6055c494163bcffa`.
Size: **48,754,181 bytes**. Runtime: **ONNX Runtime 1.26.0 CPU**.

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
checks nine inputs, nine outputs and 84 metadata entries. Its `QUALIFIED`
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

## Quality

The exact saved deployment graph scores **4.455188 dB full-band SDR** on
the unchanged development panel: 14 tracks, two 15-second excerpts per track,
four stems. This is **+0.166700 dB** versus the prior PR graph
(4.288488 dB) and **+0.386109 dB** versus C204
(4.069079 dB). Its difference from the source FP32 checkpoint is
**-0.009969 dB**.

| Deployment metric, dB | Drums | Bass | Vocals | Other |
| --- | ---: | ---: | ---: | ---: |
| Full-band SDR | 4.403 | 4.996 | 5.308 | 3.114 |
| SDR change vs prior PR | +0.153 | +0.191 | +0.217 | +0.106 |
| SIR change vs prior PR | +0.624 | +0.603 | +0.529 | +0.377 |
| SDR change vs C204 | +0.262 | +0.346 | +0.389 | +0.547 |
| Absent-source output change vs prior PR (lower is better) | +2.617 | +1.808 | +1.847 | +1.214 |
| Absent-source output change vs C204 (lower is better) | -1.567 | -0.551 | +0.855 | -3.526 |

Against the prior PR, 14 tracks improve and 0 regress in mean SDR.
The paired track-bootstrap interval for the average gain is +0.1323 to
+0.2026 dB. An average gain can still include individual stem or
absence regressions; the linked report includes every cell.
Of the 56 track/stem SDR cells, 9 regress versus the prior PR.

Mean absent-source output rises for drums, bass, vocals, other versus the prior PR. Lower is better for this metric; include instrumental and quiet passages in listening tests.

Tracks with lower mean SDR than C204: Skelpolu - Human Mistakes (-0.547 dB).

The source is the selected EMA checkpoint after a cumulative training lineage
of **39,250 updates**. Its last stage adds 4,000 updates with pitch/tempo
augmentation and parameter averaging. Parameters introduced later in the
lineage have fewer updates. The source checkpoint's FP32 score is **4.465157 dB**;
the **5 dB research target remains unmet**.

The deployment graph quantizes ten large projections to signed 8-bit weights
with unsigned 8-bit dynamic activations. Floating arithmetic feeding those
projections uses double precision; dequantization, audio decoding and public
states use float32. Phase factors, fusion refinement, attention and the two
branch memories remain unquantized. Deployment quality is measured from the
exact saved ONNX bytes, with the existing final residual reconstruction.

These 14 development tracks have informed repeated model selection. They are
not an unseen test set, and track-bootstrap intervals do not include training
seed or selection uncertainty. The [deployment report](quality-deployment.json)
retains every track, stem, band and absence comparison against the source
checkpoint, prior PR graph and C204. [Source quality](quality-development.json)
records the separate FP32 checkpoint and its saved-state audit identities.

## Numerical and native checks

Five synthetic cases pass with ORT optimizations disabled and enabled, each
with exact reset replay. They cover silence, tones, noise, boundary impulses,
nonzero states and partial final hops. A 30-second music clip plus 37 samples
also passes twice: **10,338 graph calls per run**, including one final flush.
All three GRU state tensors and both attention caches match the independently
reconstructed reference exactly; maximum waveform difference is **3.5763e-7**.

The reference independently derives signed weight bytes with NumPy and uses
PyTorch integer products. It never uses ORT outputs as expected audio. Parity
is to this declared integer inference; the source FP32 score is measured
separately. [Streaming evidence](streaming-validation.json) retains the original
thresholds, callback errors and output/state hashes.

The [native fixture](../test/fixtures/cropped1024-pytorch.json) uses the same
independent CPU reference. Eight clip lengths cover one sample, partial hops
and a 16,521-sample trajectory beyond the attention window. Its generator does
not import or execute ONNX Runtime.

The Release native suite passes **162 tests**, with one Windows-only test
skipped on Linux and seven tests disabled by default. Coverage includes all
four outputs, pre-roll/reset/EOF, queue races, PDC/Main alignment, non-finite
input, fallback reconstruction, variable offline callbacks and callback heap
traffic. [Linux evidence](linux-validation.json) records the model, binaries,
source hashes and test results. Training and CPU scoring ran concurrently.

## M4 and M4 Pro testing

The user reported successful M4 Pro testing of PR #13 and approved its release.
Testing the released AU on M4 remains pending. This report contains no inspected
target-Mac timing capture or quiet Linux timing qualification for this revision.
CI builds and tests the macOS and Windows bundles. Earlier attention-model and
C204 timings do not qualify this larger graph.

Use the [M4/M4 Pro instructions](../M4_TESTING.md) to collect evidence and audition
with one inference worker. The runtime retains aligned fallback for late
results. C204 at `6fc2382` and the prior attention candidate at `c848050` remain
available for comparison and rollback.
