# Streaming separation model

`model.onnx` is the self-contained stereo HS-TasNet model used by StemgenRT.
It separates **Drums, Bass, Vocals and Other** at **44.1 kHz**, using a 1024-sample
asymmetric analysis window, 256-sample synthesis frame and 128-sample hop.

The graph has **128 samples** of output delay. One asynchronous scheduling hop
makes the plugin's total **256 samples / 5.80 ms** with a 128-sample host buffer.
The graph source order differs from the plugin bus order: Main, Drums, Bass,
Other, Vocals.

## Artifact and state

The model is tracked with Git LFS and needs no external weights file.
SHA-256: `e354d24bfa0f167cea689c3adff4cb479b1c445905a7a31bd894ed4b735c20d1`.
Size: **34,707,454 bytes**. Runtime: **ONNX Runtime 1.26.0 CPU**.

| Input | Shape | Output |
| --- | --- | --- |
| `audio_chunk` | `[1,2,128]` | `separated_chunk`: `[1,4,2,128]` |
| `audio_history` | `[1,2,896]` | `next_audio_history`: same shape |
| `fusion_hidden` | `[2,1,1000]` | `next_fusion_hidden`: same shape |
| `spectral_numerator_tail` | `[1,4,2,128]` | `next_spectral_numerator_tail`: same shape |
| `waveform_tail` | `[1,4,2,128]` | `next_waveform_tail`: same shape |
| `attention_keys` | `[1,31,64]` | `next_attention_keys`: same shape |
| `attention_values` | `[1,31,128]` | `next_attention_values`: same shape |

All public tensors are float32. Initialize six states to zero, carry every returned
state unchanged, and discard the first output after reset. Each subsequent call
returns the preceding input hop. Pad a partial final hop once, submit exactly
one zero hop, then drain the plugin queue. Reset all state after a discontinuity.

The [CMake contract](../cmake/QualifiedModelContract.cmake) owns model identity,
geometry and metadata. Configuration verifies SHA/size; runtime loading checks
all seven inputs, seven outputs and embedded metadata. Its `QUALIFIED` variable
prefix denotes an identity lock, without guaranteeing deadline performance.

The 32-frame attention window includes the current received feature frame and
31 cached key/value frames. These caches add 23,808 bytes of persistent float32
state per stream and no audio buffering. The model also includes magnitude
features, quadrature spectral correction and a nonlinear fused-feature map.

## Output behavior

Input levels and graph states remain in the original amplitude domain. The
runtime preserves all four graph outputs. The final writer applies linked
confidence and recovery fades to Drums/Bass/Vocals, then calculates Other from
Main minus those three outputs. Complete fallback routes Main to Other.

The confidence envelope opens immediately, holds for 50 ms, releases by
60 dB per 100 ms, and smoothly opens between -96 and -72 dBFS peak. It suppresses
unreliable estimates near silence while retaining the full mix.

## Validation

The exact deployment graph scores **4.288488 dB full-band SDR** on the unchanged
development panel: 14 tracks, two 15-second excerpts per track, four stems.
It improves the previous research best by **0.021591 dB** and C204 by
**0.219409 dB**. The source FP32 attention checkpoint scores **4.288099 dB**;
the 0.000389 dB deployment difference is within panel sampling uncertainty.
The total training lineage is 21,250 updates, including 1,000 attention updates.
The saved checkpoint and its 30-tensor Adam audit pass. The 5 dB research target
remains unmet.

The deployment variant quantizes ten large projections to signed 8-bit weights
with unsigned 8-bit dynamic activations. Floating arithmetic feeding those
projections uses double precision; dequantization, audio decoding and public
states remain float32. Phase factors, fusion refinement and attention weights
remain unquantized. Quality was measured on the exact serialized graph before
saving; the saved file's hash matches those evaluated bytes.

| Deployment metric change, dB | Drums | Bass | Vocals | Other |
|---|---:|---:|---:|---:|
| SDR vs previous research best | +0.061 | +0.021 | +0.002 | +0.002 |
| SIR vs previous research best | +0.173 | +0.155 | +0.140 | -0.056 |
| SDR vs C204 | +0.109 | +0.155 | +0.172 | +0.441 |
| SIR vs C204 | +0.373 | +0.085 | +0.155 | +0.961 |

Eight tracks improve and six regress against the research parent. The paired
track-bootstrap SDR interval is -0.0412 to +0.0853 dB; this small gain is from
a selected single-seed trial. Against C204, twelve tracks improve and two
regress: Skelpolu loses 0.661 dB SDR and 3.517 dB SIR; Actions loses 0.009 dB
SDR. Mean drum low-band SDR falls 0.009 dB. These development tracks have
informed repeated model selection and are not an unseen test set.

Absent-source output rises by 0.519 dB for Drums and 0.649 dB for Other against
the research parent; Bass and Vocals fall by 1.559 and 0.482 dB. Lower is better
for this measure. All four improve against C204, by 4.184, 2.360, 0.993 and
4.740 dB respectively in graph source order. Quantization alone raises this
measure by 0.222/0.087/0.109 dB for Drums/Bass/Vocals and lowers Other by
0.058 dB relative to the source FP32 checkpoint.

The [deployment quality report](quality-deployment.json) includes every track,
stem, band and absence comparison against the source checkpoint, research
parent and C204. [Source checkpoint quality](quality-development.json) records
the separate FP32 model results.

Short synthetic/music checks pass with ORT optimizations disabled and enabled.
The 30-second music trajectory plus a partial hop passes twice with exact
reset replay for every output and state. Both hidden and attention states
match the independently reconstructed integer reference exactly; maximum
waveform error is 2.981e-7. Every callback checks all six states, float32 shapes,
finite values and exact input history. One flush recovers every real sample.
The [streaming evidence](streaming-validation.json) retains the unchanged
thresholds, errors and trajectory hashes. The original FP32 export exceeded
its long-stream internal-state tolerance; integer checks compare against the
declared integer reference, and its changed inference has its own quality score.

The [synthetic fixture metadata](../test/fixtures/cropped1024-pytorch.json)
identifies the independent CPU PyTorch integer oracle. Eight clip lengths
cover single samples, partial hops and a 16,521-sample trajectory beyond the
attention window. The fixture generator does not import or execute ONNX Runtime.

The Release native correctness suite passes: 162 tests passed, one Windows-only
test was skipped on Linux and seven tests are disabled by default. It covers
the independent fixture, all-four-stem output, pre-roll/reset/EOF, queue races,
PDC/Main alignment, non-finite input, fallback, variable offline callbacks and
callback allocation checks. The Linux build covers the two native test targets;
macOS AU/VST3 build and bundle validation remain to be checked in CI and on
the target Macs.

## Timing limits

The [Linux evidence](linux-validation.json) records Ryzen 5 5500 / WSL2 results
with one inference worker and no concurrent research training or scoring.
Six balanced fresh-process comparisons, each with 256 warmup and 2,048 measured
hops, give median-of-cycle p50 **2.634 ms** for this graph versus **4.218 ms** for
C204 FP32: **37.6% faster**, with all six pairs faster. However, 1,816 of 12,288
new-model calls exceed the 2.902 ms hop budget. This is a speed comparison,
without a passing deadline qualification.

The separate 500-hop plugin-runtime test fails its p95 deadline assertion
(p50 2.943 ms, p95 4.737 ms, 265 calls over budget). The full plugin soak also
fails: **2,925 of 10,000 measured callbacks miss their output boundary**, plus
13 warmup misses. It records 31 queue drops. The complete callback itself has
p99 23.444 microseconds, no deadline misses and no waits; all output is finite,
the three retained stems are distinct, and reconstruction error is 1.192e-7.
Apple callback/worker priority checks cannot qualify this Linux run. The
[direct log](linux-direct-timing.log) and [paced log](linux-paced-timing.log)
preserve the failures and bounded worker trace.

Actual M4/M4 Pro and DAW timings remain pending. Earlier C204 M4 results,
including occasional missed deadlines, do not qualify this model. Use the
[test instructions](../M4_TESTING.md) to measure the new build. The runtime
retains aligned fallback for late inference results.
