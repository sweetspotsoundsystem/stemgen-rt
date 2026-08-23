# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

StemgenRT is a real-time music source separation plugin built with JUCE and ONNX Runtime. This tree carries the frozen c91 graph in an explicitly unqualified hardened asynchronous 1,024-sample-PDC listening path and exposes four stereo stems: drums, bass, other, and vocals.

## Build Commands

```bash
# Initial setup - download ONNX Runtime
./scripts/download-onnxruntime.sh  # macOS
./scripts/download-onnxruntime.ps1  # Windows

# Configure and build the c91 hardened asynchronous listening candidate (debug)
cmake --preset default
cmake --build --preset default

# Future qualification/release lane (requires the official ORT 1.26.0 SDK;
# a successful build does not promote the current listening-only model)
cmake --preset release
cmake --build --preset release

# Complete target-Mac machine gate; output must be outside this source tree.
# A pass remains unpromoted until the user approves listening.
./scripts/qualify-c91-macos.sh ../c91-mac-qualification-v1

# Seal a dirty candidate for transfer; verify the emitted SHA256SUMS before
# extracting the archive into a fresh target-Mac directory.
./scripts/package-c91-macos-handoff.sh ../c91-mac-handoff-v1

# Install plugins to the current user's plugin directories (macOS)
# IMPORTANT: Always use this script instead of manual cp.
# cp -R does NOT overwrite existing .component/.vst3 bundles reliably.
./scripts/install-plugins.sh --debug    # current c91 listening build
./scripts/install-plugins.sh            # future Release lane (default)
./scripts/install-plugins.sh --release  # future Release lane (explicit)
```

## Architecture

### Audio Processing Pipeline (PluginProcessor.cpp)

The plugin uses a dual-threaded architecture:
- **Audio thread**: Preserves native input for Main/fallback, claims only a result already published for the current callback boundary, submits one complete 512-sample request, and renders without waiting or locking.
- **Inference thread**: Runs at elevated priority, owns previous-audio, overlap-add, and fusion-GRU state, and publishes through the bounded lock-free queue for the following callback.

Do not reintroduce the former external context/reflection padding, HP/LP crossover, LP reinjection, vocals-specific or input soft gates, low-band stabilizer, or chunk-tail crossfade before or around the graph. The graph owns its analysis and synthesis processing. Finite model input is passed through at its exact native floating-point level; the final writer has a residual-preserving low-level separation-confidence fade.

### Stateful ONNX Contract

The graph processes stereo float32 audio at exactly 44.1 kHz. It has a fixed 512-sample hop and the following static interface:

| Direction | Name | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 512]` |
| Input | `past_audio` | `[1, 2, 512]` |
| Input | `overlap_add_buffer` | `[1, 4, 2, 1024]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Output | `separated_chunk` | `[1, 4, 2, 512]` |
| Output | `next_past_audio` | `[1, 2, 512]` |
| Output | `next_overlap_add_buffer` | `[1, 4, 2, 1024]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |

Initialize all persistent state tensors to zero. `separated_chunk` is aligned to the preceding call's `audio_chunk`. Sequence zero after reset is successful pre-roll with `outputValid == false`; sequence one emits sequence zero with `outputValid == true`. One final zero `audio_chunk` advances the graph to the last real hop. Because publication is asynchronous, defer a play-to-stop reset until the following exact 512-sample callback has claimed and rendered that result; shorter stop callbacks remain outside the listening contract.

This is the exact three-state c91 ABI: `past_audio`, `overlap_add_buffer`, and `fusion_hidden`. Do not import current-chunk alignment, adapter gates, or additional history tensors from later graphs. Only the inference worker may advance model state. Reset all three states on transport starts, seeks, scrubs, loop wraps, inference failures, and input-sequence gaps. Epoch changes during an in-flight run invalidate that output; zero the state before accepting another sequence. Every reset creates exactly one successful-but-invalid pre-roll result.

### Host Sample-Rate Bridge

The graph clock and hop remain fixed at 44.1 kHz and 512 samples. The hardened asynchronous listening build accepts exactly a 44.1 kHz / 512-sample prepared host configuration, where both bridge directions are exact zero-delay bypasses. The existing higher-rate bridge is preserved for later requalification but is not enabled by this listening contract.

At higher qualified rates, use continuous stateful linear-phase Kaiser-windowed sinc conversion with a 20 kHz passband and at least 110 dB rejection at 22.05 kHz. Do not resample each callback or model hop independently. The audio thread owns the stereo host-to-model converter; the sequential inference worker owns one six-channel model-to-host converter for drums, bass, and vocals. Advance the output converter for every valid result before publication, even when that result will later be partially or fully late. Both converters use exact integer rational clocks rather than accumulated floating-point ratios, and reset to the absolute input/model origin on every streaming discontinuity.

Keep Main and the dry fallback at the native host rate. Never round-trip them through 44.1 kHz. Convert only drums, bass, and vocals, then calculate Other from native delayed Main so ultrasonic host content remains in Other and the output stays mixture-lossless. The paired FIR content delay must be an exact integer number of host samples; use the output converter's sub-sample phase correction during preparation and fail closed if the corrected delay is not integral. Include that delay in PDC, preallocate every converter/request/ring buffer, and perform no allocation or locking in steady-state conversion.

### Model-Input Level

Pass every finite sample to `audio_chunk` unchanged and retain a raw copy for previous-hop Main/residual alignment. Preserve every returned graph state in that same amplitude domain. Non-finite input must fail closed and reset all streaming state.

Do not restore the former per-hop RMS/peak boost. It was inherited from an older deployment wrapper rather than the training or frozen evaluation contract. On the frozen validation excerpt used for the listening diagnosis, it reduced c126 drum SDR from 4.31 to 3.37 dB and bass SDR from 5.44 to 3.81 dB. Its 1,024-sample detector also varied by 1.90 dB on a steady 30 Hz sine, while `fusion_hidden` could not be amplitude-migrated coherently. Raw input is therefore the bounded audition policy until a level-robust model is trained and requalified.

### Latency and Fallback

The c91 graph has one hop of output delay and the asynchronous queue contributes one additional hop. At callback N, the audio thread submits input N. The worker produces the graph's output for N-1 after that callback and publishes it for callback N+1, so input N-1 is rendered at N+1. The host therefore reports exactly 1,024 samples (23.22 ms) of PDC.

The real-time audio callback never waits for inference. The worker has the recurring 512-sample interval between submission and the following eligible callback boundary, about 11.61 ms at 44.1 kHz, to publish its result. This is a provisional audition budget, not a hard-real-time guarantee. Only an exact 44.1 kHz / 512-sample callback may use model output.

At higher qualified host rates, map model-hop boundaries with exact rational integer arithmetic. Add the paired input/output conversion delay to the rate-aware callback scheduling reserve, and make the paired delay exactly integral with the output FIR phase correction. Timestamp converted ranges on their nominal host clock and schedule them after the model scheduling reserve; native Main is delayed by the complete PDC. Never accumulate clock phase in floating point or hide SRC group delay from the host.

Tag every processed hop with its exact output sample range. If inference is unavailable or misses its following-callback publication boundary, leave the worker and graph state continuous, discard that result when its original range has elapsed, and use the complete latency-aligned dry fallback for those samples. Never shift a late result to the current read position. The final output stage must still route the residual to `Other`, so the four enabled internal stems sum to Main throughout startup and deadline misses.

If a real-time callback size requires more scheduling latency than the PDC established in `prepareToPlay`, do not change PDC or reset the stream from the audio thread. Record the unsafe callback in lock-free diagnostics and present complete latency-aligned fallback for that callback (zero Drums/Bass/Vocals, Main in Other) while continuing to advance and discard scheduled model samples on their exact timeline. Offline rendering is exempt because it waits for each submitted hop.

### Mixture-Lossless Invariant

The model output order is `[drums, bass, vocals, other]`. Preserve drums, bass, and vocals, then calculate:

```text
other = aligned_mixture - drums - bass - vocals
```

The ONNX graph applies a model-domain correction, and the runtime reapplies it against the raw aligned mixture. The native output stage applies it again after provider-specific numerical differences, fallback crossfades, or the low-level confidence fade. In floating-point processing, the four internal stems must reconstruct the latency-aligned Main mixture. Do not independently normalize, clip, gate, or quantize stems after the final correction. “Mixture-lossless” does not claim recovery of the studio-original sources; integer PCM exports require another residual correction after final quantization.

The native writer reproduces those tensor estimates unchanged only when model output is available, the underrun crossfade is complete, and separation confidence is fully open. The graph has an approximately level-independent per-stem floor near silence. `OutputWriter` must suppress that unreliable presentation without changing the graph input or recurrent state: detect the latency-aligned mixture with one stereo-linked peak envelope; open immediately; hold peaks for 50 ms; then release by 60 dB per 100 ms; map the envelope to confidence with a smoothstep over the linear-amplitude interval that is zero at and below -96 dBFS peak and one at and above -72 dBFS peak. If `x` is the underrun crossfade and `g` is the low-level confidence, apply:

```text
drums' = x * g * drums
bass' = x * g * bass
vocals' = x * g * vocals
other' = main - drums' - bass' - vocals'
```

At complete fallback (`x = 0`), route the latency-aligned mixture entirely to Other. The confidence stage must not attenuate Main or the latency-aligned dry fallback. Reset the envelope on the same transport discontinuities as the output crossfade. The initial thresholds and timing are provisional until the native numerical and listening qualification is recorded; after acceptance, threshold, timing, or routing changes require a new qualification pass.

At unqualified host sample rates, fail closed: do not start the separator, report the unsupported rate, and use the safe non-model path.

### Output Bus Layout

5 output buses total: Main (latency-aligned mixture), then 4 stereo stem buses:
Drums (model index 0), Bass (model index 1), Other (model index 3), Vocals (model index 2)

### Model

`model/model.onnx` is the only model payload bundled into plugin Resources. It is the self-contained frozen c91 streaming graph, SHA-256 `52fdc46d015819821dae19ef272b6bc4ccf441a0274d7d9c8bb44af50eafef8c`, size 129,088,022 bytes, from checkpoint SHA-256 `e966c7e98c9fa05ed6eaebfd3ee56bf82a5b7f7ef502b1a8389a50d9da40901d`. Do not require or ship the obsolete `model.onnx.data` file. The model's quality is frozen; the two-hop asynchronous timing path remains listening-only until target-Mac qualification.

`cmake/QualifiedModelContract.cmake` is the only editable source of qualified model identity and interface values. CMake generates `StemgenRT/QualifiedModelContract.h` from it for runtime and test code, while source-artifact checks and bundle sealing include the same CMake contract directly. A future model replacement starts there and requires requalification; do not duplicate contract literals in runtime, tests, or packaging scripts.

Configuration and runtime loading must fail closed unless the artifact SHA/size, all four input and all four output names, float32 types, static shapes, streaming metadata, checkpoint SHA, and residual-to-source-index-3 policy match the qualified contract.

### CPU Operations

CPU is the reference and supported deployment path. A native c114 control on the research Ryzen measured the exact c91 ONNX at 3.84 ms mean, 6.01 ms p99, 9.10 ms p99.9, and 7 misses in 10,000 direct calls. That demonstrates feasibility but does not qualify the plugin. Promotion requires complete worker-wake/run/publish/callback-boundary-claim/write timing under DAW load on each target Mac, with paired control, p99.9 margin, and miss-delta evidence.

The automatic ORT intra-op policy is platform-qualified: cap macOS sessions at three threads based on the repeated Apple Silicon 1–4 thread sweep, while retaining the previous four-thread cap on Windows until an equivalent Windows sweep is recorded. Explicit thread counts are qualification overrides only and must not silently replace either production default.

Prefer preallocated input/output/state storage and avoid allocations in both the audio callback and steady-state inference loop. The shipping runtime is CPU-only; do not add an execution provider without a separate numerical, packaging, and real-time qualification pass.

## Code Style

- C++20 standard
- Chromium-based clang-format (run `pre-commit install` for auto-formatting)
- Warnings treated as errors

## Testing

Tests are in `test/source/`. Run with `ctest --preset default`.

Changes to the streaming path should cover, at minimum: exact graph names/shapes and model identity; OLA/past/hidden state progression; successful `outputValid == false` pre-roll and one-zero-hop flush; reset determinism; sequence-gap recovery; exact 1,024-sample PDC; nonblocking callback-boundary claim, epoch invalidation, and late-result discard; prepared/actual block-size mismatch and complete-Other fallback; bus ordering; finite outputs; raw-level previous-hop alignment; non-finite failure/reset; explicit low-frequency seam measurements; low-level confidence behavior; and `Main == Drums + Bass + Other + Vocals` during normal inference, startup, deadline misses, and fallback. Keep performance benchmarks stateful and measure the complete worker-wake/run/publish/callback-boundary-claim/write path.

## Platform Notes

- **macOS**: Qualified artifacts are Apple Silicon (`arm64`) with a macOS 14.0 deployment target, matching the minimum encoded in the official ONNX Runtime 1.26.0 dylib. That release has no Intel macOS archive. Bundle ONNX Runtime with a canonical `@rpath/libonnxruntime.dylib` install name and rewrite the executable's matching load command before signing. Sign embedded dylibs before signing and strictly verifying the outer AU, VST3, or app bundle after all resource changes. Keep JUCE's optional macOS VST3 auto-manifest disabled: the helper can block in `dyld` before plugin code runs on the packaging host. Windows retains its auto-generated manifest.
- **Windows**: Bundle both the CPU ONNX Runtime import library and DLL from the official SDK. The plugin does not use linker delay-loading; it explicitly loads the sibling `onnxruntime.dll` at runtime to avoid resolving another copy from the host process.
