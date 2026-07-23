# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

StemgenRT is a real-time music source separation plugin built with JUCE and ONNX Runtime. This tree currently carries an explicitly unqualified c126 listening graph and exposes four stereo stems: drums, bass, other, and vocals.

## Build Commands

```bash
# Initial setup - download ONNX Runtime
./scripts/download-onnxruntime.sh  # macOS
./scripts/download-onnxruntime.ps1  # Windows

# Configure and build the c126 listening candidate (debug)
cmake --preset default
cmake --build --preset default

# Future qualification/release lane (requires the official ORT 1.26.0 SDK;
# a successful build does not promote the current listening-only model)
cmake --preset release
cmake --build --preset release

# Install plugins to the current user's plugin directories (macOS)
# IMPORTANT: Always use this script instead of manual cp.
# cp -R does NOT overwrite existing .component/.vst3 bundles reliably.
./scripts/install-plugins.sh --debug    # current c126 listening build
./scripts/install-plugins.sh            # future Release lane (default)
./scripts/install-plugins.sh --release  # future Release lane (explicit)
```

## Architecture

### Audio Processing Pipeline (PluginProcessor.cpp)

The plugin uses a dual-threaded architecture:
- **Audio thread**: Preserves native host-rate input for Main/fallback, converts qualified higher rates onto the model clock, collects 512-sample model requests, and returns completed host-rate ranges through bounded ring buffers.
- **Inference thread**: Owns the model's previous-audio and fusion-GRU state, runs ONNX inference asynchronously, and converts retained stems back to the host clock before publishing them.

Do not reintroduce the former external context/reflection padding, HP/LP crossover, LP reinjection, vocals-specific or input soft gates, low-band stabilizer, or chunk-tail crossfade before or around the graph. The graph owns its analysis and overlap-add processing. The only model-input preprocessing is the state-aware, boost-only fullband normalization described below; the final writer also has a residual-preserving low-level separation-confidence fade.

### Stateful ONNX Contract

The graph processes stereo float32 audio at exactly 44.1 kHz. It has a fixed 512-sample hop and the following static interface:

| Direction | Name | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 512]` |
| Input | `past_audio` | `[1, 2, 512]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Output | `separated_chunk` | `[1, 4, 2, 512]` |
| Output | `next_past_audio` | `[1, 2, 512]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |

Initialize both persistent state tensors to zero. `separated_chunk` is aligned to the same call's `audio_chunk`, so sequence zero after a reset is valid. There is no graph pre-roll result, boundary overlap-add state, or zero-hop flush.

Only the inference worker may advance model state. Reset `past_audio`, the raw prior-hop copy, `fusion_hidden`, and the model-input gain on transport starts, seeks, scrubs, loop wraps, inference failures, and input-sequence gaps. Epoch changes during an in-flight run invalidate that output; zero the state before accepting another sequence. The next successful current-sequence result is immediately valid.

### Host Sample-Rate Bridge

The graph clock and hop remain fixed at 44.1 kHz and 512 samples. The c126 listening build accepts exactly a 44.1 kHz / 512-sample prepared host configuration, where both bridge directions are exact zero-delay bypasses. The existing higher-rate bridge is preserved for later requalification but is not enabled by this listening contract.

At higher qualified rates, use continuous stateful linear-phase Kaiser-windowed sinc conversion with a 20 kHz passband and at least 110 dB rejection at 22.05 kHz. Do not resample each callback or model hop independently. The audio thread owns the stereo host-to-model converter; the sequential inference worker owns one six-channel model-to-host converter for drums, bass, and vocals. Advance the output converter for every valid result before publication, even when that result will later be partially or fully late. Both converters use exact integer rational clocks rather than accumulated floating-point ratios, and reset to the absolute input/model origin on every streaming discontinuity.

Keep Main and the dry fallback at the native host rate. Never round-trip them through 44.1 kHz. Convert only drums, bass, and vocals, then calculate Other from native delayed Main so ultrasonic host content remains in Other and the output stays mixture-lossless. The paired FIR content delay must be an exact integer number of host samples; use the output converter's sub-sample phase correction during preparation and fail closed if the corrected delay is not integral. Include that delay in PDC, preallocate every converter/request/ring buffer, and perform no allocation or locking in steady-state conversion.

### Model-Input Level

The c126 graph remains materially level-sensitive, and a -10 dB copy can separate differently from the same content at ordinary level. On the inference worker, calculate one stereo-linked RMS and peak over the exact raw analysis window: current 512-sample hop after reset, then raw past plus current hops. Apply a boost-only gain equal to the minimum of the gain toward `kModelInputTargetRms` (-12 dBFS RMS), the peak headroom below `kModelInputPeakCeiling` (0 dBFS), and `kModelInputMaxBoost` (+40 dB). Clamp the result to a minimum of one, so input already at or above the RMS target or peak ceiling is not attenuated. Hold the previous gain through exact digital silence.

When gain changes from `Gprev` to `G`, multiply `past_audio` by `G / Gprev`, multiply the raw current hop by `G`, and do not scale `fusion_hidden` because it is nonlinear feature state. Keep a separate raw past hop for the next detector window. Divide the emitted separated tensor by `G`, preserve drums/bass/vocals, and calculate Other from the raw current hop. This keeps every amplitude-domain tensor coherent without changing Main, fallback level, latency, or the graph interface. Non-finite input or gain state must fail closed and reset all streaming state.

The target and maximum boost originate from the former deployment wrapper and listening evidence, not an embedded ONNX training-level contract; the stereo-linked raw-window peak cap is part of the qualified native policy. Changes to the target, maximum boost, peak ceiling, detector window, state migration, or boost-only policy require requalification.

### Latency and Fallback

The c126 listening pipeline has a one-hop minimum latency:

- 512 samples to collect and queue the current request.

At the accepted 44.1 kHz / 512-sample configuration, report exactly 512 samples (11.61 ms) to the host for PDC. The worker receives the following callback interval to complete inference before sequence N is consumed. The preserved callback-reserve formula reports 960, 768, 512, and 1,024 samples for 64-, 256-, 512-, and 1,024-sample prepared callbacks respectively, but only the 512-sample configuration is enabled in this listening build. Keep Main and fallback aligned to the exact reported timeline.

At higher qualified host rates, map model-hop boundaries with exact rational integer arithmetic. Add the paired input/output conversion delay to the rate-aware callback scheduling reserve, and make the paired delay exactly integral with the output FIR phase correction. Timestamp converted ranges on their nominal host clock and schedule them after the model scheduling reserve; native Main is delayed by the complete PDC. Never accumulate clock phase in floating point or hide SRC group delay from the host.

Tag every processed hop with its exact output sample range. If inference is unavailable or late, discard its elapsed prefix and use the complete latency-aligned dry fallback for those samples; never shift a result to the current read position or fade the first missing sample toward zero. Fade model output back in only when an exact-timeline sample is available. The final output stage must still route the residual to `Other`, so the four enabled internal stems sum to Main throughout startup, provider transitions, and underruns. Never let a late result reappear against a newer dry timeline.

If a real-time callback size requires more scheduling latency than the PDC established in `prepareToPlay`, do not change PDC or reset the stream from the audio thread. Record the unsafe callback in lock-free diagnostics and present complete latency-aligned fallback for that callback (zero Drums/Bass/Vocals, Main in Other) while continuing to advance and discard scheduled model samples on their exact timeline. Offline rendering is exempt because it waits for each submitted hop.

### Mixture-Lossless Invariant

The model output order is `[drums, bass, vocals, other]`. After inverse model-input gain, preserve drums, bass, and vocals at every level, then calculate:

```text
other = aligned_mixture - drums - bass - vocals
```

The ONNX graph applies a model-domain correction, and the runtime reapplies it against the raw aligned mixture after inverse gain. The native output stage applies it again after provider-specific numerical differences, fallback crossfades, or the low-level confidence fade. In floating-point processing, the four internal stems must reconstruct the latency-aligned Main mixture. Do not independently normalize, clip, gate, or quantize stems after the final correction. “Mixture-lossless” does not claim recovery of the studio-original sources; integer PCM exports require another residual correction after final quantization.

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

`model/model.onnx` is the only model payload bundled into plugin Resources. It is self-contained; do not require or ship the obsolete `model.onnx.data` file. Its embedded deployment status is `unqualified_listening_only`; do not describe this artifact as promoted or shipping-qualified.

`cmake/QualifiedModelContract.cmake` is the only editable source of qualified model identity and interface values. CMake generates `StemgenRT/QualifiedModelContract.h` from it for runtime and test code, while source-artifact checks and bundle sealing include the same CMake contract directly. A future model replacement starts there and requires requalification; do not duplicate contract literals in runtime, tests, or packaging scripts.

Configuration and runtime loading must fail closed unless the artifact SHA/size, all three input and all three output names, float32 types, static shapes, streaming metadata, checkpoint SHA, and residual-to-source-index-3 policy match the qualified contract.

### CPU Operations

CPU is the reference and supported deployment path. The worker must complete each graph run plus retained-stem output conversion inside the 11.61 ms hop interval; the host-visible PDC does not increase the recurring compute budget. Include audio-thread input conversion in complete-path benchmarks as well. The previous c91 timing figures do not qualify c126; retain underrun telemetry and benchmark this exact graph on every target platform before promotion.

The automatic ORT intra-op policy is platform-qualified: cap macOS sessions at three threads based on the repeated Apple Silicon 1–4 thread sweep, while retaining the previous four-thread cap on Windows until an equivalent Windows sweep is recorded. Explicit thread counts are qualification overrides only and must not silently replace either production default.

Prefer preallocated input/output/state storage and avoid allocations in both the audio callback and steady-state inference loop. The shipping runtime is CPU-only; do not add an execution provider without a separate numerical, packaging, and real-time qualification pass.

## Code Style

- C++20 standard
- Chromium-based clang-format (run `pre-commit install` for auto-formatting)
- Warnings treated as errors

## Testing

Tests are in `test/source/`. Run with `ctest --preset default`.

Changes to the streaming path should cover, at minimum: exact graph names/shapes and model identity; state progression; first-current-hop validity and absence of a flush protocol; reset determinism; sequence-gap recovery; exact reported main/stem alignment across host rates and block sizes; callback-phase latency calculation; prepared/actual block-size mismatch diagnostics and complete-Other fallback; timestamped late-result rejection; bus ordering; finite outputs; boost-only normalization at ordinary level; -10/-20 dB gain invariance; gain changes with coherent past state; raw current-input preservation; exact-zero gain hold; non-finite failure/reset; SRC passband, alias/image rejection, exact bypass, rational frame counts, long-run drift, block-partition invariance, integer paired delay, absolute-phase reset, and ultrasonic residual routing; the low-level confidence endpoints, transition, stereo linking, attack/hold/release timing, reset, and host-block independence; isolation of the dry fallback from confidence; and `Main == Drums + Bass + Other + Vocals` during normal inference, near-silence fading, startup, and underrun fallback. Keep performance benchmarks stateful and include both bridge directions—the old stateless 2,560-sample benchmark is not representative.

## Platform Notes

- **macOS**: Qualified artifacts are Apple Silicon (`arm64`) with a macOS 14.0 deployment target, matching the minimum encoded in the official ONNX Runtime 1.26.0 dylib. That release has no Intel macOS archive. Bundle ONNX Runtime with a canonical `@rpath/libonnxruntime.dylib` install name and rewrite the executable's matching load command before signing. Sign embedded dylibs before signing and strictly verifying the outer AU, VST3, or app bundle after all resource changes. Keep JUCE's optional macOS VST3 auto-manifest disabled: the helper can block in `dyld` before plugin code runs on the packaging host. Windows retains its auto-generated manifest.
- **Windows**: Bundle both the CPU ONNX Runtime import library and DLL from the official SDK. The plugin does not use linker delay-loading; it explicitly loads the sibling `onnxruntime.dll` at runtime to avoid resolving another copy from the host process.
