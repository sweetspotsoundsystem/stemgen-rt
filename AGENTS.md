# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

StemgenRT is a real-time music source separation plugin built with JUCE and ONNX Runtime. This detached worktree is an unpromoted audition integration for terminal-selected c236: a true 256-sample current-chunk graph with a 1,024-sample past-plus-current analysis window, three explicit state tensors, one asynchronous queue hop, and 256 samples of PDC. The authenticated checked ONNX and receipt-bound contract are installed. Do not call it production-ready or promote it over the qualified 512-hop branch before listening approval and target-Mac qualification.

## Build Commands

```bash
# Initial setup - download ONNX Runtime
./scripts/download-onnxruntime.sh  # macOS
./scripts/download-onnxruntime.ps1  # Windows

# After the terminal c236 artifact and receipt substitutions are complete,
# configure and build the current-chunk candidate (debug)
cmake --preset default
cmake --build --preset default

# Qualification/release lane (requires the official ORT 1.26.0 SDK;
# a successful build alone does not complete target-Mac qualification)
cmake --preset release
cmake --build --preset release

# Install plugins to the current user's plugin directories (macOS)
# IMPORTANT: Always use this script instead of manual cp.
# cp -R does NOT overwrite existing .component/.vst3 bundles reliably.
./scripts/install-plugins.sh --debug    # current candidate debug build
./scripts/install-plugins.sh            # Release lane (default)
./scripts/install-plugins.sh --release  # Release lane (explicit)
```

## Architecture

### Audio Processing Pipeline (PluginProcessor.cpp)

The plugin uses a dual-threaded architecture:
- **Audio thread**: Preserves native input for Main/fallback, submits one complete 256-sample request, and renders completed results one asynchronous queue hop later under the fixed 256-sample PDC. It never waits for inference in the real-time callback.
- **Inference thread**: Runs at elevated priority, owns the three persistent graph states, and publishes current-chunk results through the bounded lock-free queue.

Do not reintroduce the former external context/reflection padding, HP/LP crossover, LP reinjection, vocals-specific or input soft gates, low-band stabilizer, or chunk-tail crossfade before or around the graph. The graph owns its analysis and synthesis processing. Finite model input is passed through at its exact native floating-point level; the final writer has a residual-preserving low-level separation-confidence fade.

### Stateful ONNX Contract

The graph processes stereo float32 audio at exactly 44.1 kHz. It has a fixed 256-sample hop and the following static interface:

| Direction | Name | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 256]` |
| Input | `analysis_history` | `[1, 2, 768]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Input | `emitted_db_history` | `[1, 4, 2048]` |
| Output | `separated_chunk` | `[1, 4, 2, 256]` |
| Output | `next_analysis_history` | `[1, 2, 768]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |
| Output | `next_emitted_db_history` | `[1, 4, 2048]` |

Initialize the three persistent state tensors—`analysis_history`, `fusion_hidden`, and `emitted_db_history`—to positive float32 zero. `separated_chunk` is aligned to the current call's `audio_chunk`. Sequence zero after reset is valid current-chunk output. There is no invalid pre-roll and no end-of-stream flush.

Only the inference worker may advance model state. Host reset notifications and transport starts, seeks, scrubs, loop wraps, inference failures, and input-sequence gaps reset all three state families. A play-to-stop transition observed in the callback first renders the final current-chunk result already owed by the one-hop plugin PDC, then resets after that callback; this drains the queue timeline and is not a graph flush. Epoch changes during an in-flight run invalidate that output; zero the state before accepting another sequence. The first successful result after every reset is current-aligned and valid. Request publication and audio-thread epoch reset must remain atomic-only; the worker owns the bounded idle polling backoff. The host transport snapshot query remains part of the callback and must be covered by AU/VST target-host timing evidence.

### Host Sample-Rate Bridge

The graph clock and hop remain fixed at 44.1 kHz and 256 samples. The checked-export current-chunk candidate accepts exactly a 44.1 kHz / 256-sample prepared host configuration, where both bridge directions are exact zero-delay bypasses. The existing higher-rate bridge is preserved for later requalification but is not enabled by this candidate contract.

At higher qualified rates, use continuous stateful linear-phase Kaiser-windowed sinc conversion with a 20 kHz passband and at least 110 dB rejection at 22.05 kHz. Do not resample each callback or model hop independently. The audio thread owns the stereo host-to-model converter; the sequential inference worker owns one six-channel model-to-host converter for drums, bass, and vocals. Advance the output converter for every valid result before publication, even when that result will later be partially or fully late. Both converters use exact integer rational clocks rather than accumulated floating-point ratios, and reset to the absolute input/model origin on every streaming discontinuity.

Keep Main and the dry fallback at the native host rate. Never round-trip them through 44.1 kHz. Convert only drums, bass, and vocals, then calculate Other from native delayed Main so ultrasonic host content remains in Other and the output stays mixture-lossless. The paired FIR content delay must be an exact integer number of host samples; use the output converter's sub-sample phase correction during preparation and fail closed if the corrected delay is not integral. Include that delay in PDC, preallocate every converter/request/ring buffer, and perform no allocation or locking in steady-state conversion.

### Model-Input Level

Pass every finite sample to `audio_chunk` unchanged and retain a raw copy for current-hop Main/residual alignment. Preserve every returned graph state in that same amplitude domain. Non-finite input must fail closed and reset all streaming state.

Do not restore the former per-hop RMS/peak boost. It was inherited from an older deployment wrapper rather than the training or frozen evaluation contract. On the frozen validation excerpt used for the listening diagnosis, it reduced c126 drum SDR from 4.31 to 3.37 dB and bass SDR from 5.44 to 3.81 dB. Its 1,024-sample detector also varied by 1.90 dB on a steady 30 Hz sine, while `fusion_hidden` could not be amplitude-migrated coherently. Raw input is therefore the bounded deployment policy until a level-robust model is trained and requalified.

### Latency and Fallback

The c236 graph has zero graph-output delay: a successful call for input N returns the separated current chunk N. The asynchronous worker/queue contributes one scheduling hop, so callback N's result is rendered on callback N+1 and the host reports exactly 256 samples (5.80 ms) of PDC.

There is no same-callback wait, audition deadline, pre-roll, or flush in this path. Only an exact 44.1 kHz / 256-sample callback may use model output.

At higher qualified host rates, map model-hop boundaries with exact rational integer arithmetic. Add the paired input/output conversion delay to the rate-aware callback scheduling reserve, and make the paired delay exactly integral with the output FIR phase correction. Timestamp converted ranges on their nominal host clock and schedule them after the model scheduling reserve; native Main is delayed by the complete PDC. Never accumulate clock phase in floating point or hide SRC group delay from the host.

Tag every processed hop with its exact output sample range. If inference is unavailable or a result is not published before its one-hop output range, leave the worker and graph state continuous, discard that result when its original range has elapsed, and use the complete latency-aligned dry fallback for those samples. Never shift a late result to the current read position. The final output stage must still route the residual to `Other`, so the four enabled internal stems sum to Main throughout startup and late-result fallback.

If a real-time callback size requires more scheduling latency than the PDC established in `prepareToPlay`, do not change PDC or reset the stream from the audio thread. Record the unsafe callback in lock-free diagnostics and present complete latency-aligned fallback for that callback (zero Drums/Bass/Vocals, Main in Other) while continuing to advance and discard scheduled model samples on their exact timeline. Offline rendering is exempt because it waits for each submitted hop.

### Mixture-Lossless Invariant

The model output order is `[drums, bass, vocals, other]`. Preserve drums, bass, and vocals, then calculate:

```text
other = current_input_mixture - drums - bass - vocals
```

The ONNX graph applies that correction against the current `audio_chunk`, and the runtime reapplies it against the raw mixture carried with the same request. The native output stage applies it again after provider-specific numerical differences, fallback crossfades, or the low-level confidence fade. In floating-point processing, the four internal stems must reconstruct the latency-aligned Main mixture. Do not independently normalize, clip, gate, or quantize stems after the final correction. “Mixture-lossless” does not claim recovery of the studio-original sources; integer PCM exports require another residual correction after final quantization.

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

`model/model.onnx` is the only model payload bundled into plugin Resources. It is the checked c236 graph (`6dd58f05ee6bdf4beadb5df24849320e4f3dec4dbd0c7767fedd0dde2b557a02`, 111,374,870 bytes); the final contract and `model/C236_AUDITION_SUBSTITUTION.json` bind its terminal, recovery, qualification, and export evidence. Do not require or ship `model.onnx.data`.

The draft contract ID is `c236-terminal-selected-hop256-audition-v1`. It validates the exact c236 exporter metadata, including terminal selection/materialization lineage, recovery/final-chain receipt identities, the bound post-training qualification result, and the unchanged exact-c191 correction-head identity. The external checked-export receipt is schema 3 kind `hs_tasnet_c236_recovery_checked_onnx_export_v3`; its independently verified SHA is contract evidence but must not be ONNX metadata because the receipt already binds the ONNX hash. Listening approval and the target-Mac 10,000-callback paced test remain before promotion.

`cmake/QualifiedModelContract.cmake` is the only editable source of qualified model identity and interface values. CMake generates `StemgenRT/QualifiedModelContract.h` from it for runtime and test code, while source-artifact checks and bundle sealing include the same CMake contract directly. A future model replacement starts there and requires requalification; do not duplicate contract literals in runtime, tests, or packaging scripts.

Configuration and runtime loading must fail closed unless the artifact SHA/size, all four input and all four output names, float32 types, static shapes, three-state streaming metadata, c236 terminal/materialization provenance, exact-c191 head identity, deployment policy, and residual-to-source-index-3 policy match the contract.

### CPU Operations

CPU is the reference and intended deployment path. Promotion requires an exact-model target-Mac paced test of 10,000 measured 256-sample callbacks within the 5.80 ms deadline, with zero deadline misses, underruns, queue/ring drops, unsafe callbacks, non-finite outputs, or reconstruction failures.

The existing automatic ORT intra-op implementation caps macOS sessions at two threads and retains the previous four-thread cap on Windows. Those defaults are unchanged by this ABI draft. Explicit thread counts remain diagnostic overrides and must not silently replace production defaults.

Prefer preallocated input/output/state storage and avoid allocations in both the audio callback and steady-state inference loop. The shipping runtime is CPU-only; do not add an execution provider without a separate numerical, packaging, and real-time qualification pass.

## Code Style

- C++20 standard
- Chromium-based clang-format (run `pre-commit install` for auto-formatting)
- Warnings treated as errors

## Testing

Tests are in `test/source/`. Run with `ctest --preset default`.

Changes to the streaming path should cover, at minimum: exact four-input/four-output graph names, shapes, and model identity; progression and reset of all three persistent states; valid current-aligned callback zero with no pre-roll or flush; reset determinism; sequence-gap recovery; exact 256-sample PDC from graph delay zero plus one asynchronous queue hop; publication success, epoch invalidation, and late-result discard without a real-time wait; prepared/actual block-size mismatch and complete-Other fallback; bus ordering; finite outputs; raw-level current-input alignment; non-finite failure/reset; explicit low-frequency seam measurements; low-level confidence behavior; and `Main == Drums + Bass + Other + Vocals` during normal inference, startup, late results, and fallback. Keep performance benchmarks stateful and measure the complete worker-run/publication/drain/write path.

## Platform Notes

- **macOS**: Qualified artifacts are Apple Silicon (`arm64`) with a macOS 14.0 deployment target, matching the minimum encoded in the official ONNX Runtime 1.26.0 dylib. That release has no Intel macOS archive. Bundle ONNX Runtime with a canonical `@rpath/libonnxruntime.dylib` install name and rewrite the executable's matching load command before signing. Sign embedded dylibs before signing and strictly verifying the outer AU, VST3, or app bundle after all resource changes. Keep JUCE's optional macOS VST3 auto-manifest disabled: the helper can block in `dyld` before plugin code runs on the packaging host. Windows retains its auto-generated manifest.
- **Windows**: Bundle both the CPU ONNX Runtime import library and DLL from the official SDK. The plugin does not use linker delay-loading; it explicitly loads the sibling `onnxruntime.dll` at runtime to avoid resolving another copy from the host process.
