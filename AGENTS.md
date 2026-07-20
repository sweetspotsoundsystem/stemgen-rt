# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

StemgenRT is a real-time music source separation plugin built with JUCE and ONNX Runtime. Its native VST3/AU pipeline runs a qualified, stateful HS-TasNet deployment graph and exposes four stereo stems: drums, bass, other, and vocals.

## Build Commands

```bash
# Initial setup - download ONNX Runtime
./scripts/download-onnxruntime.sh  # macOS
./scripts/download-onnxruntime.ps1  # Windows

# Configure and build (debug)
cmake -S . -B build
cmake --build build

# Release build
cmake -S . -B build-release
cmake --build build-release

# Install plugins to system directories (macOS)
# IMPORTANT: Always use this script instead of manual cp.
# cp -R does NOT overwrite existing .component/.vst3 bundles reliably.
./scripts/install-plugins.sh            # debug build
./scripts/install-plugins.sh --release  # release build
```

## Architecture

### Audio Processing Pipeline (PluginProcessor.cpp)

The plugin uses a dual-threaded architecture:
- **Audio thread**: Collects raw fullband samples into 512-sample requests and returns completed hops through bounded ring buffers.
- **Inference thread**: Owns the model's recurrent/overlap-add state and runs ONNX inference asynchronously.

Do not reintroduce the former external context/reflection padding, HP/LP crossover, input normalization, LP reinjection, vocals/soft gates, low-band stabilizer, or chunk-tail crossfade into the qualified path. The graph was validated on unmodified fullband audio and owns its analysis and overlap-add processing.

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

Initialize all three persistent state tensors to zero. `separated_chunk` is aligned to `past_audio`, not `audio_chunk`; therefore the first result after initialization is pre-roll and must be discarded. Feed one final zero `audio_chunk` to flush the last real hop in finite renders.

Only the inference worker may advance model state. Reset `past_audio`, `overlap_add_buffer`, and `fusion_hidden` on transport starts, seeks, scrubs, loop wraps, inference failures, and input-sequence gaps. Epoch changes during an in-flight run invalidate that output; zero the state before accepting another sequence. A reset always creates a new pre-roll result.

### Latency and Fallback

The native pipeline has two fixed hops of latency:

- 512 samples to collect and queue the current request.
- 512 samples because the graph emits the preceding hop.

Report 1,024 samples (23.22 ms at 44.1 kHz) to the host for PDC, and keep the main/fallback delay aligned to the same timeline. Do not advertise the 11.61 ms hop interval as end-to-end plugin latency.

If inference is unavailable or late, use a latency-aligned dry fallback. The final output stage must still route the residual to `Other`, so the four enabled internal stems sum to Main throughout startup, provider transitions, and underruns. Never let a late result reappear against a newer dry timeline.

### Mixture-Lossless Invariant

The model output order is `[drums, bass, vocals, other]`. Preserve drums, bass, and vocals, then calculate:

```text
other = aligned_mixture - drums - bass - vocals
```

The ONNX graph applies this correction, and the native output stage reapplies it after provider-specific numerical differences or fallback crossfades. In floating-point processing, the four internal stems must reconstruct the latency-aligned Main mixture. Do not independently normalize, clip, gate, or quantize stems after the final correction. “Mixture-lossless” does not claim recovery of the studio-original sources; integer PCM exports require another residual correction after final quantization.

At host sample rates other than 44.1 kHz, fail closed: do not start the separator, report the unsupported rate, and use the safe non-model path. Do not silently run the graph at the wrong rate.

### Output Bus Layout

5 output buses total: Main (latency-aligned mixture), then 4 stereo stem buses:
Drums (model index 0), Bass (model index 1), Other (model index 3), Vocals (model index 2)

### Model

`model/model.onnx` is the only model payload bundled into plugin Resources. It is self-contained; do not require or ship the obsolete `model.onnx.data` file.

Qualified model SHA-256:

```text
52fdc46d015819821dae19ef272b6bc4ccf441a0274d7d9c8bb44af50eafef8c
```

Configuration and runtime loading must fail closed unless the artifact SHA/size, all four input/output names, float32 types, static shapes, streaming metadata, checkpoint SHA, and residual-to-source-index-3 policy match the qualified contract.

### CPU Operations

CPU is the reference and supported deployment path. The worker must complete each graph run inside the 11.61 ms hop interval; the larger 23.22 ms PDC does not double the recurring compute budget. A 500-hop native Windows ORT 1.26.0 qualification measured 3.69 ms mean, 4.10 ms p95, 4.42 ms p99, 5.83 ms maximum, and zero deadline misses on the development CPU. Retain underrun telemetry and benchmark the complete native path on every target platform.

Prefer preallocated input/output/state storage and avoid allocations in both the audio callback and steady-state inference loop. The shipping runtime is CPU-only; do not add an execution provider without a separate numerical, packaging, and real-time qualification pass.

## Code Style

- C++20 standard
- Chromium-based clang-format (run `pre-commit install` for auto-formatting)
- Warnings treated as errors

## Testing

Tests are in `test/source/`. Run with `ctest --preset default`.

Changes to the streaming path should cover, at minimum: exact graph names/shapes and model identity; state progression; pre-roll and final flush; reset determinism; sequence-gap recovery; 1,024-sample main/stem alignment across host block sizes; bus ordering; finite outputs; and `Main == Drums + Bass + Other + Vocals` during normal inference, startup, and underrun fallback. Keep performance benchmarks stateful—the old stateless 2,560-sample benchmark is not representative.

## Platform Notes

- **macOS**: ONNX Runtime bundled via install_name_tool rpath fixes.
- **Windows**: the CPU ONNX Runtime DLL is delay-loaded and bundled beside the plugin binary.
