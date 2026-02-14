# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

StemgenRT is a real-time low-latency music source separation plugin built with JUCE and ONNX Runtime. It separates stereo audio into 4 stems (drums, bass, other, vocals) using an HS-TasNet neural network model.

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
- **Audio thread**: Collects fullband samples into ring buffer, applies gating, retrieves processed stems
- **Inference thread**: Runs ONNX model inference asynchronously to avoid blocking audio

Key DSP components:
- **Overlap-add streaming**: 512-sample chunks with 1024-sample context windows
- **Chunk boundary crossfade**: Extracts overlap tail (extra samples beyond center region) from each model output and crossfades with the next chunk's start to eliminate boundary discontinuities
- **HP/LP split with LP reinjection**: Input is split with LR4 crossover; HP is fed to the model, while LP is bypassed and reinjected after inference (currently biased toward bass vs drums)
- **Input normalization**: Context-aware normalization that computes RMS over combined context + input buffers to -12dB target. Prevents extreme gains when levels differ between context and input (e.g., loud kick tail in context, silence in input).
- **Vocals gate**: Dual-criteria gate (energy ratio + absolute level) with asymmetric attack/release to eliminate spurious vocals content on instrumentals. Gated content transfers to "other" stem.
- **Soft input gating**: Eliminates noise floor artifacts when input is silent
- **Low-band stabilizer**: Rebuilds stable low-frequency stem distribution from dry-constrained low-band energy and suppresses synthetic high-band leakage on low-only material (e.g., LPF kick buzz in other/vocals/drums)
- **Dry signal fallback**: Crossfades to latency-aligned dry signal on inference underruns

Main bus is delayed dry passthrough (delayed by `kOutputChunkSize` so it is latency-aligned with the stems and matches PDC). Stem outputs are model output with LP reinjection, stabilizer, and gates applied (no residual redistribution).

### Output Bus Layout

5 output buses total: Main (delayed dry passthrough), then 4 stereo stem buses:
Drums (model index 0), Bass (model index 1), Other (model index 3), Vocals (model index 2)

### Model

`model/model.onnx` - HS-TasNet model bundled into plugin Resources. Input: stereo audio chunks. Output: 4 separated stems.

## Code Style

- C++20 standard
- Chromium-based clang-format (run `pre-commit install` for auto-formatting)
- Warnings treated as errors

## Testing

Tests are in `test/source/AudioProcessorTest.cpp`. Run with `ctest --preset default`.

## Platform Notes

- **macOS**: ONNX Runtime bundled via install_name_tool rpath fixes.
- **Windows**: CUDA GPU support available. DLLs are delay-loaded.
