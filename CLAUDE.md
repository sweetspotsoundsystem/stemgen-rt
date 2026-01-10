# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
```

## Architecture

### Audio Processing Pipeline (PluginProcessor.cpp)

The plugin uses a dual-threaded architecture:
- **Audio thread**: Collects samples into ring buffer, applies crossover/gating, retrieves processed stems
- **Inference thread**: Runs ONNX model inference asynchronously to avoid blocking audio

Key DSP components:
- **Overlap-add streaming**: 512-sample chunks with 1024-sample context windows
- **LR4 crossover at 80Hz**: Low frequencies bypass the neural network (model can't handle sub-bass)
- **Input normalization**: Normalizes input to -12dB RMS before inference, applies inverse gain after. Reduces model noise floor on quiet passages.
- **Vocals gate**: Dual-criteria gate (energy ratio + absolute level) with asymmetric attack/release to eliminate spurious vocals content on instrumentals. Gated content transfers to "other" stem.
- **Soft input gating**: Eliminates noise floor artifacts when input is silent
- **Dry signal fallback**: Crossfades to input signal on inference underruns

### Output Bus Layout

4 stereo output buses: Drums (model index 0), Bass (model index 1), Other (model index 3), Vocals (model index 2)

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
