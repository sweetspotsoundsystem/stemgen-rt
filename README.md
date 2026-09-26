# StemgenRT

Separate a stereo music mix into **Drums, Bass, Other and Vocals** in your DAW.
Main carries the complete delayed mix; the four stem outputs reconstruct it.

StemgenRT uses a trained [HS-TasNet](https://github.com/sweetspotsoundsystem/HS-TasNet)
model at **44.1 kHz**, processing 128 samples at a time. With a **128-sample host
buffer**, the plugin reports **256 samples / 5.80 ms** of delay. Inference runs on
one dedicated CPU worker, and the audio callback never waits for it.

## Download

[StemgenRT 0.6.1](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/tag/v0.6.1)
includes macOS AU and VST3 for Apple Silicon (macOS 14+) and Windows x86-64 VST3.
The model and ONNX Runtime are bundled. Replace the complete plugin bundle and
restart the DAW. macOS downloads are ad-hoc signed, without Developer ID signing
or notarization. Archive checksums accompany the release.

## Use

1. Set the session to **44.1 kHz** and select a **128-sample buffer** for the
   lowest reported latency.
2. Insert StemgenRT on a stereo track and enable its additional stereo outputs.
3. Route **Main, Drums, Bass, Other, Vocals**. Summing the four stems reconstructs
   Main; adding Main again doubles the mix.

The editor shows the logo with small white latency and fallback-sample readouts.
The fallback count remains visible at zero. When a result misses
its deadline, the complete delayed mix goes to Other for that interval. Late
results are discarded, and separation fades back in over 64 samples when ready.
Other sample rates use Main/Other fallback.

Host bypass preserves the reported delay: Main and Other carry the complete
delayed mix, and Drums, Bass and Vocals are silent. The worker keeps processing
while bypassed so separation can resume on the current timeline with the usual
recovery fade. Intentional bypass does not increase the fallback counter.
Host reset requests clear pending audio and model state at the next callback.

| Prepared host buffer | Reported delay | At 44.1 kHz |
| --- | --- | --- |
| 64 | 320 samples | 7.26 ms |
| 128 | 256 samples | 5.80 ms |
| 256 | 384 samples | 8.71 ms |
| 512 | 640 samples | 14.51 ms |
| 1024 | 1152 samples | 26.12 ms |

Smaller host buffers require an extra scheduling reserve after a complete model
hop arrives. The host must reprepare the plugin after changing its buffer setup.
The inference queue is allocated during preparation to hold a complete callback
burst, including large supported buffers up to 65,536 samples. This prevents
capacity-related drops; meeting playback deadlines still depends on the machine.
Offline rendering supports varying callbacks and waits for inference. Render the
reported tail at transport stop to retain the final samples.

## Build and install

Requirements: CMake 3.22+, C++20, Git LFS and the ONNX Runtime 1.26.0 CPU SDK.
CMake fetches JUCE 8.0.6 and GoogleTest 1.16.0. macOS builds target Apple Silicon
and macOS 14 or newer; Windows builds produce VST3.

```bash
git lfs pull
./scripts/download-onnxruntime.sh
cmake --preset release
cmake --build --preset release
ctest --preset release
./scripts/install-plugins.sh --release
```

On Windows, use the corresponding `.ps1` download and install scripts. The macOS
installer replaces complete AU/VST3 bundles and verifies signing. Restart your
DAW after installation. Existing plugin identifiers and output routing remain
compatible with saved sessions.

For Linux development, install JUCE's ALSA, FreeType, Fontconfig and X11 headers,
place the ORT CPU SDK in `libs/onnxruntime`, and configure a Release Ninja build.

## Model and checks

This candidate uses the frozen **StemgenRT-5.8 teacher004 EMA** checkpoint.
The exact integer graph scores **4.564148 dB SDR**, versus 4.455173 dB
for shipped v0.6.1 on the unchanged development panel. The FP32 source score
is 4.564402 dB. See [the model report](model/README.md) for per-stem changes,
leakage, provenance and evidence boundaries.

The 17 integer products, eight states, 128-sample hop and **256-sample total
latency** with a 128-sample host buffer are retained. The teacher is absent
from inference. Raw input levels, the confidence fade, residual Other routing,
one inference worker and `mlas.disable_kleidiai=1` are unchanged.

Independent short/long PyTorch parity and 174 Linux native tests pass.
Physical M4 parity, sustained zero-fallback playback and listening acceptance
remain pending for these weights. Follow [the M4 instructions](M4_TESTING.md)
and preserve the shipped v0.6.1 bundles for comparison and rollback.

Built with [JUCE](https://github.com/juce-framework/JUCE) and
[ONNX Runtime](https://github.com/microsoft/onnxruntime).
