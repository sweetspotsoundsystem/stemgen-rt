# StemgenRT

Separate a stereo music mix into **Drums, Bass, Other and Vocals** in your DAW.
Main carries the complete delayed mix; the four stem outputs reconstruct it.

StemgenRT uses a trained [HS-TasNet](https://github.com/sweetspotsoundsystem/HS-TasNet)
model at **44.1 kHz**, processing 128 samples at a time. With a **128-sample host
buffer**, the plugin reports **256 samples / 5.80 ms** of delay. Inference runs on
one dedicated CPU worker, and the audio callback never waits for it.

## Use

1. Set the session to **44.1 kHz** and select a **128-sample buffer** for the
   lowest reported latency.
2. Insert StemgenRT on a stereo track and enable its additional stereo outputs.
3. Route **Main, Drums, Bass, Other, Vocals**. Summing the four stems reconstructs
   Main; adding Main again doubles the mix.

The editor shows the active delay and separation status. When a result misses
its deadline, the complete delayed mix goes to Other for that interval. Late
results are discarded, and separation fades back in over 64 samples when ready.
Other sample rates use Main/Other fallback and display a setup message.

| Prepared host buffer | Reported delay | At 44.1 kHz |
| --- | --- | --- |
| 64 | 320 samples | 7.26 ms |
| 128 | 256 samples | 5.80 ms |
| 256 | 384 samples | 8.71 ms |
| 512 | 640 samples | 14.51 ms |
| 1024 | 1152 samples | 26.12 ms |

Smaller host buffers require an extra scheduling reserve after a complete model
hop arrives. The host must reprepare the plugin after changing its buffer setup.
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

The model combines spectrogram and waveform estimates with causal attention
and separate recurrent memories for the two branches. This development graph
extends integer arithmetic to fourteen matrix products. It scores 4.455150 dB
SDR on the development panel, versus 4.455188 dB for v0.4.0; its source EMA
checkpoint scores 4.465157 dB. Instrumental vocal leakage is essentially
unchanged. The [deployment report](model/quality-deployment.json) records the
exact graph separately from its source checkpoint and includes all regressions.
It consumes raw stereo samples and preserves their level. The plugin applies a
linked near-silence confidence fade, then calculates
`Other = Main - Drums - Bass - Vocals` to preserve the complete mix.
See the [model interface and validation](model/README.md) for details.

Tests cover independent PyTorch waveform parity, streaming resets, partial
final clips, sample alignment, queue recovery, output reconstruction, variable
offline callbacks and C++ heap traffic in the audio callback. Model validation
and target-Mac testing are documented in [model/README.md](model/README.md).

Real-time performance depends on the machine and host load. The user reported
1,408 fallback samples after a few minutes on M4 with released v0.4.0.
This candidate passed 162 native Linux correctness tests and measured about
40% lower local inference cost under concurrent load. M4 correctness, paced
timing and installed-DAW playback remain unmeasured. Follow the
[M4 test instructions](M4_TESTING.md) with the same single inference worker.
This checkout is a speed test candidate; no release replacement has been selected.

Built with [JUCE](https://github.com/juce-framework/JUCE) and
[ONNX Runtime](https://github.com/microsoft/onnxruntime).
