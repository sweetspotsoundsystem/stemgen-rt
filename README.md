# StemgenRT

Separate a stereo music mix into **Drums, Bass, Other and Vocals** in your DAW.
Main carries the complete delayed mix; the four stem outputs reconstruct it.

StemgenRT uses a trained [HS-TasNet](https://github.com/sweetspotsoundsystem/HS-TasNet)
model at **44.1 kHz**, processing 128 samples at a time. With a **128-sample host
buffer**, the plugin reports **256 samples / 5.80 ms** of delay. Inference runs on
one dedicated CPU worker, and the audio callback never waits for it.

## Download

[StemgenRT 0.5.0](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/tag/v0.5.0)
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
and separate recurrent memories. Version 0.5.0 packs the attention query,
key and value projections into one integer product, bringing the total to
seventeen. The exact graph scores **4.455173 dB SDR** on the unchanged
development panel, versus 4.455153 dB for PR #17. The source
FP32 checkpoint scores 4.465157 dB. The [deployment report](model/quality-deployment.json)
retains all 56 track/stem comparisons and 840 paired source-view windows.
The 5 dB goal and instrumental listening acceptance remain unmet.

Raw input levels, the linked near-silence confidence fade and
`Other = Main - Drums - Bass - Vocals` are preserved. The native Linux suite
and the [physical M4 Pro suite](model/macos-validation.json) each passed 164
tests, with one platform-specific skip and seven disabled tests. Coverage
includes independent PyTorch parity, resets, partial EOF,
alignment, queue recovery, reconstruction and callback heap traffic.

The plugin now disables KleidiAI through ORT's session configuration. The
retained [M4 Pro parent evidence](model/macos-runtime-parity-diagnostic.json)
fails all eight cases with backend defaults and passes all eight with KleidiAI
disabled. Both independent PyTorch parity tests now pass on the physical M4 Pro
with the production setting and unchanged `1e-5` tolerance. All 24 backend
diagnostic cases also pass on Linux for this graph.

The graph reduced local median block p50 by 5.14% compared with PR #17 under
concurrent training. The backend setting's cost on M4 must be measured together
with this graph. The user reports that the installed PR #19 plugin works well.
That report has no recorded duration or fallback-counter trace; sustained zero
fallback and net M4 performance remain unqualified.

Follow [the M4 test instructions](M4_TESTING.md) for numerical checks, repeated
extended soaks and installed-DAW playback. Keep one inference worker and retain
the previous complete plugin bundle. See [the model report](model/README.md)
for exact quality changes, timing evidence and limitations.

Built with [JUCE](https://github.com/juce-framework/JUCE) and
[ONNX Runtime](https://github.com/microsoft/onnxruntime).
