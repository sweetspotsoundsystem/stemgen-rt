# StemgenRT

Separate a stereo mix into Drums, Bass, Other and Vocals in your DAW. Main carries the complete latency-aligned mix; the four stem outputs reconstruct it.

This experimental branch bundles the **hop128 asymmetric-window teacher** model. With a **44.1 kHz session and a 128-sample host buffer**, it reports **256 samples / 5.80 ms** of delay. The real-time audio callback never waits for inference. Production inference runs on one dedicated worker with no ORT helper threads.

The model passes short export and long native waveform checks. It remains below the accepted 11.6 ms model on the fixed quality panel, and its bass probes are mixed. Listening acceptance and performance in the intended DAW on Apple M4 are pending. See [model provenance and validation](model/README.md) for the measurements.

## Use

1. Set the session sample rate to **44.1 kHz** and the audio buffer to **128 samples** for the lowest latency.
2. Insert StemgenRT on a stereo track and enable its additional stereo outputs in your host.
3. Route the outputs in this order: **Main, Drums, Bass, Other, Vocals**. Avoid summing Main with the stems unless that is intentional.

The editor shows the active delay and whether separation is available. If inference misses its deadline, Main remains intact and the delayed mix goes to Other for that exact interval. Late stems are discarded; they are never replayed over newer audio. Model audio fades back in over 64 samples when processing catches up.

The graph runs at 44.1 kHz. Other session rates use immediate Main/Other fallback and display a setup message. The existing sample-rate converter code remains disabled for this model.

### Host buffers and delay

Other prepared block sizes are supported at 44.1 kHz. The plugin includes accumulation and worker scheduling time in the delay reported to the host:

| Prepared buffer | Reported delay | Delay at 44.1 kHz |
| --- | --- | --- |
| 64 | 320 samples | 7.26 ms |
| **128** | **256 samples** | **5.80 ms** |
| 256 | 384 samples | 8.71 ms |
| 512 | 640 samples | 14.51 ms |
| 1024 | 1152 samples | 26.12 ms |

A smaller host buffer leaves less time after a complete model hop arrives, so additional delay preserves the worker's scheduling reserve. The host must reprepare the plugin when it changes its buffer configuration. An actual real-time callback that needs more delay than prepared uses aligned fallback without changing PDC inside the audio callback.

Offline bounces wait for each model hop and support varying callback sizes. At transport stop, a partial final hop is padded, exactly one zero hop flushes the graph, and the remaining callbacks drain the delayed output. The tail length reported to the host includes the complete delay; the host must render that tail to retain the last samples.

## Build and install

Use CMake 3.22+, a C++20 compiler, Git LFS, and the pinned ONNX Runtime 1.26.0 CPU SDK. JUCE 8.0.6 and GoogleTest 1.16.0 are fetched by CMake. macOS builds target Apple Silicon and macOS 14 or newer; Windows supports its native SDK architecture.

```bash
git lfs pull
./scripts/download-onnxruntime.sh  # macOS; use the .ps1 script on Windows
cmake --preset release
cmake --build --preset release
ctest --preset release
./scripts/install-plugins.sh --release
```

On Windows, use `scripts/install-plugins.ps1`. On macOS, the installer replaces complete AU/VST3 bundles and verifies signing; use it instead of manually copying over old bundles. Restart the DAW after installing. The existing plugin identifier is preserved so sessions retain their routing. Installing this experimental branch replaces the installed StemgenRT version; keep the accepted installer available for rollback.

For Linux development, install JUCE's ALSA, FreeType, Fontconfig and X11 development dependencies and place the ORT CPU SDK in `libs/onnxruntime`, then configure a Release Ninja build. Linux is useful for correctness checks; this PR does not qualify a Linux DAW release.

A Release configuration verifies the bundled model's size and SHA-256. Runtime loading also verifies every input/output name, float32 shape and export metadata entry. The model is a single LFS-tracked `model/model.onnx`; it needs no external weights file.

## Validation

The tests include a stopped worker with a saturated queue and transport seek, plus ready-result recovery, with a C++ heap-traffic probe around real-time callbacks. The probe covers C++ new/delete on the calling thread, not C allocation or provider internals. They also cover streaming state/reset behavior, queue epochs, late-output discard, sample-accurate Main and stem alignment, output reconstruction, low-frequency seams, non-finite input, low-level confidence, and variable offline buffers. The checked-in [PyTorch fixtures](test/fixtures/cropped1024-pytorch.json) independently verify all four native stems and the actual output buses through eight final clip lengths, including one sample and partial hops.

Performance tests are separate from correctness tests. Run the native Mac gate from a fresh checkout with an output directory outside the repository:

```bash
./scripts/qualify-macos.sh ../stemgenrt-m4-evidence
```

That gate builds and verifies the AU/VST3 bundles, requires the model parity tests, sweeps CPU thread counts, and measures 10,000 paced plugin callbacks. A successful machine test still needs an installed-plugin check under representative DAW load. To transfer a sealed source snapshot, use `scripts/package-macos-handoff.sh`.

## Audio contract

The model consumes raw finite stereo samples without gain normalization, filters or external context padding. It uses a 1024-sample analysis window, a 128-sample hop, a 256-sample synthesis frame and four persistent state tensors. Only the inference worker advances those states. Starts, seeks, loop wraps, input gaps and invalid input reset the stream; stale outputs from the old generation are invalidated.

The graph returns **Drums, Bass, Vocals, Other**. The bus order swaps the last two for compatibility. The runtime preserves all four graph estimates. The output writer applies the existing low-level confidence and recovery fade to Drums/Bass/Vocals, then calculates `Other = Main - Drums - Bass - Vocals`. At ordinary listening levels with output available, this reproduces the candidate deployed stems within floating-point rounding.

Main stays at its native input level. The confidence envelope holds peaks for 50 ms, releases by 60 dB per 100 ms, and smoothly opens between -96 and -72 dBFS peak. It suppresses unreliable near-silence model output while preserving the complete mix in Other.

Built with [JUCE](https://github.com/juce-framework/JUCE) and [ONNX Runtime](https://github.com/microsoft/onnxruntime).
