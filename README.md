# StemgenRT

**StemgenRT** separates stereo audio into Drums, Bass, Vocals and Other
using three independently trained models. Other is the input minus those three
estimates. The plug-in retains the **StemgenRT** name, host IDs and controls.
The independent model was introduced in 0.7.0.

This checkout prepares **0.8.0**, adding [three independent inference workers](docs/parallel-workers.md)
and [platform scheduling](RUNTIME_SCHEDULING.md) to the existing plugin. Source
builds and CI bundles select three workers by default.
The model weights, streaming interface and latency are unchanged.

The 0.8.0 launch candidate targets **Apple Silicon macOS 14+, 44.1 kHz**
(AU/VST3). **Windows x86-64 is experimental** and still has measured underruns.
The old 0.7.0 binaries predate the runtime fixes and are retained as historical
artifacts; use the current candidate for qualification. A new official release
has not been tagged. Current macOS CI packages are ad-hoc signed and unnotarized.
See [installation/signing](docs/mac-installation.md) and
[the final M4 DAW checks](docs/m4-final-daw-checks.md).

## Audio and performance

Use **44.1 kHz**. Output buses are Main, Drums, Bass, Other, Vocals; Main is the
complete delayed input. A 128-sample host buffer gives **256 samples / 5.805 ms**
of graph-plus-worker latency, excluding hardware latency. Larger or irregular
buffers include their additional scheduling reserve in reported latency.
At unsupported sample rates the release UI explicitly requests 44.1 kHz.

| Host buffer (samples) | 64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported PDC (samples) | 320 | 256 | 384 | 640 | 1152 |
| Reported PDC (ms) | 7.26 | 5.80 | 8.71 | 14.51 | 26.12 |

Model synthesis-frame latency is separately 256 samples / 5.805 ms; adding the
asynchronous queue under that convention gives 384 samples / 8.707 ms at a
128-sample host buffer. Reported PDC and synthesis-frame latency use different
reference points. Neither is device round-trip latency.

The listening comparison favored cleaner vocals, better silence behavior and
less vocal bleed in drums and bass. Other can contain stronger residual errors.
The three-worker runtime measured 0.694 ms per complete inference hop on the
M4, versus 2.304 ms with one worker. Its balanced sustained standalone test
recorded no missed results or underruns across 60 measured minutes per mode.
The M1 Pro quick comparison measured 1.060 ms versus 2.244 ms, with no missed
results across 40,000 measured plugin callbacks per mode. These are separate
machine results; the M1 Pro has not received the same endurance test. The user
also reports stable playback in Ableton on the M4, without a separately inspected
DAW timing trace. On the Ryzen 5 5500, the Windows candidate improved average
inference from 3.246 ms with one worker to 1.525 ms with three, but still missed 224 results across
40,400 callbacks. It does not yet pass the 128-sample playback gate. See
[platform evidence](docs/parallel-workers.md) for exposure and comparison limits.
The published 0.7.0 release bundles precede these runtime improvements.
Correctness tests and algorithmic latency do not establish the CPU deadline.
The audio callback never waits for inference; missed output uses aligned fallback.
See [model details](model/README.md) and [M4 testing](M4_TESTING.md).

## Build

On an Apple Silicon Mac, use native arm64 CMake/Ninja and Xcode command-line tools:

```sh
git lfs pull
./scripts/download-onnxruntime.sh
cmake --preset release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build --preset release --parallel 4
ctest --preset release
./scripts/install-plugins.sh --release
```

The macOS installer replaces complete bundles and signs embedded libraries before
the outer bundle. For Windows x64 (Windows 10 version 1803 or later), use a Visual
Studio developer shell with Ninja and the x64 Visual C++ runtime installed ([official Microsoft download](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)).
This is also a prerequisite for prebuilt Windows ZIPs; install the x64 v14
redistributable, including `MSVCP140_ATOMIC_WAIT.dll`. Windows ZIPs are unsigned.
[Host validation and crash evidence](docs/windows-debugging.md) remain necessary
for the Ableton insertion report; the synthetic restart regression is fixed.

Build with:

```powershell
git lfs pull
./scripts/download-onnxruntime.ps1
cmake --preset release
cmake --build --preset release --parallel 4
ctest --preset release
./scripts/install-plugins.ps1 -Release
```

The Windows installer validates all four models and required binaries, verifies
the staged copy, and preserves the previous installation until replacement
succeeds. Close the DAW before installing. Keep `onnxruntime.dll` beside the VST3
binary and retain the complete Resources directory. Single-worker diagnostic
builds can set `-DSTEMGENRT_DEFAULT_INDEPENDENT_WORKERS=1` during configuration.
The source contains the actual model via Git LFS and independent PyTorch fixtures.

## Model and training code

The self-contained FP32 ONNX graph has 14 inputs and 14 outputs: audio plus 13
persistent states. Five belong to Drums, four to Bass and four to Vocals.
The full interface and authenticated EMA identities are under `model/`.

Matching model, training, objectives, data augmentation, optimizer/EMA recovery,
evaluation and export code is maintained in
[StemgenRT-5.8](https://github.com/sweetspotsoundsystem/StemgenRT-5.8), with a
matching source archive at `model/training-source/stemgenrt-0.8.0.tar.gz`.
Private experiment archives and draft papers are excluded from this release.

## Scheduling diagnostics

The optional single-worker macOS comparison independently controls synthetic
callback and worker scheduling while retaining this exact graph and the
production ORT configuration. It explicitly selects one worker in both traced
and untraced trials; it does not measure the three-worker default. It runs four
balanced short rounds, two five-minute endurance
trials with a real-time synthetic callback, then separate worker traces:

```sh
git lfs pull
./scripts/download-onnxruntime.sh
python3 scripts/compare-paced-scheduling-macos.py \
  --output /path/to/new-scheduling-results \
  --power-notes "AC; Low Power Mode off; actual workloads"
```

Allow about 20 minutes after building. Return the result ZIP. Initial and final
Mach policy readbacks, callback timing, queue misses, underruns and reconstruction
must all pass their existing gates. Failed timing trials are retained while the
comparison continues. These explicit controls do not install a plugin or qualify
a DAW. See [the protocol](M4_TESTING.md).

For the next qualification stage, use `--protocol long-soak`. It runs two
separate 30-minute real-time-worker trials with a 30-minute QoS-worker control
between them, all with a real-time synthetic callback and tracing disabled.
Allow about 90 minutes after compilation. The plan records every trial and its
exact exposure before measurement; all original correctness and timing gates
remain required.

## Licences and source

Original code is MIT. The JUCE-linked open-source binary uses AGPLv3 terms;
JUCE 8.0.11 embeds the MIT VST3 SDK 3.8. See [bundled notices](licenses/README.txt).
Distribute each binary with its matching `StemgenRT-Source.tar.gz` CI artifact,
which includes plugin code, pinned JUCE source and the matching model/training
package. Rebuild with the supplied JUCE source using `-DCPM_JUCE_SOURCE=/path/to/JUCE`.
Downloaded ORT and build dependencies retain their own terms. ONNX Runtime
telemetry is disabled before sessions are created.

[Weights have separate non-commercial terms](licenses/WEIGHTS-LICENSE.md).
Those terms do not establish third-party dataset rights or commercial clearance.
The [model card and quality results](https://github.com/sweetspotsoundsystem/StemgenRT-5.8/blob/main/MODEL_CARD.md)
report the vocal gain and residual Other regression. The manuscript remains private.
