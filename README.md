# StemgenRT

**StemgenRT 0.7.0** separates stereo audio into Drums, Bass, Vocals and Other
using three independently trained models. Other is the input minus those three
estimates. The plug-in retains the **StemgenRT** name, host IDs and controls.
This release replaces the previous joint separation model.

This development branch adds [three independent inference workers](docs/parallel-workers.md)
and [platform scheduling](RUNTIME_SCHEDULING.md) to the existing plugin. CI
candidate bundles select three workers; ordinary source builds default to one.
The model weights, streaming interface and latency are unchanged.

[Download the release](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/tag/v0.7.0)
for Apple Silicon macOS 14+ (AU/VST3) or Windows x86-64 (VST3). The model and ONNX
Runtime 1.26.0 are bundled. macOS packages are ad-hoc signed, not notarized.
Close the DAW before replacing the complete plug-in bundle, then restart it.
Existing projects will use the new model after updating.

## Audio and performance

Use **44.1 kHz**. Output buses are Main, Drums, Bass, Other, Vocals; Main is the
complete delayed input. A 128-sample host buffer gives **256 samples / 5.805 ms**
of graph-plus-worker latency, excluding hardware latency. Larger or irregular
buffers include their additional scheduling reserve in reported latency.

The listening comparison favored cleaner vocals, better silence behavior and
less vocal bleed in drums and bass. Other can contain stronger residual errors.
The three-worker candidate measured 0.694 ms per complete inference hop on the
M4, versus 2.304 ms with one worker. Its balanced sustained standalone test
recorded no missed results or underruns across 60 measured minutes per mode.
The M1 Pro quick comparison measured 1.060 ms versus 2.244 ms, with no missed
results across 40,000 measured plugin callbacks per mode. These are separate
machine results; the M1 Pro has not received the same endurance test. The user
also reports stable playback in Ableton on the M4, without a separately inspected
DAW timing trace. On the Ryzen 5 5500, the Windows candidate improved average
parallel inference from 3.753 to 1.525 ms, but still missed 224 results across
40,400 callbacks. It does not yet pass the 128-sample playback gate. See
[platform evidence](docs/parallel-workers.md) for exposure and comparison limits.
The published 0.7.0 release bundles precede this experimental runtime work.
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

The installer replaces complete bundles and signs embedded libraries before the
outer bundle. For Windows, use `scripts/download-onnxruntime.ps1`, configure with
CMake/Visual Studio, build and test Release, then use `scripts/install-plugins.ps1`.
The source contains the actual model via Git LFS and independent PyTorch fixtures.

## Model and training code

The self-contained FP32 ONNX graph has 14 inputs and 14 outputs: audio plus 13
persistent states. Five belong to Drums, four to Bass and four to Vocals.
The full interface and authenticated EMA identities are under `model/`.

Matching model, training, objectives, data augmentation, optimizer/EMA recovery,
evaluation and export code is maintained in
[StemgenRT-5.8](https://github.com/sweetspotsoundsystem/StemgenRT-5.8), with a
matching source archive at `model/training-source/stemgenrt-0.7.0.tar.gz`.
Private experiment archives and draft papers are excluded from this release.

## Scheduling diagnostics

The optional macOS comparison independently controls synthetic callback and
worker scheduling while retaining this exact graph and the production ORT
configuration. It runs four balanced short rounds, two five-minute endurance
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
