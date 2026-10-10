# StemgenRT Independent — experimental M4 testing

This branch integrates three separately trained 40,000-update EMA models for
Drums, Bass and Vocals into one FP32 ONNX graph. Other is the aligned input minus
those three estimates. It uses a distinct plug-in name, bundle ID and host ID,
so **StemgenRT Independent** can be installed alongside released StemgenRT.

This is a test build. Native source-model evaluation found three failed Other
preservation/leakage guards. The combined ONNX graph and this plug-in still need
physical M4/M4 Pro numerical and timing results; an earlier model's measurements
do not qualify this one. See [model details](model/README.md).

## Build on an M4 Mac

Requires macOS 14+, native arm64 CMake/Ninja and Xcode command-line tools. The
repository includes the trained model and independent PyTorch fixtures via Git
LFS. A supplied source ZIP contains the real files and needs no LFS checkout.

```sh
# Git checkout only: git lfs pull
./scripts/download-onnxruntime.sh
cmake --preset release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build --preset release --parallel 4
ctest --preset release
./scripts/install-plugins.sh --release
./scripts/qualify-macos.sh ../independent-m4-evidence
```

Use 44.1 kHz for DAW tests. Start at a 128-sample host buffer: the declared graph
and worker latency is 256 samples (5.805 ms). Other buffer sizes include their
additional scheduling reserve. Buses are Main, Drums, Bass, Other, Vocals.
Main remains the complete delayed input. The audio callback never waits for
inference; worker misses use aligned fallback and are recorded by qualification.

The separate `stemgenrt-independent-trained-m4` worker packet compares the single
graph against three independent ORT sessions. Its `run_macos.py` needs only Python
3 and Apple's command-line tools; it does not measure DAW callbacks. The plug-in
qualification script exercises the complete callback/worker path.

## Model and training source

The ONNX file has 14 inputs and 14 outputs: audio plus 13 persistent states.
Initialize states to zero and carry every returned state unchanged. The first
output after reset is invalid; each later call emits the preceding 128-sample
input hop. Pad a partial final hop once and submit exactly one zero flush hop.
All input levels remain unchanged. See `model/independent-model.json` for the
full contract and `model/independent-checkpoints.json` for EMA identities.

Matching model, training, objectives, data, recovery, evaluation and export code
is included at `model/training-source/stemgenrt-0.6.2.tar.gz`. The public research branch
is `codex/independent-combined-onnx-m4` in
[StemgenRT-5.8](https://github.com/sweetspotsoundsystem/StemgenRT-5.8).
The exporter is `scripts/export_independent_model.py` in that source package.
The package's version number is inherited from the research source and does not
make this experimental model a v0.6.2 release replacement.
