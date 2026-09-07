# Working on StemgenRT

StemgenRT is a JUCE/ONNX Runtime stereo source-separation plugin. This experimental branch integrates the asymmetric-window hop128 teacher endpoint (5000 total updates). Its 128-sample graph delay plus 128 samples of asynchronous scheduling gives 256 samples / 5.80 ms at a 44.1 kHz / 128-sample prepared host configuration. Long native waveform parity passes; Linux realtime timing fails. Quality and listening acceptance and Apple M4 DAW timing are pending. The model contract's historical `QUALIFIED` variable prefix is an identity/ABI lock, not proof of platform qualification.

## Commands

```bash
./scripts/download-onnxruntime.sh  # macOS; .ps1 on Windows
cmake --preset release
cmake --build --preset release
ctest --preset release
./scripts/install-plugins.sh --release
./scripts/qualify-macos.sh ../stemgenrt-m4-evidence
```

Use the installer to replace entire macOS bundles, rather than `cp -R` over existing bundles. Sign embedded dylibs before outer bundles. Apple builds use arm64/macOS 14+, with the official ORT 1.26.0 CPU SDK. Windows loads the sibling ORT DLL explicitly. Keep the macOS VST3 auto-manifest helper disabled; Windows retains its manifest.

## Model identity and ABI

`cmake/QualifiedModelContract.cmake` is the authoritative identity, geometry and metadata source. CMake generates the C++ contract; shell tools obtain it through `cmake/PrintModelContract.cmake`. Do not duplicate identities in packaging scripts. `model/model.onnx` is self-contained and tracked by Git LFS. Configuration verifies SHA/size; loading validates all five inputs, five outputs, float32 static shapes and 55 metadata entries.

| Input | Shape | Output |
| --- | --- | --- |
| `audio_chunk` | `[1,2,128]` | `separated_chunk`: `[1,4,2,128]` |
| `audio_history` | `[1,2,896]` | `next_audio_history`: same shape |
| `fusion_hidden` | `[2,1,1000]` | `next_fusion_hidden`: same shape |
| `spectral_numerator_tail` | `[1,4,2,128]` | `next_spectral_numerator_tail`: same shape |
| `waveform_tail` | `[1,4,2,128]` | `next_waveform_tail`: same shape |

Initialize all four states to zero and carry every returned state unchanged. The first call after reset succeeds with `outputValid=false`. Call N emits input N-1. Pad a partial final hop once, submit exactly one zero graph hop, then only drain queued output. Do not import the old c91 three-state ABI, reuse its history, or add a second graph flush.

Only the inference worker advances/resets model state. Audio-thread resets invalidate epochs in bounded time; an in-flight old-epoch run must never publish into the new stream. Input sequence gaps reset state and create one invalid pre-roll result.

## Real-time and routing rules

- No waits, locks, allocation or PDC changes in real-time `processBlock`. Offline rendering may wait with a bounded timeout.
- Preserve exact sample timestamps through the queue and output ring. Discard elapsed output instead of shifting it to a newer range. Only already-published results can be claimed at a real-time callback boundary.
- Prepared 44.1 kHz block sizes from 1 to 65536 are admitted. Report the full accumulation/scheduling reserve. Larger or unexpected callback requirements use aligned fallback. Other sample rates are disabled until the new graph is validated through the existing converters.
- Main is the complete delayed native input. Output buses are Main, Drums, Bass, Other, Vocals; graph source order is Drums, Bass, Vocals, Other.
- Keep raw input levels unchanged. Do not add per-hop normalization, external context/reflection padding, crossover reinjection, bass processing, input gates or output clipping.
- Preserve all four graph outputs in `OnnxRuntime`. After the existing output confidence/recovery fade, `Other = Main - Drums - Bass - Vocals`. Complete fallback is zero Drums/Bass/Vocals and Main in Other.
- Preserve the linked confidence envelope: instantaneous open, 50 ms hold, 60 dB/100 ms release, smoothstep from -96 to -72 dBFS peak. Changes need new numerical/listening evidence.
- Retain the user's two-thread macOS ORT cap as the starting policy. Other platforms retain four. A previous model's timing does not qualify this one. Explicit thread overrides are for measurement.

## Verification

Use C++20, Chromium clang-format and warnings as errors. Keep ONNX compile definitions PUBLIC because `PluginProcessor.h` has conditional class members and tests must exercise the same runtime layout.

Streaming changes require model identity/ABI checks; all-four-stem PyTorch parity; first-call invalid pre-roll; one-flush/partial-EOF recovery; reset determinism; queue gap/epoch races; exact PDC/Main alignment; non-finite input handling; fallback reconstruction; and variable offline callback coverage. Fixtures are synthetic CPU FP32 PyTorch outputs with checkpoint/source hashes in `test/fixtures/cropped1024-pytorch.json`; the generator never uses ORT as its oracle.

Keep timing qualification separate from correctness. The disabled paced test measures the complete plugin callback/worker path; direct `Run` timings alone cannot establish DAW readiness. Preserve failed Linux timing and measured quality deltas in `model/README.md`. Do not imply an M4 measurement was performed here.
