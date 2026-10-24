# Working on StemgenRT

StemgenRT is a JUCE/ONNX Runtime stereo source-separation plugin. The released independent model uses a 128-sample hop at 44.1 kHz with thirteen persistent states. Graph delay plus asynchronous scheduling totals 256 samples with a 128-sample host buffer. Keep model identity and streaming behavior synchronized across runtime, tests and packaging. Use the established StemgenRT product name, bundle ID and plug-in code for release compatibility.

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

`cmake/QualifiedModelContract.cmake` is the authoritative identity, geometry and metadata source. CMake generates the C++ contract; shell tools obtain it through `cmake/PrintModelContract.cmake`. Regenerate with `scripts/generate-independent-contract.py`. Do not duplicate identities in packaging scripts. `model/model.onnx` is self-contained and tracked by Git LFS. Configuration verifies SHA/size; loading validates all fourteen inputs, fourteen outputs, float32 static shapes and the complete experimental metadata entry.

The graph contains independent Drums (five states), Bass (four states) and Vocals (four states), then physical-residual Other. Exact prefixed names and shapes are in `model/independent-model.json`. Bass carries 3968 history samples; Drums/Vocals carry 896. There are no shared learned parameters. State scales are native FP32; never apply the released model's state rescaling.

Initialize all thirteen states to zero and carry every returned state unchanged. The first call after reset succeeds with `outputValid=false`. Call N emits input N-1. Pad a partial final hop once, submit exactly one zero graph hop, then only drain queued output. Do not reuse states from a different model or add a second graph flush.

Only the inference worker advances/resets model state. Audio-thread resets invalidate epochs in bounded time; an in-flight old-epoch run must never publish into the new stream. Input sequence gaps reset state and create one invalid pre-roll result.

## Real-time and routing rules

- No waits, locks, allocation or PDC changes in real-time `processBlock`. Offline rendering may wait with a bounded timeout.
- Preserve exact sample timestamps through the queue and output ring. Discard elapsed output instead of shifting it to a newer range. Only already-published results can be claimed at a real-time callback boundary.
- Prepared 44.1 kHz block sizes from 1 to 65536 are admitted. Report the full accumulation/scheduling reserve. Larger or unexpected callback requirements use aligned fallback. Other sample rates are disabled until the new graph is validated through the existing converters.
- Main is the complete delayed native input. Output buses are Main, Drums, Bass, Other, Vocals; graph source order is Drums, Bass, Vocals, Other.
- Keep raw input levels unchanged. Do not add per-hop normalization, external context/reflection padding, crossover reinjection, bass processing, input gates or output clipping.
- Preserve all four graph outputs in `OnnxRuntime`. After the existing output confidence/recovery fade, `Other = Main - Drums - Bass - Vocals`. Complete fallback is zero Drums/Bass/Vocals and Main in Other.
- Preserve the linked confidence envelope: instantaneous open, 50 ms hold, 60 dB/100 ms release, smoothstep from -96 to -72 dBFS peak. Changes need new numerical/listening evidence.
- Keep `mlas.disable_kleidiai=1` in production sessions. The retained exact-parent M4 Pro diagnostic fails independent numerical parity with backend defaults and passes with this setting. A new graph still requires its own physical M4 parity and timing evidence.
- Production inference uses one dedicated worker on every platform: ORT sequential execution, intra-op 1, inter-op 1, and spinning disabled. Explicit overrides are for measurement only; qualification must report an actual configured count of one. A previous model's timing does not qualify this one.
- The macOS scheduling candidate requests a time-constraint worker for playback and QoS when prepared offline. AU context changes use the bounded mailbox; only the worker joins/leaves and retires retained workgroups. Preserve overflow invalidation and stop/start ownership. Qualify the default playback path without a diagnostic scheduling override, and keep native membership tests distinct from installed DAW evidence.

## Verification

Use C++20, Chromium clang-format and warnings as errors. Keep ONNX compile definitions PUBLIC because `PluginProcessor.h` has conditional class members and tests must exercise the same runtime layout.

Streaming changes require model identity/ABI checks; all-four-stem PyTorch parity; first-call invalid pre-roll; one-flush/partial-EOF recovery; reset determinism; queue gap/epoch races; exact PDC/Main alignment; non-finite input handling; fallback reconstruction; and variable offline callback coverage. Fixtures are synthetic float32 audio from independent CPU PyTorch reconstruction of the declared native FP32 models, with portable identity and format metadata in `test/fixtures/independent-pytorch.json`. Keep the independent PyTorch reference; never regenerate expected outputs from ORT. Record deployment-graph quality separately from the source checkpoint's FP32 score.

Keep timing qualification separate from correctness. The disabled paced test measures the complete plugin callback/worker path; direct `Run` timings alone cannot establish DAW readiness. Keep the concise model quality and timing limitations in `model/README.md`. Distinguish user-reported machine results from independently inspected evidence.

## Experimental independent-worker candidate

The default source build still selects one worker. This experimental PR's CI
artifacts explicitly use `STEMGENRT_DEFAULT_INDEPENDENT_WORKERS=3`. The queue
worker runs Bass and gathers persistent Drums/Vocals helpers; each ORT session
remains single-threaded. Preserve the atomic callback path and whole-hop epoch
ownership. Never publish a partial set of stem outputs. The current schema-2
worker trace is single-worker only; do not label gather time as a single ORT run
or primary-thread CPU as total inference CPU. Compare modes using the native
complete-hop benchmark and the uninstrumented actual-plugin paced harness.
Retain the full matching model/training partition exporter and parent graph.
