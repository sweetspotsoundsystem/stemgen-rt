# StemgenRT

A real-time music source separation plugin. Drop it on a track and get 4 separate stems: drums, bass, other, and vocals.

This checked c212 export of the c193 current-chunk model processes stereo audio in 512-sample hops and reports exactly 512 samples of latency (11.61 ms at 44.1 kHz) when the host is prepared at 44.1 kHz with a 512-sample callback. It is made for spatializing DJ sets in real time: split the mix into stems, place them in the room, and create an immersive experience.

> [!WARNING]
> The bundled model is the sealed c212 native-DFT export of c193. c194 measured a 4.7142 dB validation score and passed all 40 declared latency-adjusted production guardrails. On the 12-track c197 electronic holdout versus c91 it improved aggregate SI-SDR by 0.1813 dB and low-band SI-SDR by 0.2026 dB; projection SIR changed by -0.1928 dB, above its -0.25 dB floor, so all three holdout gates passed. Raw and optimized ONNX Runtime 1.26 checks also passed. Its status is nevertheless `checked_export_pending_native_plugin_qualification`: complete the target-Mac AU/VST numerical, reset, real-time, and listening qualification before treating it as a public release.

Built with [JUCE](https://github.com/juce-framework/JUCE) and [ONNX Runtime](https://onnxruntime.ai), using [HS-TasNet](https://github.com/sweetspotsoundsystem/HS-TasNet).

Available as VST3 and AU.

![Screenshot](./screenshots/StemgenRT.png)

## Usage

StemgenRT is a multi-output plugin with 4 stereo output buses:

1. **Drums**
2. **Bass**
3. **Other** (synths, guitars, etc.)
4. **Vocals**

To set it up:

1. Insert StemgenRT on your source track (e.g., a DJ mix or full song)
2. Create 4 auxiliary/bus tracks to receive each stem
3. Route each of the plugin's stem outputs to its corresponding aux track

Check your DAW's documentation for multi-output plugin routing.

> [!IMPORTANT]
> This checked-export candidate accepts exactly 44.1 kHz with a 512-sample prepared callback. Other configurations fail closed. The existing higher-rate bridge remains in the source for later qualification.

## Downloads

- [StemgenRT-macOS-AU.zip](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/download/latest/StemgenRT-macOS-AU.zip)
- [StemgenRT-macOS-VST3.zip](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/download/latest/StemgenRT-macOS-VST3.zip)
- [StemgenRT-Windows-VST3.zip](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/download/latest/StemgenRT-Windows-VST3.zip)

> [!NOTE]
> CI seals the macOS bundles with an ad-hoc signature and verifies them strictly after embedding the model and ONNX Runtime. Ad-hoc signing is not Developer ID signing or notarization, so Gatekeeper may still reject a quarantined download. A public trusted release must be Developer ID signed and notarized.

## Building

First, grab the pinned official ONNX Runtime 1.26.0 dependency:

```bash
# macOS
./scripts/download-onnxruntime.sh

# Windows (PowerShell)
./scripts/download-onnxruntime.ps1
```

The provided macOS dependency is Apple Silicon (`arm64`) only, and StemgenRT targets macOS 14.0 or newer because that is the minimum encoded in Microsoft's official ONNX Runtime 1.26.0 dylib. Microsoft does not publish an Intel macOS archive for ONNX Runtime 1.26.0, so Intel macOS artifacts are not built or qualified. Windows downloads the SDK matching the native x64 or ARM64 PowerShell process.

The optional JUCE-generated VST3 `moduleinfo.json` is disabled on macOS because its build-time helper can block in `dyld` before plugin code is entered. The VST3 bundle remains host-scannable and is still sealed and strictly verified; Windows keeps the generated manifest.

Then build with CMake presets:

```bash
cmake --preset default
cmake --build --preset default
```

The one-hop current-chunk scheduler is not yet qualified for a public release on the target Mac. For local listening on macOS, use the default build and install it explicitly as a debug build:

```bash
./scripts/install-plugins.sh --debug
```

The release preset remains available for later qualification work and writes to `build-release`. It enables `STEMGENRT_REQUIRE_QUALIFIED_ORT`, which fails configuration unless the complete official 1.26.0 SDK is present. Do not treat a successful release build as model promotion.

On macOS, install the sealed AU and VST3 bundles into your user plugin directories with:

```bash
./scripts/install-plugins.sh
```

The installer uses Release artifacts by default, so this checked-export candidate must be selected with `./scripts/install-plugins.sh --debug`. Run `./scripts/install-plugins.sh --help` for signing and configuration options.

## How it works

The plugin runs a stateful HS-TasNet graph on a high-priority inference thread:

1. Callback N preserves the native mix for Main/fallback and submits one complete 512-sample request.
2. The inference thread carries previous-audio, fusion-GRU, c130 feature-history, previous-hidden, adapter-valid, 2,048-sample raw-parent history, and 2,048-sample emitted-per-source history state. c193 emits the separated result aligned to input N.
3. The result is scheduled one queue hop later at host sample `(N + 1) * 512`, and Other is re-derived as the exact residual of the raw aligned mixture.

At 44.1 kHz both converters are bypassed with zero added delay and a bit-exact input copy. The preserved higher-rate bridge is disabled by this candidate contract until it is requalified.

The model graph has eight inputs and eight outputs:

| Direction | Tensor | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 512]` |
| Input | `past_audio` | `[1, 2, 512]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Input | `c130_history` | `[1, 20, 128]` |
| Input | `previous_hidden` | `[1, 32, 512]` |
| Input | `adapter_valid` | `[1, 1]` |
| Input | `raw_parent_history` | `[1, 4, 2048]` |
| Input | `emitted_db_history` | `[1, 4, 2048]` |
| Output | `separated_chunk` | `[1, 4, 2, 512]` |
| Output | `next_past_audio` | `[1, 2, 512]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |
| Output | `next_c130_history` | `[1, 20, 128]` |
| Output | `next_previous_hidden` | `[1, 32, 512]` |
| Output | `next_adapter_valid` | `[1, 1]` |
| Output | `next_raw_parent_history` | `[1, 4, 2048]` |
| Output | `next_emitted_db_history` | `[1, 4, 2048]` |

All seven persistent state tensors—`past_audio`, `fusion_hidden`, `c130_history`, `previous_hidden`, `adapter_valid`, `raw_parent_history`, and `emitted_db_history`—start at zero. `separated_chunk` is aligned with the current call's `audio_chunk`, and sequence zero after reset is already valid current-chunk output. There is no invalid pre-roll and no graph tail or zero-hop flush.

This checked-export build deliberately sends the unmodified finite input level into the graph. The former per-hop RMS boost was not part of the trained or frozen evaluation contract; deployment-path measurements showed that it modulated sub-bass and damaged drum and bass quality. Main and the residual use the same exact raw current hop.

The c193 graph contributes zero output-delay hops and the asynchronous collection/queue contributes one hop, so PDC remains exactly 512 samples without blocking the audio callback for inference.

The preserved bridge can map the interval onto other exact rational host clocks and include paired SRC delay in PDC, but those paths are disabled until separately qualified.

### Mixture-lossless output

Within the deployment graph, drums, bass, and vocals retain their estimates at every level, and `Other` receives the exact residual. The native buses reproduce those estimates unchanged whenever model output is available and confidence is fully open:

```text
Other = Main - Drums - Bass - Vocals
```

The plugin enforces the same invariant after runtime-provider differences, during dry fallback, and through its near-silence safety fade, so the four stem buses sum to the latency-aligned main bus to floating-point precision. This means the separation is mixture-lossless; it does not mean the estimated stems are identical to unrecoverable studio-original recordings. Independent clipping, normalization, or PCM quantization downstream can also break exact summation.

The model path has no external gain normalization, crossover, low-frequency reinjection, vocals-specific gate, or chunk-boundary crossfade. Finite audio goes directly to the causal current-chunk graph; there is no native overlap-add state.

The graph has a small, approximately level-independent floor in its individual stem estimates near silence. The final output stage therefore uses one stereo-linked peak envelope of the latency-aligned mixture to fade only the available model contribution. The envelope maps to fully enabled separation at and above -72 dBFS peak, fully disabled separation at and below -96 dBFS peak, and a smooth blend over the linear-amplitude interval between them. The detector opens immediately, holds peaks for 50 ms, then releases by 60 dB per 100 ms. Main and the dry underrun fallback are unchanged; as confidence falls, `Other` receives the residual.

Every processed hop is tagged with its exact output sample range. A result that arrives after its scheduled range has elapsed is discarded; it is never replayed against newer audio. On missing samples StemgenRT uses the complete latency-aligned dry split and routes the mixture to `Other`. Host reset notifications plus observed transport changes, seeks, scrubs, loop wraps, input gaps, and inference failures reset previous-audio, fusion-hidden, c130 history, previous-hidden, adapter-valid, raw-parent history, emitted-per-source history, output-crossfade, confidence, queue epoch, and SRC phase state. A play-to-stop transition observed in the callback first drains the final result already owed by the one-hop plugin PDC, then resets after that callback; this is queue-tail drainage, not a graph flush. The first successful result after reset is valid.

Real-time request publication and epoch reset use lock-free atomics only; they do not enter a condition-variable or OS wake path. The inference worker owns a bounded 100 microsecond idle polling backoff. The processor still reads the host transport snapshot once per callback to preserve exact start, stop, seek, scrub, and loop semantics; that host-provided call must be included in AU/VST target-host timing qualification.

### Model identity

The plugin bundles one self-contained file: `model/model.onnx`. There is no companion `.onnx.data` file. It is the c212 native-DFT export of candidate `c193-selected-drums7-bass16-over128-c191-step128`, SHA-256 `07557c7756815c0a84960c02faed4becd31413b53e1e4bb5b28177f2d6a97159`, size 114,645,969 bytes. Its sealed qualification receipt has SHA-256 `d5141507cc75c3bb0157982a4b17a750ed46a706c0bf42a6c796e2ff35d37d03`, and its c212 terminal `SEAL.json` has SHA-256 `e7cad784bcef80e3a2b426170c78687d41042958eeaca5d848f18c154fe5232f`. The receipt binds the c193 selected head/runtime state, c194 40-of-40 qualification, c197 all-pass electronic holdout, exact residual policy, and native ONNX Runtime 1.26 checks.

The graph has the eight-input/eight-output ABI above and seven explicit state tensors. For ONNX Runtime 1.26.0 compatibility, inverse real DFT reconstructs the full Hermitian spectrum before invoking native inverse `DFT`; no dense Fourier matrices are restored. The public fusion state remains opaque and is threaded at the export's `2^-18` scale. The authoritative artifact identity, tensor interface, streaming metadata, lineage, residual index, and current-hop alignment live in `cmake/QualifiedModelContract.cmake`. The contract ID is `c212-c193-ort126-checked-export-pending-native-plugin`.

The runtime validates the artifact, graph input/output contract, and embedded deployment metadata before enabling separation.

## CPU operation

CPU inference is the intended deployment path. Every worker wake, graph run, publication, and output write must fit inside the one-hop scheduling reserve. The native-DFT c212/c193 artifact has not yet completed target-Mac native-plugin timing qualification; measure the complete path under DAW load before promotion.

Use a modern CPU, close competing real-time workloads, and watch the plugin's underrun diagnostics when qualifying a system. The shipping runtime is deliberately CPU-only so host hardware cannot silently select a different numerical or scheduling path.

### Target-Mac candidate check

Run this gate on an Apple Silicon Mac before the first listening test. It uses the pinned official ONNX Runtime 1.26.0 build, checks the normal test suite and both plugin bundles, compares the 1–4-thread CPU choices, then paces 10,000 measured 512-sample callbacks after 100 warmups:

```bash
./scripts/download-onnxruntime.sh
cmake --preset release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build --preset release --target AudioPluginTest_BundleModel
ctest --preset release
cmake --build --preset release --target StemgenRT_VerifyMacBundles

build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep

build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenRealtimePaced
```

The paced check takes about two minutes and fails closed if the accepted model/runtime is unavailable. Do not continue unless it ends with `STEMGENRT_QUALIFICATION_SUMMARY status=pass`, `worker_priority=applied`, zero deadline misses, zero underruns, zero queue/ring drops, zero unsafe callbacks, finite and distinct retained stems, and reconstruction error at or below `1e-6`. `STEMGENRT_QUALIFICATION_CALLBACKS` may raise the measured callback count, but values below 10,000 are rejected.

After that automated gate passes, install both sealed Release bundles, force the Audio Unit rescan, validate the AU, and perform the same music listening test in the AU and VST3 hosts:

```bash
./scripts/install-plugins.sh --release
killall AudioComponentRegistrar || true
auval -v aufx Stem Swee
```

The synthetic soak is candidate-only evidence. Final promotion still requires the AU/VST listening result and complete-path behavior under representative DAW load; it does not manufacture paired-control miss-delta evidence.

To compare ONNX Runtime's CPU thread-pool size on a macOS Release build, run the disabled stateful sweep explicitly:

```bash
cmake --build --preset release --target AudioPluginTest_BundleModel
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep
```

The sweep creates a fresh session for each 1–4 thread candidate and runs three order-balanced passes of 25 warmup plus 500 measured hops. It reports wall-clock percentiles, aggregate process CPU time, deadline misses, and the retained-stem numerical delta from the single-thread reference. A previous Apple Silicon qualification selected three intra-op threads, so macOS sessions currently retain that cap; it is not c193 timing evidence. Windows retains the previous four-thread cap until the same target-platform qualification is completed there. Explicit benchmark overrides do not change either production policy.

The preserved multi-rate timing benchmark remains available for future qualification:

```bash
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuWithPreservedSampleRateBridges
```

Re-run the bridge and complete current-chunk path before enabling any non-native rate.

### Streaming diagnostics

The Release editor keeps the logo and adds a lightweight health panel showing model status, active PDC, the prepared and current host block sizes, dry-fallback samples, callback-timing warnings, and inference-worker priority. A `PDC timing warning` means the host is delivering a callback size that needs more latency than the value established in `prepareToPlay`. StemgenRT does not change PDC from the audio thread or splice late model fragments into the output; it keeps Main latency-aligned and routes the complete mixture to Other until callback timing is safe again. Stop playback and make the host re-prepare the plugin at 44.1 kHz / 512 samples (or reload the plugin) before judging separation. Persistent dry fallback at that exact configuration means inference is not completing within the one-hop reserve; reduce competing CPU load or use a faster target CPU.

## License

MIT
