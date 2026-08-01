# StemgenRT

A real-time music source separation plugin. Drop it on a track and get 4 separate stems: drums, bass, other, and vocals.

This native-DFT c166i listening build processes stereo audio in 512-sample hops and reports exactly 512 samples of latency (11.61 ms at 44.1 kHz) when the host is prepared at 44.1 kHz with a 512-sample callback. It is made for spatializing DJ sets in real time: split the mix into stems, place them in the room, and create an immersive experience.

> [!WARNING]
> The bundled model keeps the c166i L13/g31-over-32 weights and replaces the former dense real-DFT export lowering with native ONNX `DFT` operators. This directly addresses the export error implicated in the audible sub-bass distortion without changing the trained model. Its full validation score remains 4.7146 dB, 0.1668 dB above c91, but it passes only 37 of 40 model guardrails: Bass SIR, isolated-Bass SDR, and isolated-Other gain still need repair. It is intentionally labelled an unqualified listening candidate. Test its sound, especially Bass/Other low-frequency behavior, and complete-path timing on the target Mac before treating it as a release.

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
> This listening build accepts exactly 44.1 kHz with a 512-sample prepared callback. Other configurations fail closed. The existing higher-rate bridge remains in the source for later qualification.

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

The one-hop current-chunk scheduler is not yet qualified for a public release. For local listening on macOS, use the default build and install it explicitly as a debug build:

```bash
./scripts/install-plugins.sh --debug
```

The release preset remains available for later qualification work and writes to `build-release`. It enables `STEMGENRT_REQUIRE_QUALIFIED_ORT`, which fails configuration unless the complete official 1.26.0 SDK is present. Do not treat a successful release build as model promotion.

On macOS, install the sealed AU and VST3 bundles into your user plugin directories with:

```bash
./scripts/install-plugins.sh
```

The installer uses Release artifacts by default, so this listening candidate must be selected with `./scripts/install-plugins.sh --debug`. Run `./scripts/install-plugins.sh --help` for signing and configuration options.

## How it works

The plugin runs a stateful HS-TasNet graph on a high-priority inference thread:

1. Callback N preserves the native mix for Main/fallback and submits one complete 512-sample request.
2. The inference thread carries previous-audio, fusion-GRU, c130 feature-history, c157 hidden-history, adapter-valid, and 2,048-sample raw-parent history state. c166 emits the separated result aligned to input N.
3. The result is scheduled one queue hop later at host sample `(N + 1) * 512`, and Other is re-derived as the exact residual of the raw aligned mixture.

At 44.1 kHz both converters are bypassed with zero added delay and a bit-exact input copy. The preserved higher-rate bridge is disabled by this listening contract until it is requalified.

The model graph has seven inputs and seven outputs:

| Direction | Tensor | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 512]` |
| Input | `past_audio` | `[1, 2, 512]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Input | `c130_history` | `[1, 20, 128]` |
| Input | `previous_hidden` | `[1, 32, 512]` |
| Input | `adapter_valid` | `[1, 1]` |
| Input | `raw_parent_history` | `[1, 4, 2048]` |
| Output | `separated_chunk` | `[1, 4, 2, 512]` |
| Output | `next_past_audio` | `[1, 2, 512]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |
| Output | `next_c130_history` | `[1, 20, 128]` |
| Output | `next_previous_hidden` | `[1, 32, 512]` |
| Output | `next_adapter_valid` | `[1, 1]` |
| Output | `next_raw_parent_history` | `[1, 4, 2048]` |

All six persistent state tensors start at zero. `separated_chunk` is aligned with the current call's `audio_chunk`. Sequence zero after reset is already a valid exact-c157 output while the raw-parent history initializes. There is no invalid pre-roll and no graph tail or zero-hop flush.

This listening build deliberately sends the unmodified finite input level into the graph. The former per-hop RMS boost was not part of the trained or frozen evaluation contract; deployment-path measurements showed that it modulated sub-bass and damaged drum and bass quality. Main and the residual use the same exact raw current hop.

The c166 graph contributes zero output-delay hops and the asynchronous collection/queue contributes one hop, so PDC remains exactly 512 samples without blocking the audio callback for inference.

The preserved bridge can map the interval onto other exact rational host clocks and include paired SRC delay in PDC, but those paths are disabled until separately qualified.

### Mixture-lossless output

Within the deployment graph, drums, bass, and vocals retain their estimates at every level, and `Other` receives the exact residual. The native buses reproduce those estimates unchanged whenever model output is available and confidence is fully open:

```text
Other = Main - Drums - Bass - Vocals
```

The plugin enforces the same invariant after runtime-provider differences, during dry fallback, and through its near-silence safety fade, so the four stem buses sum to the latency-aligned main bus to floating-point precision. This means the separation is mixture-lossless; it does not mean the estimated stems are identical to unrecoverable studio-original recordings. Independent clipping, normalization, or PCM quantization downstream can also break exact summation.

The model path has no external gain normalization, crossover, low-frequency reinjection, vocals-specific gate, or chunk-boundary crossfade. Finite audio goes directly to the causal current-chunk graph; there is no native overlap-add state.

The graph has a small, approximately level-independent floor in its individual stem estimates near silence. The final output stage therefore uses one stereo-linked peak envelope of the latency-aligned mixture to fade only the available model contribution. The envelope maps to fully enabled separation at and above -72 dBFS peak, fully disabled separation at and below -96 dBFS peak, and a smooth blend over the linear-amplitude interval between them. The detector opens immediately, holds peaks for 50 ms, then releases by 60 dB per 100 ms. Main and the dry underrun fallback are unchanged; as confidence falls, `Other` receives the residual.

Every processed hop is tagged with its exact output sample range. A result that arrives after its scheduled range has elapsed is discarded; it is never replayed against newer audio. On missing samples StemgenRT uses the complete latency-aligned dry split and routes the mixture to `Other`. Transport changes, seeks, scrubs, loop wraps, input gaps, and inference failures reset previous-audio, fusion-hidden, c130 history, c157 hidden history, adapter-valid, raw-parent history, output-crossfade, confidence, queue epoch, and SRC phase state. A play-to-stop transition first drains the final result already owed by the one-hop plugin PDC, then resets after that callback; this is queue-tail drainage, not a graph flush. The first successful result after reset is valid.

### Model identity

The plugin bundles one self-contained file: `model/model.onnx`. There is no companion `.onnx.data` file. It is the native-DFT c166i streaming artifact (SHA-256 `31a280e628f632d052d73828783f5ad974f0be6c7db18bd6233157153a781f02`, 114,526,643 bytes) from deploy artifact SHA-256 `c1d75192192112122d30e5d94aad6b96e97eed466103914f3f7803b3bf06a173`. Its refiner state SHA-256 is `fdc71a7bbe4753343401c31ac92c176a48d9329550ed59c3b3ccbf77eb21ae1d`; the c157 step-6 parent remains bound by its checkpoint and head-state identities. The seven-input/seven-output ABI is unchanged. For compatibility with the bundled ONNX Runtime 1.26.0, inverse real DFT reconstructs the full Hermitian spectrum before invoking native inverse `DFT`; no dense Fourier matrices are restored. The public fusion state is an opaque state threaded at a `2^-18` scale so its ONNX round trip stays within the long-horizon numerical bound. The authoritative artifact identity, tensor interface, streaming metadata, dimensions, residual index, and current-hop alignment live in `cmake/QualifiedModelContract.cmake`.

The runtime validates the artifact, graph input/output contract, and embedded deployment metadata before enabling separation.

## CPU operation

CPU inference is the intended deployment path. Every worker wake, graph run, publication, and output write must fit inside the one-hop scheduling reserve. The native-DFT c166i artifact has not yet completed target-Mac native timing qualification; measure the complete path under DAW load before promotion.

Use a modern CPU, close competing real-time workloads, and watch the plugin's underrun diagnostics when qualifying a system. The shipping runtime is deliberately CPU-only so host hardware cannot silently select a different numerical or scheduling path.

To compare ONNX Runtime's CPU thread-pool size on a macOS Release build, run the disabled stateful sweep explicitly:

```bash
cmake --build --preset release --target AudioPluginTest
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep
```

The sweep creates a fresh session for each 1–4 thread candidate and runs three order-balanced passes of 25 warmup plus 500 measured hops. It reports wall-clock percentiles, aggregate process CPU time, deadline misses, and the retained-stem numerical delta from the single-thread reference. The qualified Apple Silicon run selected three intra-op threads, so macOS production sessions cap the automatic policy at three. Windows retains the previous four-thread cap until the same target-platform qualification is completed there. Explicit benchmark overrides do not change either production policy.

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
