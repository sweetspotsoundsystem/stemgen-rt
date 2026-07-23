# StemgenRT

A real-time music source separation plugin. Drop it on a track and get 4 separate stems: drums, bass, other, and vocals.

This c126 listening build processes stereo audio in 512-sample hops and reports exactly 512 samples of latency (11.61 ms at 44.1 kHz) when the host is prepared at 44.1 kHz with a 512-sample callback. It is made for spatializing DJ sets in real time: split the mix into stems, place them in the room, and create an immersive experience.

> [!WARNING]
> The bundled c126 step-100k model is an unqualified listening candidate. Its aggregate validation quality is better than c91, but three frozen per-stem quality guards remain unresolved and its no-overlap current-chunk synthesis has a measured low-frequency hop seam. Test and compare it by ear; do not publish it as a qualified release yet.

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
> This listening build accepts exactly 44.1 kHz with a 512-sample prepared callback. Other configurations fail closed. The existing higher-rate bridge remains in the source for later c126 qualification.

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

The c126 model is not yet qualified for a public release. For local listening on macOS, use the default build and install it explicitly as a debug build:

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

The plugin runs a stateful HS-TasNet graph on a background inference thread so the audio callback never waits for the model:

1. The audio thread preserves the native host-rate mix for Main and the dry fallback and collects one 512-sample request.
2. The inference thread runs one current-chunk hop at the input's native floating-point level while carrying previous-audio and fusion-GRU state.
3. The audio thread publishes the result on the exact sequence-N timeline and derives Other as the exact residual of raw current-chunk Main.

At 44.1 kHz both converters are bypassed with zero added delay and a bit-exact input copy. The preserved higher-rate bridge is disabled by this listening contract until it is requalified with c126.

The model graph has three inputs and three outputs:

| Direction | Tensor | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 512]` |
| Input | `past_audio` | `[1, 2, 512]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Output | `separated_chunk` | `[1, 4, 2, 512]` |
| Output | `next_past_audio` | `[1, 2, 512]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |

Both persistent state tensors start at zero. `separated_chunk` is aligned with the same call's `audio_chunk`, including sequence zero after a reset. There is no boundary overlap-add tensor, pre-roll result, or graph flush hop.

The model is observably sensitive to input level: lowering the same material can change which stem receives it. This listening build deliberately sends the unmodified finite input level into the graph. The former per-hop RMS boost was not part of the trained or frozen evaluation contract; deployment-path measurements showed that it modulated sub-bass and materially reduced c126 drum and bass quality. Main and the residual use the same exact raw current hop.

There is one host-visible 512-sample stage: collect and queue the current request. The causal graph adds zero output-delay hops. With a fixed 512-sample callback, the worker receives one complete 11.61 ms callback interval before sequence N is consumed at sequence N+1, so PDC is exactly 512 samples.

The preserved bridge can map the one-hop interval onto other exact rational host clocks and include paired SRC delay in PDC, but those paths are disabled until separately qualified with c126.

### Mixture-lossless output

Within the deployment graph, drums, bass, and vocals retain their estimates at every level, and `Other` receives the exact residual. The native buses reproduce those estimates unchanged whenever model output is available and confidence is fully open:

```text
Other = Main - Drums - Bass - Vocals
```

The plugin enforces the same invariant after runtime-provider differences, during dry fallback, and through its near-silence safety fade, so the four stem buses sum to the latency-aligned main bus to floating-point precision. This means the separation is mixture-lossless; it does not mean the estimated stems are identical to unrecoverable studio-original recordings. Independent clipping, normalization, or PCM quantization downstream can also break exact summation.

The model path has no external gain normalization, crossover, low-frequency reinjection, vocals-specific gate, chunk-boundary crossfade, or boundary overlap-add tensor. Finite audio goes directly to the stateful graph, whose causal analysis/synthesis is internal.

The graph has a small, approximately level-independent floor in its individual stem estimates near silence. The final output stage therefore uses one stereo-linked peak envelope of the latency-aligned mixture to fade only the available model contribution. The envelope maps to fully enabled separation at and above -72 dBFS peak, fully disabled separation at and below -96 dBFS peak, and a smooth blend over the linear-amplitude interval between them. The detector opens immediately, holds peaks for 50 ms, then releases by 60 dB per 100 ms. Main and the dry underrun fallback are unchanged; as confidence falls, `Other` receives the residual.

Every processed hop is tagged with its exact output sample range. A result that misses part or all of that range is trimmed or discarded, never replayed later against newer audio. On those missing samples StemgenRT uses the complete latency-aligned dry split immediately, then fades exact-timeline model output back in and routes the final residual to `Other`. Transport starts, seeks, stopped scrubs, and loop wraps reset previous-audio, fusion-hidden, output-crossfade, confidence, queue epoch, and SRC phase state. The first successful result after reset is valid sequence-N current-chunk output.

### Model identity

The plugin bundles one self-contained file: `model/model.onnx`. There is no companion `.onnx.data` file. The authoritative artifact identity, checkpoint identity, tensor interface, streaming metadata, dimensions, residual index, output alignment, and explicit `unqualified_listening_only` status live in `cmake/QualifiedModelContract.cmake`.

The runtime validates the artifact, graph input/output contract, and embedded deployment metadata before enabling separation.

## CPU operation

CPU inference is the intended deployment path. Every 512-sample inference must finish within the 11.61 ms hop interval. Previous c91 timing evidence does not qualify c126, so watch the plugin's underrun diagnostics and repeat the native macOS timing sweep before promotion.

Use a modern CPU, close competing real-time workloads, and watch the plugin's underrun diagnostics when qualifying a system. The shipping runtime is deliberately CPU-only so host hardware cannot silently select a different numerical or scheduling path.

To compare ONNX Runtime's CPU thread-pool size on a macOS Release build, run the disabled stateful sweep explicitly:

```bash
cmake --build --preset release --target AudioPluginTest
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep
```

The sweep creates a fresh session for each 1–4 thread candidate and runs three order-balanced passes of 25 warmup plus 500 measured hops. It reports wall-clock percentiles, aggregate process CPU time, deadline misses, and the retained-stem numerical delta from the single-thread reference. The qualified Apple Silicon run selected three intra-op threads, so macOS production sessions cap the automatic policy at three. Windows retains the previous four-thread cap until the same target-platform qualification is completed there. Explicit benchmark overrides do not change either production policy.

The preserved multi-rate timing benchmark remains available for future c126 qualification:

```bash
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuWithPreservedSampleRateBridges
```

Historical c91 measurements are useful only as bridge evidence. Re-run it with c126 before enabling any non-native rate.

### Streaming diagnostics

The Release editor keeps the logo and adds a lightweight health panel showing model status, active PDC, the prepared and current host block sizes, dry-fallback samples, callback-timing warnings, and inference-worker priority. A `PDC timing warning` means the host is delivering a callback size that needs more latency than the value established in `prepareToPlay`. StemgenRT does not change PDC from the audio thread or splice late model fragments into the output; it keeps Main latency-aligned and routes the complete mixture to Other until callback timing is safe again. Stop playback and make the host re-prepare the plugin at its current audio-buffer size (or reload the plugin) before judging separation. Persistent dry fallback with safe timing indicates inference is missing its recurring 11.61 ms deadline; reduce competing CPU load or use a larger host buffer.

## License

MIT
