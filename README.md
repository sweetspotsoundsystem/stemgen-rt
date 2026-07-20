# StemgenRT

A real-time music source separation plugin. Drop it on a track and get 4 separate stems: drums, bass, other, and vocals.

StemgenRT processes stereo audio in 512-sample hops and reports an honest minimum of 1,024 samples of latency (23.22 ms at 44.1 kHz) to the host. The reported delay adapts to the host callback size so asynchronous results stay on their exact sample timeline with a full inference interval. It is made for spatializing DJ sets in real time: split the mix into stems, place them in the room, and create an immersive experience.

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
> StemgenRT is qualified at 44.1, 48, 88.2, 96, 176.4, and 192 kHz. Other host rates fail closed and do not start separation.

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

For a qualified release build:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

The release preset writes to `build-release` and enables `STEMGENRT_REQUIRE_QUALIFIED_ORT`, which fails configuration unless the complete official 1.26.0 SDK is present. On Windows that means the headers, `onnxruntime.lib`, and the explicitly loaded sibling `onnxruntime.dll`; a fallback-only plugin cannot be published accidentally. Debug builds allow another locally installed ORT for development, but CMake marks it as unqualified.

On macOS, install the sealed AU and VST3 bundles into your user plugin directories with:

```bash
./scripts/install-plugins.sh
```

The installer uses the qualified Release artifacts by default. `--release` remains available when an explicit configuration is useful, while local development builds must be selected with `./scripts/install-plugins.sh --debug`. Run `./scripts/install-plugins.sh --help` for signing and configuration options.

## How it works

The plugin runs a stateful HS-TasNet graph on a background inference thread so the audio callback never waits for the model:

1. The audio thread preserves the native host-rate mix for Main and the dry fallback. At qualified rates above 44.1 kHz, a stateful band-limited converter also supplies the graph's fixed 44.1 kHz clock.
2. The inference thread applies stereo-linked model-input gain staging and runs one 512-sample model hop while carrying the model's previous audio, overlap-add, and fusion-GRU state.
3. The worker restores the graph's original gain and converts drums, bass, and vocals back to the host clock. The audio thread derives Other as the exact residual of the native-rate Main mix.

At 44.1 kHz both converters are bypassed with zero added delay and a bit-exact input copy. At higher qualified rates, the streaming linear-phase Kaiser-windowed sinc converters are designed for a 20 kHz passband and at least 110 dB rejection at the model Nyquist limit. Main never makes a 44.1 kHz round trip: content above 22.05 kHz remains in the native Main timeline and therefore in the host-domain Other residual instead of being aliased into a named stem.

The model graph has four inputs and four outputs:

| Direction | Tensor | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 512]` |
| Input | `past_audio` | `[1, 2, 512]` |
| Input | `overlap_add_buffer` | `[1, 4, 2, 1024]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Output | `separated_chunk` | `[1, 4, 2, 512]` |
| Output | `next_past_audio` | `[1, 2, 512]` |
| Output | `next_overlap_add_buffer` | `[1, 4, 2, 1024]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |

All persistent state starts at zero. Because `separated_chunk` is aligned with `past_audio`, the first graph result is pre-roll and is discarded. The next run emits the first real input hop. A final zero audio hop flushes the last real hop during a render.

The model is observably sensitive to input level: lowering the same material can change which stem receives it. Based on the former deployment wrapper and listening tests, the inference worker measures one stereo RMS and one stereo-linked peak over the graph's exact raw `[past hop, current hop]` analysis window. Its boost is the minimum of the gain needed to reach -12 dBFS RMS, the headroom available below the 0 dBFS peak ceiling, and the +40 dB maximum boost. The result is clamped to a minimum gain of one, so input already at or above the RMS target or peak ceiling is never attenuated. If the gain changes, the runtime moves `past_audio` and `overlap_add_buffer` into the new amplitude domain while leaving the nonlinear fusion-GRU state untouched. It then divides the separated output by the same gain and uses the untouched raw past hop for Main and the residual. An exact-zero flush hop holds the preceding gain. This adds no lookahead or host latency.

There are two 512-sample model stages in the native plugin: one hop to collect and queue audio and one hop of model output delay. At 44.1 kHz the minimum plugin delay is therefore 1,024 samples—not 512. The inference worker must also receive a full 512-sample compute interval after the callback that completes a model hop. Because results are consumed only at host callback boundaries, 44.1 kHz preparation adds `ceil(512 / blockSize) * blockSize - gcd(blockSize, 512)` samples. For example, 64-, 256-, 512-, and 1,024-sample callbacks report 1,472, 1,280, 1,024, and 1,536 samples respectively.

At higher qualified rates, StemgenRT maps the same two-hop interval onto the host's exact rational clock and includes the paired sample-rate converters' group delay in PDC. A fractional output-filter correction makes that total an exact integer number of host samples. Main, model output, and fallback all use that same reported timeline; the plugin never hides conversion delay or rounds a late result onto a newer sample.

### Mixture-lossless output

Within the deployment graph, drums, bass, and vocals retain their estimates at every level, and `Other` receives the exact residual. The native buses reproduce those estimates unchanged whenever model output is available and confidence is fully open:

```text
Other = Main - Drums - Bass - Vocals
```

The plugin enforces the same invariant after runtime-provider differences, during dry fallback, and through its near-silence safety fade, so the four stem buses sum to the latency-aligned main bus to floating-point precision. This means the separation is mixture-lossless; it does not mean the estimated stems are identical to unrecoverable studio-original recordings. Independent clipping, normalization, or PCM quantization downstream can also break exact summation.

The model path has no external crossover, low-frequency reinjection, vocals-specific gate, or chunk-boundary crossfade. Apart from the state-aware fullband gain staging above, the audio goes directly to the stateful graph, which owns its analysis and overlap-add processing.

The graph has a small, approximately level-independent floor in its individual stem estimates near silence. The final output stage therefore uses one stereo-linked peak envelope of the latency-aligned mixture to fade only the available model contribution. The envelope maps to fully enabled separation at and above -72 dBFS peak, fully disabled separation at and below -96 dBFS peak, and a smooth blend over the linear-amplitude interval between them. The detector opens immediately, holds peaks for 50 ms, then releases by 60 dB per 100 ms. Main and the dry underrun fallback are unchanged; as confidence falls, `Other` receives the residual.

Every processed hop is tagged with its exact output sample range. A result that misses part or all of that range is trimmed or discarded, never replayed later against newer audio. On those missing samples StemgenRT uses the complete latency-aligned dry split immediately, then fades exact-timeline model output back in and routes the final residual to `Other`. Transport starts, seeks, stopped scrubs, and loop wraps reset the recurrent, overlap-add, output-crossfade, and confidence-envelope state; inference failures, non-finite input, and dropped input hops reset the model state. The first result after each model reset is pre-roll and is not presented as separated audio.

### Model identity

The plugin bundles one self-contained file: `model/model.onnx`. There is no companion `.onnx.data` file. The authoritative artifact identity, checkpoint identity, tensor interface, streaming metadata, dimensions, residual index, and output alignment live in `cmake/QualifiedModelContract.cmake`. Configuration generates the C++ contract used by the runtime and tests from that file, while source-artifact validation and Apple bundle sealing consume it directly. Replacing the model therefore begins with one contract update followed by a full qualification pass.

The runtime validates the artifact, graph input/output contract, and embedded deployment metadata before enabling separation.

## CPU operation

CPU inference is the qualified and supported path. Every 512-sample inference must finish within the 11.61 ms hop interval even though the minimum host-visible PDC is 23.22 ms. A 500-hop native Windows qualification with ONNX Runtime 1.26.0 measured 3.69 ms mean, 4.10 ms p95, 4.42 ms p99, 5.83 ms maximum, and zero deadline misses on the research machine. Treat those figures as a smoke result, not a guarantee for every processor or DAW workload.

Use a modern CPU, close competing real-time workloads, and watch the plugin's underrun diagnostics when qualifying a system. The shipping runtime is deliberately CPU-only so host hardware cannot silently select a different numerical or scheduling path.

To compare ONNX Runtime's CPU thread-pool size on a macOS Release build, run the disabled stateful sweep explicitly:

```bash
cmake --build --preset release --target AudioPluginTest
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep
```

The sweep creates a fresh session for each 1–4 thread candidate and runs three order-balanced passes of 25 warmup plus 500 measured hops. It reports wall-clock percentiles, aggregate process CPU time, deadline misses, and the retained-stem numerical delta from the single-thread reference. The qualified Apple Silicon run selected three intra-op threads, so macOS production sessions cap the automatic policy at three. Windows retains the previous four-thread cap until the same target-platform qualification is completed there. Explicit benchmark overrides do not change either production policy.

The multi-rate timing benchmark includes stereo input conversion, one stateful ONNX hop, and six-channel retained-stem output conversion at 48, 96, and 192 kHz:

```bash
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuWithQualifiedSampleRateBridges
```

On the Apple Silicon development machine, its 500-hop Release run measured p95 values of 3.06 ms at 48 kHz, 3.34 ms at 96 kHz, and 3.96 ms at 192 kHz, with zero 11.61 ms deadline misses at every rate. Re-run it on every release target; these figures are not a guarantee for another CPU or host workload.

### Streaming diagnostics

The Release editor keeps the logo and adds a lightweight health panel showing model status, active PDC, the prepared and current host block sizes, dry-fallback samples, callback-timing warnings, and inference-worker priority. A `PDC timing warning` means the host is delivering a callback size that needs more latency than the value established in `prepareToPlay`. StemgenRT does not change PDC from the audio thread or splice late model fragments into the output; it keeps Main latency-aligned and routes the complete mixture to Other until callback timing is safe again. Stop playback and make the host re-prepare the plugin at its current audio-buffer size (or reload the plugin) before judging separation. Persistent dry fallback with safe timing indicates inference is missing its recurring 11.61 ms deadline; reduce competing CPU load or use a larger host buffer.

## License

MIT
