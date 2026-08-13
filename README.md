# StemgenRT

A real-time music source separation plugin. Drop it on a track and get 4 separate stems: drums, bass, other, and vocals.

This audition draft processes stereo audio in 256-sample hops and reports exactly 256 samples of latency (5.80 ms at 44.1 kHz) when the host is prepared at 44.1 kHz with a 256-sample callback. It is made for spatializing DJ sets in real time: split the mix into stems, place them in the room, and create an immersive experience.

> [!WARNING]
> This detached worktree contains the authenticated c236 256-hop audition candidate. The checked model and final receipt-bound contract are installed, but the candidate is still unpromoted: target-Mac build/soak and listening approval—especially kick/sub buzzing and Other leakage—remain mandatory. See `C236_AUDITION_DRAFT.md` for exact hashes and qualification commands.

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
> After receipt materialization, this candidate accepts exactly 44.1 kHz with a 256-sample prepared callback. Other configurations fail closed. The existing higher-rate bridge remains in the source for later qualification.

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

After completing every receipt/model substitution in `C236_AUDITION_DRAFT.md`, build with CMake presets:

```bash
cmake --preset default
cmake --build --preset default
```

For local listening on macOS, use the default build and install it explicitly as a debug build:

```bash
./scripts/install-plugins.sh --debug
```

The release preset writes to `build-release` and enables `STEMGENRT_REQUIRE_QUALIFIED_ORT`, which fails configuration unless the complete official 1.26.0 SDK is present. It is the required lane for the one remaining target-Mac paced gate; a successful build alone is not model promotion.

On macOS, install the sealed AU and VST3 bundles into your user plugin directories with:

```bash
./scripts/install-plugins.sh
```

The installer uses Release artifacts by default. Run `./scripts/install-plugins.sh --help` for signing and configuration options.

## How it works

The plugin runs a stateful HS-TasNet graph on a high-priority inference thread:

1. Callback N preserves the native mix for Main/fallback and submits one complete 256-sample request.
2. The inference thread carries 768 samples of causal analysis history, the fusion-GRU hidden state, and 2,048 samples of emitted Drums/Bass history. c236 emits the separated result aligned to input N.
3. The result is scheduled one queue hop later at host sample `(N + 1) * 256`, and Other is re-derived as the exact residual of the raw aligned mixture.

At 44.1 kHz both converters are bypassed with zero added delay and a bit-exact input copy. The preserved higher-rate bridge is disabled by this candidate contract until it is requalified.

The model graph has four inputs and four outputs:

| Direction | Tensor | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 256]` |
| Input | `analysis_history` | `[1, 2, 768]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Input | `emitted_db_history` | `[1, 4, 2048]` |
| Output | `separated_chunk` | `[1, 4, 2, 256]` |
| Output | `next_analysis_history` | `[1, 2, 768]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |
| Output | `next_emitted_db_history` | `[1, 4, 2048]` |

All three persistent state tensors—`analysis_history`, `fusion_hidden`, and `emitted_db_history`—start at positive float32 zero. `separated_chunk` is aligned with the current call's `audio_chunk`, and sequence zero after reset is already valid current-chunk output. There is no invalid pre-roll and no graph tail or zero-hop flush.

This checked-export build deliberately sends the unmodified finite input level into the graph. The former per-hop RMS boost was not part of the trained or frozen evaluation contract; deployment-path measurements showed that it modulated sub-bass and damaged drum and bass quality. Main and the residual use the same exact raw current hop.

The c236 graph contributes zero output-delay hops and the asynchronous collection/queue contributes one hop, so PDC remains exactly 256 samples without blocking the audio callback for inference.

The preserved bridge can map the interval onto other exact rational host clocks and include paired SRC delay in PDC, but those paths are disabled until separately qualified.

### Mixture-lossless output

Within the deployment graph, drums, bass, and vocals retain their estimates at every level, and `Other` receives the exact residual. The native buses reproduce those estimates unchanged whenever model output is available and confidence is fully open:

```text
Other = Main - Drums - Bass - Vocals
```

The plugin enforces the same invariant after runtime-provider differences, during dry fallback, and through its near-silence safety fade, so the four stem buses sum to the latency-aligned main bus to floating-point precision. This means the separation is mixture-lossless; it does not mean the estimated stems are identical to unrecoverable studio-original recordings. Independent clipping, normalization, or PCM quantization downstream can also break exact summation.

The model path has no external gain normalization, crossover, low-frequency reinjection, vocals-specific gate, or chunk-boundary crossfade. Finite audio goes directly to the causal current-chunk graph; there is no native overlap-add state.

The graph has a small, approximately level-independent floor in its individual stem estimates near silence. The final output stage therefore uses one stereo-linked peak envelope of the latency-aligned mixture to fade only the available model contribution. The envelope maps to fully enabled separation at and above -72 dBFS peak, fully disabled separation at and below -96 dBFS peak, and a smooth blend over the linear-amplitude interval between them. The detector opens immediately, holds peaks for 50 ms, then releases by 60 dB per 100 ms. Main and the dry underrun fallback are unchanged; as confidence falls, `Other` receives the residual.

Every processed hop is tagged with its exact output sample range. A result that arrives after its scheduled range has elapsed is discarded; it is never replayed against newer audio. On missing samples StemgenRT uses the complete latency-aligned dry split and routes the mixture to `Other`. Host reset notifications plus observed transport changes, seeks, scrubs, loop wraps, input gaps, and inference failures reset analysis history, fusion hidden state, emitted Drums/Bass history, output crossfade, confidence, queue epoch, and SRC phase state. A play-to-stop transition observed in the callback first drains the final result already owed by the one-hop plugin PDC, then resets after that callback; this is queue-tail drainage, not a graph flush. The first successful result after reset is valid.

Real-time request publication and epoch reset use lock-free atomics only; they do not enter a condition-variable or OS wake path. The inference worker owns a bounded 100 microsecond idle polling backoff. The processor still reads the host transport snapshot once per callback to preserve exact start, stop, seek, scrub, and loop semantics; that host-provided call must be included in AU/VST target-host timing qualification.

### Model identity

This audition tree bundles one self-contained c236 file: `model/model.onnx`, with no companion `.onnx.data` file. Its SHA-256 is `6dd58f05ee6bdf4beadb5df24849320e4f3dec4dbd0c7767fedd0dde2b557a02`. Configuration binds that payload to the authenticated terminal qualification and checked-export receipt through `cmake/QualifiedModelContract.cmake`; mismatched or stale payloads fail closed.

The graph has the four-input/four-output ABI above and three explicit state tensors. For ONNX Runtime 1.26.0 compatibility, inverse real DFT reconstructs the full Hermitian spectrum before invoking native inverse `DFT`; no dense Fourier matrices are restored. The authoritative artifact identity, tensor interface, terminal/materialization lineage, residual index, and current-hop alignment live in `cmake/QualifiedModelContract.cmake`. The draft contract ID is `c236-terminal-selected-hop256-audition-v1`.

The final handoff must use a passing schema-3
`hs_tasnet_c236_recovery_checked_onnx_export_v3` receipt, with the accepted
post-training qualification hash and candidate status bound into the ONNX
metadata. The independently recomputed export-receipt SHA is contract evidence
but is deliberately not ONNX metadata: the receipt already binds the model
hash, so embedding the receipt hash would be self-referential.

The terminal c236 selection, full post-training comparison, and checked ONNX receipt are complete and bound into this audition tree. Listening approval and the target-Mac soak remain required. Prior 512-hop evidence is fallback evidence only and is not transferred to this candidate.

The runtime validates the artifact, graph input/output contract, and embedded deployment metadata before enabling separation.

## CPU operation

CPU inference is the intended deployment path. Every worker wake, graph run, publication, and output write must fit inside the 5.80 ms one-hop scheduling reserve. This exact terminal c236 candidate still requires the target-Mac paced gate below before promotion.

Use a modern CPU, close competing real-time workloads, and watch the plugin's underrun diagnostics when qualifying a system. The shipping runtime is deliberately CPU-only so host hardware cannot silently select a different numerical or scheduling path.

### Target-Mac candidate check

The authenticated receipt/model substitution is complete in this audition tree. On an Apple Silicon Mac, build and verify the exact tree with the pinned official ONNX Runtime 1.26.0, install it for the listening test, then run both timing gates. Record the commit, model hash, receipt hash, and complete test output:

```bash
./scripts/download-onnxruntime.sh
git lfs pull --include=model/model.onnx
git rev-parse HEAD
shasum -a 256 model/model.onnx
cmake --preset release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build --preset release --target StemgenRT_VerifyMacBundles
cmake --build --preset release --target AudioPluginTest_BundleModel
ctest --preset release

build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep

STEMGENRT_QUALIFICATION_CALLBACKS=10000 \
  build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenRealtimePaced

./scripts/install-plugins.sh --release
```

The model hash must be
`6dd58f05ee6bdf4beadb5df24849320e4f3dec4dbd0c7767fedd0dde2b557a02`.
Also record the independently authenticated external checked-export receipt
hash `a00f2a6c764be9cfb902b871607f40b93b18ed81e39fb1cd10848fe3ded36917`;
the receipt is deliberately not packaged as a plugin resource.

The paced check takes about one minute and fails closed if the accepted model/runtime is unavailable. Do not continue unless it ends with `STEMGENRT_QUALIFICATION_SUMMARY status=pass`, `callback_samples=256`, `sample_rate=44100`, a deadline of approximately `5804.989` microseconds, `worker_priority=applied`, `finite_outputs=1`, zero deadline misses, zero underruns, zero queue/ring drops, zero unsafe callbacks, finite and distinct retained stems, and reconstruction error at or below `1e-6`. Its p99.9 and maximum callback times must also remain below the deadline. `STEMGENRT_QUALIFICATION_CALLBACKS` may raise the measured callback count, but values below 10,000 are rejected.

The existing two-thread macOS automatic cap is preserved; Windows retains four. The c236 graph still needs listening approval and exact-hash target-Mac evidence, so neither prior listening nor prior 512-hop timing is transferred.

The preserved multi-rate timing benchmark remains available for future qualification:

```bash
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuWithPreservedSampleRateBridges
```

Re-run the bridge and complete current-chunk path before enabling any non-native rate.

### Streaming diagnostics

The Release editor keeps the logo and adds a lightweight health panel showing model status, active PDC, the prepared and current host block sizes, dry-fallback samples, callback-timing warnings, and inference-worker priority. A `PDC timing warning` means the host is delivering a callback size that needs more latency than the value established in `prepareToPlay`. StemgenRT does not change PDC from the audio thread or splice late model fragments into the output; it keeps Main latency-aligned and routes the complete mixture to Other until callback timing is safe again. Stop playback and make the host re-prepare the plugin at 44.1 kHz / 256 samples (or reload the plugin) before judging separation. Persistent dry fallback at that exact configuration means inference is not completing within the one-hop reserve; reduce competing CPU load or use a faster target CPU.

## License

MIT
