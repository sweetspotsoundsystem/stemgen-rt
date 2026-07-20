# StemgenRT

A real-time music source separation plugin. Drop it on a track and get 4 separate stems: drums, bass, other, and vocals.

StemgenRT processes stereo audio in 512-sample hops and reports an honest 1,024 samples of latency (23.22 ms at 44.1 kHz) to the host. It is made for spatializing DJ sets in real time: split the mix into stems, place them in the room, and create an immersive experience.

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
> Set your DAW to 44.1 kHz. The qualified model accepts raw fullband 44.1 kHz stereo audio only; at any other sample rate the plugin fails closed and does not start separation.

## Downloads

- [StemgenRT-macOS-AU.zip](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/download/latest/StemgenRT-macOS-AU.zip)
- [StemgenRT-macOS-VST3.zip](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/download/latest/StemgenRT-macOS-VST3.zip)
- [StemgenRT-Windows-VST3.zip](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/download/latest/StemgenRT-Windows-VST3.zip)

> [!NOTE]
> The macOS plugin is not signed (yet). You need to sign it yourself: `codesign --force --deep --sign - StemgenRT.component`

## Building

First, grab the ONNX Runtime dependency:

```bash
# macOS
./scripts/download-onnxruntime.sh

# Windows (PowerShell)
./scripts/download-onnxruntime.ps1
```

Then build with CMake:

```bash
cmake -S . -B build
cmake --build build
```

For a release build:

```bash
cmake -S . -B build-release
cmake --build build-release
```

## How it works

The plugin runs a stateful HS-TasNet graph on a background inference thread so the audio callback never waits for the model:

1. The audio thread collects unmodified, fullband stereo audio into 512-sample hops.
2. The inference thread runs one hop while carrying the model's previous audio, overlap-add, and fusion-GRU state.
3. The graph emits the separation for the previous hop, and the audio thread returns it through the stem buses.

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

There are two 512-sample stages in the native plugin: one hop to collect and queue audio and one hop of model output delay. StemgenRT therefore reports 1,024 samples—not 512—to the host for plugin delay compensation. The main bus is delayed by the same amount.

### Mixture-lossless output

The deployment graph preserves the mixture in the floating-point domain. Drums, bass, and vocals keep their model estimates, and `Other` receives the exact residual:

```text
Other = Main - Drums - Bass - Vocals
```

The plugin enforces the same invariant after runtime-provider differences and during dry fallback, so the four stem buses sum to the latency-aligned main bus to floating-point precision. This means the separation is mixture-lossless; it does not mean the estimated stems are identical to unrecoverable studio-original recordings. Independent clipping, normalization, or PCM quantization downstream can also break exact summation.

The qualified path intentionally has no external crossover, input normalization, low-frequency reinjection, vocals gate, or chunk-boundary crossfade. Fullband audio goes directly to the stateful graph, which owns its overlap-add processing.

On an inference underrun, StemgenRT crossfades to a latency-aligned dry split and routes the final residual to `Other`, avoiding a glitch while preserving the stem-sum invariant. Transport starts, seeks, scrubs, loop wraps, inference failures, and dropped input hops reset all recurrent and overlap-add state. The first result after each reset is pre-roll and is not presented as separated audio.

### Model identity

The plugin bundles one self-contained file: `model/model.onnx`. There is no companion `.onnx.data` file.

```text
SHA-256  52fdc46d015819821dae19ef272b6bc4ccf441a0274d7d9c8bb44af50eafef8c
```

The runtime validates the graph's input/output contract and embedded deployment metadata before enabling separation.

## CPU operation

CPU inference is the qualified and supported path. Every 512-sample inference must finish within the 11.61 ms hop interval even though the host-visible PDC is 23.22 ms. A 500-hop native Windows qualification with ONNX Runtime 1.26.0 measured 3.69 ms mean, 4.10 ms p95, 4.42 ms p99, 5.83 ms maximum, and zero deadline misses on the research machine. Treat those figures as a smoke result, not a guarantee for every processor or DAW workload.

Use a modern CPU, close competing real-time workloads, and watch the plugin's underrun diagnostics when qualifying a system. The shipping runtime is deliberately CPU-only so host hardware cannot silently select a different numerical or scheduling path.

## License

MIT
