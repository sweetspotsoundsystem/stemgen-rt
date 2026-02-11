# StemgenRT

A real-time low-latency music source separation plugin. Drop it on a track and get 4 separate stems: drums, bass, other, and vocals.

With a latency of 11.6 milliseconds, it is made for spatializing DJ sets in real-time: split the mix into stems, place them in the room, and create an immersive experience.

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

> [!NOTE]
> Set your DAW to 44.1 kHz, the model only works at this sample rate.

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

The plugin runs a neural network to separate audio, but neural networks are slow and audio callbacks are fast. To bridge the gap:

1. **Audio thread** collects incoming samples and feeds them to a ring buffer
2. **Inference thread** runs the model asynchronously in the background
3. **Audio thread** picks up the processed stems when ready

If inference can't keep up, the plugin gracefully crossfades to the dry signal rather than glitching.

A few DSP tricks help the model out:

- **HP/LP split + LP reinjection** — Input is split by LR4 crossover. HP goes to the model; LP bypasses inference and is reinjected after model output (currently bass-biased) to keep low end stable.
- **Chunk boundary crossfade** — The model outputs more samples than the 512-sample center region. Extra samples from the right context are crossfaded with the next chunk's start, eliminating discontinuities at chunk boundaries.
- **Input normalization** — Context-aware: RMS is computed over both the context window and the current input chunk, then normalized to a consistent level before inference. This avoids extreme gain swings at transients (e.g., loud kick tail in context, silence in input) and pushes the model's noise floor below the signal level.
- **Vocals gate** — Detects spurious low-level content in the vocals stem (common on instrumentals) using both energy ratio and absolute level thresholds. Gated content is transferred to the "other" stem to preserve total energy. Asymmetric attack/release smoothing prevents pumping.
- **Soft gating** — When input is silent, output is silent. Prevents the model from hallucinating noise.
- **Low-band stabilizer** — Reconstructs low-band stem balance using dry-constrained low-frequency energy and suppresses synthetic high-frequency leakage on low-only inputs.

The main bus is dry passthrough. The stem buses carry model output with LP reinjection, low-band stabilization, and gates applied.

## A note on GPU acceleration

You might expect GPU to be faster, but for this particular model it often isn't:

- **1D convolutions** — GPUs are optimized for 2D (images). 1D audio convolutions don't parallelize as well.
- **Batch size of 1** — Real-time audio processes one chunk at a time. GPUs shine with large batches.
- **Memory-bound ops** — Reshapes and audio operations are limited by memory bandwidth, not compute. Your CPU cache is actually fast for this.
- **Kernel launch overhead** — Each GPU operation has ~5-20μs overhead. With many small ops, it adds up.

That said, GPU builds are available if you want to try.

## License

MIT
