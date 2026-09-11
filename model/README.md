# Streaming separation model

`model.onnx` is the self-contained stereo HS-TasNet model used by StemgenRT.
It separates **Drums, Bass, Vocals and Other** at **44.1 kHz**, using a 1024-sample
asymmetric analysis window, 256-sample synthesis frame and 128-sample hop.

The graph has **128 samples** of output delay. One asynchronous scheduling hop
makes the plugin's total **256 samples / 5.80 ms** with a 128-sample host buffer.
The graph source order differs from the plugin bus order: Main, Drums, Bass,
Other, Vocals.

## Artifact and state

The model is tracked with Git LFS and needs no external weights file.
SHA-256: `b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3`.
Size: **111,344,465 bytes**. Runtime: **ONNX Runtime 1.26.0 CPU**.

| Input | Shape | Output |
| --- | --- | --- |
| `audio_chunk` | `[1,2,128]` | `separated_chunk`: `[1,4,2,128]` |
| `audio_history` | `[1,2,896]` | `next_audio_history`: same shape |
| `fusion_hidden` | `[2,1,1000]` | `next_fusion_hidden`: same shape |
| `spectral_numerator_tail` | `[1,4,2,128]` | `next_spectral_numerator_tail`: same shape |
| `waveform_tail` | `[1,4,2,128]` | `next_waveform_tail`: same shape |

All tensors are float32. Initialize four states to zero, carry every returned
state unchanged, and discard the first output after reset. Each subsequent call
returns the preceding input hop. Pad a partial final hop once, submit exactly
one zero hop, then drain the plugin queue. Reset all state after a discontinuity.

The [CMake contract](../cmake/QualifiedModelContract.cmake) owns model identity,
geometry and metadata. Configuration verifies SHA/size; runtime loading checks
all five inputs, five outputs and embedded metadata. Its `QUALIFIED` variable
prefix denotes an identity lock, without guaranteeing deadline performance.

## Output behavior

Input levels and graph states remain in the original amplitude domain. The
runtime preserves all four graph outputs. The final writer applies linked
confidence and recovery fades to Drums/Bass/Vocals, then calculates Other from
Main minus those three outputs. Complete fallback routes Main to Other.

The confidence envelope opens immediately, holds for 50 ms, releases by
60 dB per 100 ms, and smoothly opens between -96 and -72 dBFS peak. It suppresses
unreliable estimates near silence while retaining the full mix.

## Validation

On a 14-track development panel, mean full-band SDR improves from **3.85 to
4.07 dB**. Additional intervals on those same tracks support the gain, and
controlled inputs show reduced vocal/instrument spill. Some passages and quiet
instruments still regress; these measurements are not an unseen-track benchmark.

Independent CPU float32 PyTorch fixtures verify all four stems and actual plugin
buses through eight clip lengths, including partial hops and final-sample
recovery. A continuous native comparison also passes the waveform and alignment
checks. The [fixture metadata](../test/fixtures/cropped1024-pytorch.json) retains
portable model identity and the binary format.

Correctness and packaging checks pass on macOS and Windows. User-reported Apple
M4 checks also passed (162 passed, one skipped), with **2 missed deadlines in
10,000 callbacks** and **16 in 30,000** in production-paced runs. Raw M4 reports
were not independently inspected. The strict zero-miss criterion remains unmet;
Linux paced timing failed too. The plugin retains aligned fallback for late
results, and timing should be measured under the intended DAW load.
