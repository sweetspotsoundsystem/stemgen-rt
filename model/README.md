# Streaming separation model

`model.onnx` is the self-contained stereo HS-TasNet graph used by StemgenRT.
It separates **Drums, Bass, Vocals and Other** at **44.1 kHz**, using a 1024-sample
asymmetric analysis window, 256-sample synthesis frame and 128-sample hop.
The graph delay is **128 samples**; one asynchronous scheduling hop makes the
plugin delay **256 samples / 5.80 ms** with a 128-sample host buffer.

## Artifact and state

The model is tracked with Git LFS and needs no external weights file.
SHA-256: `c7ea50ac67bf4bfddf1f5ff41c6eb419af00fe420ce1a0b0eaeef11a1861cd61`.
Size: **38,298,020 bytes**. Runtime: **ONNX Runtime 1.26.0 CPU**.

| Input | Shape | Output |
| --- | --- | --- |
| `audio_chunk` | `[1,2,128]` | `separated_chunk`: `[1,4,2,128]` |
| `audio_history` | `[1,2,896]` | `next_audio_history`: same shape |
| `fusion_hidden` | `[2,1,1000]` | `next_fusion_hidden`: same shape |
| `spectral_numerator_tail` | `[1,4,2,128]` | `next_spectral_numerator_tail`: same shape |
| `waveform_tail` | `[1,4,2,128]` | `next_waveform_tail`: same shape |
| `attention_keys` | `[1,31,64]` | `next_attention_keys`: same shape |
| `attention_values` | `[1,31,128]` | `next_attention_values`: same shape |
| `spec_memory_hidden` | `[1,1,500]` | `next_spec_memory_hidden`: same shape |
| `waveform_memory_hidden` | `[1,1,500]` | `next_waveform_memory_hidden`: same shape |

All public tensors are float32. Initialize all eight states to zero and carry
every returned state unchanged. Fusion and branch GRU states use the public
scale 2^-18. Discard the first output after reset; each subsequent call returns
the preceding input hop. Pad a partial final hop once, submit exactly one zero
hop, then drain the plugin queue. Reset every state after a discontinuity.

The [CMake contract](../cmake/QualifiedModelContract.cmake) owns identity,
geometry and metadata. Configuration verifies the graph hash and size; loading
checks nine inputs, nine outputs and 90 metadata entries. Its `QUALIFIED`
variable prefix denotes this identity lock, without a timing guarantee.

The attention cache and two branch memories use only received features. The
new branch states add 4,000 bytes per stream and no audio lookahead or queue.
The graph also includes magnitude features, quadrature spectral correction and
a nonlinear fused-feature map.

## Output behavior

The graph receives raw stereo input levels and returns all four stems. Plugin
buses are Main, Drums, Bass, Other, Vocals; graph order is Drums, Bass, Vocals,
Other. The writer applies the existing linked confidence and recovery fades to
Drums/Bass/Vocals, then computes `Other = Main - Drums - Bass - Vocals`.
Complete fallback routes Main to Other.

The confidence envelope opens immediately, holds for 50 ms, releases by
60 dB per 100 ms, and smoothly opens between -96 and -72 dBFS peak.

## Quality and current scope

This graph converts the two branch-memory output projections to signed integer
weights and dynamic unsigned activations. Sixteen matrix products now use this
arithmetic. The previous fourteen integer projections, nonlinearities and
remaining floating operations retain their previous graph definitions.
The source EMA checkpoint still scores 4.465157 dB full-band SDR. The current
fourteen-projection deployment scores 4.455150 dB; v0.4.0 scores 4.455188 dB.
These are separate endpoints from this candidate.

The exact candidate scores **4.455153 dB full-band SDR** on the unchanged
14-track / 28-excerpt panel. Its change from PR #15 is +0.000003 dB, with a
paired-track 95% interval of -0.000033 to +0.000037 dB. Against v0.4.0, the
change is -0.000035 dB, with an interval of -0.000224 to +0.000127 dB.
This is an inference-cost candidate for M4 testing; the 5 dB goal remains unmet.

| Full-band SDR, dB | Drums | Bass | Vocals | Other |
| --- | ---: | ---: | ---: | ---: |
| Candidate | 4.403038 | 4.995898 | 5.307888 | 3.113788 |
| Change from PR #15 | +0.000014 | +0.000041 | -0.000027 | -0.000015 |

Thirty-three of 56 track/stem SDR cells regress against PR #15; the largest
loss is 0.000258 dB on Young Griffo drums. SIR regresses in 32 cells, with the
largest loss on Skelpolu Other: -5.665968 to -5.697832 dB (-0.031864 dB).
That same cell is 0.228016 dB below v0.4.0. Thirteen of 23 natural-absence
cells have more unwanted output; the largest increase is 0.010058 dB on
Rockshow bass, to -60.403596 dBFS. The largest low-band loss is 0.002400 dB
on Rockshow vocals at 80–250 Hz. The [deployment report](quality-deployment.json)
retains all cells, baseline comparisons and 840 paired source-view windows.
Embedded graph metadata retains its export-time quality status; that report
records the completed deployment measurements against the exact graph hash.

Instrumental vocal output averages -47.258101 dBFS, or -26.661551 dB relative
to the mix. Its 0.000523 dB decrease from PR #15 leaves the leakage problem
essentially unchanged. The same 17/420 active instrumental windows remain
within 10 dB of the mix; Rockshow at 80–81 seconds remains only 0.591037 dB
below it. Isolated-vocal SDR is 22.502550 dB, signed gain is 0.922531, and
instrumental Other SDR is 5.484778 dB. Representative instrumental material,
quiet real vocals and user listening remain necessary alongside this panel.

## Numerical and native checks

The candidate passed ten short cases across optimized/unoptimized execution
and 8,216 longer graph calls per implementation, including exact reset replay.
Maximum waveform error was 8.940697e-8 against the independent sixteen-projection
reference, within the unchanged tolerance. The portable fixture covers eight
clip lengths, including partial EOF; PyTorch generated the expectations with
ORT imports actively blocked. See [streaming evidence](streaming-validation.json).

The native Release suite passed 164 tests, with one Windows-only skip and
seven performance tests disabled by default. See [Linux evidence](linux-validation.json).
Configuration now authenticates fixture
bytes and their declared model identity. Both parity tests also compare the
fixture with the model identity compiled into their binary.

The local native comparison measured a 9.4% reduction in median block p50,
from 4.526781 to 4.101744 ms, against the fourteen-projection graph. It used one
ORT worker, preallocated tensors, eight alternating blocks and concurrent
training on a shared Linux host. Every measured call exceeded that host's
2.902494 ms hop budget. This establishes neither M4 timing nor plugin deadline
acceptance. The graph file is 1,491,894 bytes smaller.

## M4 playback and numerical diagnosis

The user reported 3,072 cumulative fallback samples after ten minutes with the
PR #15 fourteen-projection candidate. Earlier local Mac tests also failed both
independent PyTorch parity checks, while hosted macOS and Windows correctness
runs passed. Those results belong to the previous graph; the new candidate has
not been tested on a physical M4.

[The diagnostic instructions](../M4_TESTING.md) compare default ORT execution,
KleidiAI disabled and graph optimizations disabled against the same independent
fixture. Alternate quantization kernels are a hypothesis for the M4 mismatch;
production settings and accuracy tolerances are unchanged. All 24 diagnostic
comparisons passed locally for this candidate. A passing Linux result does not
resolve the reported Mac failures.

The extended soak now retains complete callback/worker timing for the requested
30 minutes when tracing is enabled. Untraced repeated soaks and installed-DAW
playback remain the timing gates: zero new steady-playback fallback, with
startup/reset counters retained separately, at 44.1 kHz / 128 host samples /
one inference worker. Include transport changes, representative instrumental
passages, quiet real vocals and Other-stem listening. Preserve v0.4.0 and the
PR #15 bundle for rollback.
