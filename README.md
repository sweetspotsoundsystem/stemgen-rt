# StemgenRT

StemgenRT is a real-time VST3/AU music-source-separation plugin with stereo
Drums, Bass, Other, and Vocals outputs.

This branch contains the unpromoted c214 step-100,000 listening candidate. It
processes 256-sample hops and reports 256 samples of latency: 5.80499 ms at
44.1 kHz. The qualified 512-hop model remains untouched on the existing
production-candidate branch and is the fallback until c214 passes target-Mac
timing and listening review.

> [!WARNING]
> c214 is for listening and qualification, not release. The ONNX export passed
> checked 64-hop CPU parity and exact reconstruction. Full quality
> qualification, the target-Mac thread sweep, the 10,000-callback paced test,
> and human listening approval remain required before promotion.

Built with [JUCE](https://github.com/juce-framework/JUCE),
[ONNX Runtime](https://onnxruntime.ai), and
[HS-TasNet](https://github.com/sweetspotsoundsystem/HS-TasNet).

## Host contract

The listening build accepts exactly 44.1 kHz with a 256-sample prepared host
callback. Other configurations fail closed. One asynchronous inference-queue
hop supplies the complete 256-sample PDC; the audio callback never waits for
the model.

The model emits the current input chunk, uses no future context, pre-roll,
flush, or external overlap-add, and carries three explicit states:

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

All states start at positive zero and reset together on transport
discontinuities, sequence gaps, and inference failures. Only the inference
worker advances them.

## Mixture-lossless output

The graph and native runtime preserve Drums, Bass, and Vocals, then derive:

```text
Other = Main - Drums - Bass - Vocals
```

The four buses therefore reconstruct latency-aligned Main to float32
precision. This means mixture-lossless routing, not recovery of the original
studio stems. Independent clipping, normalization, or integer quantization
downstream can break exact summation.

## Model identity

The only model payload is `model/model.onnx`:

- Candidate: c214 step 100,000 plus exact c191 step-128 causal correction
- SHA-256: `8262f56503f1f8acf2cdb105822db474de2ec546b5f9dd0521bdc6b06b3a02ab`
- Size: 111,372,674 bytes
- Export qualification receipt SHA-256:
  `a07cbf17bdc7a8653339094fae43e22ce4eaa8432312328b253221ac750dda4f`

The 64-hop CPU check measured maximum absolute errors of `2.3842e-7` for
separated audio, `0` for analysis history, `1.9744e-7` for emitted history,
and `6.4433e-5` for the opaque fusion state. Reconstruction error was
`7.4506e-9`; reset replay was bit-exact and all states were live and finite.

`cmake/QualifiedModelContract.cmake` is the single editable source of model
identity, ABI, metadata, and PDC. Configuration, runtime loading, tests, and
bundle sealing fail closed against it.

## Build and install on macOS

The official ONNX Runtime 1.26.0 Apple Silicon package requires macOS 14 or
newer.

```bash
./scripts/download-onnxruntime.sh

cmake --preset release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build --preset release --target AudioPluginTest_BundleModel
ctest --preset release --output-on-failure

./scripts/install-plugins.sh --release
```

StemgenRT exposes Main plus four stereo stem buses. Route the Drums, Bass,
Other, and Vocals buses to separate tracks in the DAW.

## Target-Mac qualification

The 256-hop graph has half the former callback budget, so rerun both the CPU
thread sweep and the paced real-time test on the exact candidate:

```bash
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep

STEMGENRT_QUALIFICATION_CALLBACKS=10000 \
  build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenRealtimePaced
```

The paced summary must report a 256-sample callback and 5.80499 ms deadline,
zero deadline misses, underruns, queue/ring drops, unsafe callbacks, and
non-finite output, plus reconstruction error no greater than `1e-6`.

For the listening review, pay special attention to kick/sub transients,
low-frequency roughness around hop boundaries, reset/seek stability, clicks,
and sustained CPU headroom. Do not promote this branch based on a successful
build alone.

## License

MIT
