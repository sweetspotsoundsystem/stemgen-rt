# AGENTS.md

## Scope

This branch is the unpromoted c214 step-100,000 256-hop listening lane. Keep
it separate from the qualified 512-hop fallback. Do not merge or push it over
`ax/qualified-stateful-pipeline` without explicit listening approval and the
remaining target-Mac qualification evidence.

## Contract

`cmake/QualifiedModelContract.cmake` is the sole editable model-contract
source. CMake generates `StemgenRT/QualifiedModelContract.h`; runtime loading,
tests, packaging, and bundle sealing must fail closed on the exact model
SHA/size, names, shapes, and metadata.

The graph clock is 44.1 kHz and the only admitted listening configuration is a
256-sample host callback. The static corrected ABI is:

| Direction | Name | Shape |
| --- | --- | --- |
| Input | `audio_chunk` | `[1, 2, 256]` |
| Input | `analysis_history` | `[1, 2, 768]` |
| Input | `fusion_hidden` | `[2, 1, 1000]` |
| Input | `emitted_db_history` | `[1, 4, 2048]` |
| Output | `separated_chunk` | `[1, 4, 2, 256]` |
| Output | `next_analysis_history` | `[1, 2, 768]` |
| Output | `next_fusion_hidden` | `[2, 1, 1000]` |
| Output | `next_emitted_db_history` | `[1, 4, 2048]` |

The three state families start at positive zero and reset together. Only the
inference worker advances them, and only after every output has passed shape
and finite-value checks. The model has zero future context and graph delay,
no pre-roll/flush/external OLA, and current-input-chunk alignment.

## Real-time rules

- The audio callback must not allocate, lock, wait, or invoke inference.
- The asynchronous queue contributes one hop, so honest PDC is 256 samples.
- Preserve exact timeline tags; discard late output instead of replaying it.
- On fallback, route latency-aligned Main entirely to Other.
- Preserve raw finite model input; do not add per-hop gain normalization,
  crossover/reinjection, boundary crossfades, or hidden lookahead.
- Reset the queue epoch, all graph states, output confidence, and scheduling
  state on transport discontinuities and sequence gaps.
- Retain the macOS two-thread automatic ORT cap only as an audition starting
  point. The exact c214 graph requires a fresh target-Mac sweep.

## Mixture consistency

Output order is `[drums, bass, vocals, other]`. Preserve the first three and
derive `other = current_input - drums - bass - vocals` after ORT. Apply the
same invariant after native fallback/confidence processing. Never normalize,
clip, gate, or quantize stems independently after the final residual.

## Artifact

`model/model.onnx` is the only bundled payload. Expected identity:

- SHA-256 `8262f56503f1f8acf2cdb105822db474de2ec546b5f9dd0521bdc6b06b3a02ab`
- Size 111,372,674 bytes
- Export qualification receipt SHA-256
  `a07cbf17bdc7a8653339094fae43e22ce4eaa8432312328b253221ac750dda4f`

The candidate is listening-only. Do not describe old c191/c212 qualification,
timing, or listening evidence as transferred to c214.

## Build and test

```bash
./scripts/download-onnxruntime.sh
cmake --preset release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build --preset release --target AudioPluginTest_BundleModel
ctest --preset release --output-on-failure

build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep

STEMGENRT_QUALIFICATION_CALLBACKS=10000 \
  build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenRealtimePaced
```

The target-Mac paced test must cover 100 warmups plus at least 10,000 measured
256-sample callbacks with a 5.80499 ms deadline. Require zero misses,
underruns, drops, unsafe callbacks, and non-finite values, plus reconstruction
error at or below `1e-6`. Then install with
`./scripts/install-plugins.sh --release` and perform DAW reset/seek and
kick/sub listening checks.

Use C++20 and the repository clang-format configuration. Tests should derive
hop sizes from contract constants except when deliberately testing arbitrary
host/converter fragmentation.
