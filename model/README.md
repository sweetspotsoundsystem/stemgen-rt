# Bundled experimental model

This is the asymmetric-window hop128 endpoint after 500 adaptation updates and 250 matched teacher updates, 5000 total updates in its full ancestry. The frozen accepted 11.6 ms model provided an additional native-output L1 target at weight 0.5 during training. The teacher is absent from inference. Listening acceptance and Apple M4 timing are pending.

| Identity | Value |
| --- | --- |
| ONNX SHA-256 | `6f380e2a1e5e644b0222ff61450a8222af41ca5c51a668a360b1de2f4e6829c7` |
| ONNX size | 111,342,157 bytes |
| Checkpoint SHA-256 | `05a973af6efa3482e759f63cb3d268646a80f547a3bdc13679f0e228efba6222` |
| Tensor-state SHA-256 | `1daa6edb7be90eb641817b788b5540c89de57b125e454bdf463469ffcd3366ea` |
| Architecture | `cropped1024-asymmetric256-hop128-v1` |
| Sample rate / hop | 44,100 Hz / 128 samples |
| Graph / queue delay | 128 / 128 samples at a matching host block |

The [CMake contract](../cmake/QualifiedModelContract.cmake) owns build/runtime identity and interface values. Its historical `QUALIFIED` names are an identity lock; they do not certify M4 DAW performance. The checkpoint's [canonical metadata receipt](validation/canonical-checkpoint.json) corrects four inherited lineage descriptions and proves every tensor bit-exact to the scored endpoint.

## Quality

The fixed 14-track full/low/SIR macro scores are **3.846589 / 2.772156 / 6.700406 dB**. Differences against the accepted 11.6 ms model are **-0.211127 / -0.237271 / -0.613642 dB**, each with a negative paired 95% interval.

The [matched comparison](validation/matched-quality-comparison.json) uses exactly the same 250 augmented batches and learning rates for teacher and weight-zero control. Teacher-minus-control full/low SDR differences are **+0.033602 / +0.047605 dB**, with paired intervals **[+0.005419,+0.062447] / [+0.018561,+0.076508]**. The **+0.045516 dB** SIR difference has an interval crossing zero. Bass probes are mixed: relative unexplained energy on hop-frequency tones remains **4.1271 dB** above the accepted model. These measurements do not establish preserved audible bass fidelity.

## Runtime evidence and limits

The [export verification](validation/export-verification.json) passes all six short comparisons of the original CPU PyTorch model, export copy and ORT, including nonzero state, reset, partial EOF and final-sample recovery.

The [native Linux diagnostic](validation/native-linux-diagnostic.json) processes 75 seconds of continuous music from sample zero with no interior flush. All four stems in seconds 60–75 pass the unchanged waveform thresholds: maximum absolute error **1.812e-5** (limit 1e-4), maximum callback RMS **8.499e-6** (limit 1e-5). Every real input sample is recovered, with one graph flush and one queue drain. Physical samples map to callback samples plus **256**, and mixture reconstruction error is at most **1.491e-8**. All eight partial EOF cases and deterministic reset replays pass.

The same run **fails realtime timing and paced correctness on Linux**: median worker inference is **3.893 ms**, exceeding the **2.902494 ms** deadline. It records **646 missed observed output boundaries**, queue drops and sequence gaps. Its [actual exit code is 3](validation/native-linux-execution.json); the overall native result remains failed. Offline mapping and waveform agreement do not override that failure. The user identified Apple M4 as the target and accepted ending Linux tuning.

The independently regenerated [synthetic PyTorch fixtures](../test/fixtures/cropped1024-pytorch.json) cover lengths 1, 127, 128, 129, 255, 256, 257 and 16521. Their generator authenticates the canonical checkpoint and research sources, runs CPU FP32 with zero state and exactly one flush, and never uses ORT as an oracle. Plugin tests retain all-four-stem and actual output-bus checks, including variable offline callbacks and the existing 64-sample recovery fade. The Release build and fixture generation pass. [Linux integration validation](validation/linux-plugin-validation.json) records 150 distinct passing checks, one platform-specific skip and seven opt-in tests disabled. The first full suite found five stale geometry/metadata fixture expectations; all five corrected checks and both native/bus parity checks pass on the rebuilt final binary. Production code is identical across those test runs.

Run `CroppedModelParityTest.*` for integration checks and `scripts/qualify-macos.sh` for a fresh native Mac build and paced timing evidence. An installed-plugin check under representative DAW load and listening acceptance remain required. The accepted model and its working 11.6 ms branch remain available unchanged.
