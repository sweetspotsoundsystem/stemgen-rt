# StemgenRT-5.8 teacher004 candidate

This candidate exports the frozen best EMA checkpoint at 45,750 cumulative
updates. Its FP32 source score is **4.564402 dB** on the fixed 14-track,
28-excerpt development panel. The teacher is absent from inference.
Matching model, training, data, objective, recovery and export code are maintained
in [HS-TasNet PR #5](https://github.com/sweetspotsoundsystem/HS-TasNet/pull/5).

The graph SHA-256 is
`77164d6a581fafb2a31f53fd8ffde44c07cf618472952a4cdba14e68dda3b8b9`
(37,529,132 bytes). The contract locks all 60 metadata entries and the existing
nine inputs/outputs: audio plus eight persistent FP32 states. The 17 integer
products, 44.1 kHz sample rate, 128-sample hop and **256-sample graph-plus-host
latency** with a 128-sample host buffer are retained. Raw levels, residual Other,
confidence fade, one inference worker and KleidiAI disabling are unchanged.

[Short and long numerical parity](streaming-validation.json) pass with pinned
ORT 1.26.0. Eight independent NumPy/PyTorch fixtures cover partial EOF, reset,
all four stems and carried states; their expectations never come from ORT.
The [native Linux suite](linux-validation.json) passes **174 tests**, with one
platform skip and seven disabled performance tests.

The exact graph's [quality evaluation](quality-deployment.json) is running.
The source score must not be used as its deployment score. The shipped v0.6.1
control reproduced every retained metric for the first validation track exactly.
The PR remains a draft until the full report and portable package checks finish.

Physical M4 parity, sustained timing and listening acceptance remain pending
for these weights. Follow [the M4 instructions](../M4_TESTING.md) and preserve
the shipped v0.6.1 bundles for rollback. Historical reports are labelled under
[baseline-v0.6.1](baseline-v0.6.1/README.md); they do not qualify this graph.
