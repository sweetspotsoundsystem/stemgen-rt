# StemgenRT-5.8 deployment candidate

This graph exports the frozen **teacher004 EMA** research baseline at 45,750
cumulative updates. The teacher is used only during training. Matching model,
training, objectives, data, recovery, evaluation and export code accompany the
candidate in [StemgenRT-5.8 PR #5](https://github.com/sweetspotsoundsystem/StemgenRT-5.8/pull/5).

The graph SHA-256 is `77164d6a581fafb2a31f53fd8ffde44c07cf618472952a4cdba14e68dda3b8b9` (37,529,132 bytes).
The source checkpoint SHA-256 is `7fcd444f83985c0aab0c76923c4355fea3c3e11aa32c81410a6bb9388b755154` and decoded EMA
state SHA-256 is `f78c49b3755d6a71b7890482d3a40a5662b417b7a63ed3ad9d393cfb0da0037b`.
`cmake/QualifiedModelContract.cmake` locks the exact identity and all 60 metadata
entries. Metadata uses the `stemgenrt.*` namespace and retains transformation
ancestry. The file is self-contained and tracked with Git LFS.

## Interface and latency

The existing 17-product U8/S8 deployment arithmetic, eight FP32 states, stereo
44.1 kHz input and 128-sample hop are retained. Analysis is 1024 samples and
synthesis is 256 samples. Graph delay is 128 samples; the asynchronous worker
adds 128 samples with a 128-sample host buffer, for **256 samples (5.8 ms)** total.
There is no new audio buffer. Carry every state unchanged, reset to zero and
flush once after a padded partial final hop.

The plugin retains one ORT worker with spinning disabled and
`mlas.disable_kleidiai=1`. Raw input levels, the confidence envelope and
`Other = Main - Drums - Bass - Vocals` are unchanged.

## Measured quality

The exact deployment graph scores **4.564148 dB full-band SDR**, **+0.108976
dB** versus shipped v0.6.1 (4.455173 dB). The source FP32 checkpoint scores
**4.564402 dB**; it is a separate endpoint. Scoring uses the same 14 tracks,
28 physical excerpts, continuous carried state and unchanged metric code.
The retained product control reproduced every metric for its first track exactly.
The paired-track bootstrap 95% interval for the mean gain is [-0.006083,
+0.202519] dB. This repeatedly used development panel does not establish a
statistically conclusive improvement on unseen tracks.

| Stem | v0.6.1 (dB) | Candidate (dB) | Change (dB) |
| --- | ---: | ---: | ---: |
| Drums | 4.403057 | 4.474373 | +0.071316 |
| Bass | 4.995904 | 5.146444 | +0.150540 |
| Vocals | 5.307862 | 5.446602 | +0.138740 |
| Other | 3.113867 | 3.189174 | +0.075306 |

16/56 track/stem cells decrease; the worst change is
-1.103074 dB for drums on Skelpolu - Human Mistakes. The
[deployment report](quality-deployment.json) includes every track/stem/band/absence
comparison, paired track bootstrap intervals and all 840 source-view windows.

On instrumental remixes, mean unwanted vocal output is -48.024391
dBFS (-0.769824 dB versus v0.6.1; positive means more leakage).
The largest active instrumental-window increase is
+6.774830 dB on
Skelpolu - Human Mistakes; all 840 paired window deltas are retained.
On isolated vocals, desired-vocal SDR changes by +3.450404 dB.
These development measurements do not establish instrumental listening acceptance
or new held-out performance. The 5 dB research target remains unmet.

## Validation and testing

[Streaming checks](streaming-validation.json) pass on ORT 1.26.0 with the
independent NumPy/PyTorch integer reference, including short trajectories,
2,048-hop carried-state cases, nonzero initial states, silence, partial EOF and
bit-exact reset replay. ORT inference was blocked while generating the eight
fixture cases; expected outputs never come from ORT.

The [Linux native suite](linux-validation.json) passed **174 tests**,
with one platform-specific skip and seven disabled performance tests. Coverage
includes four-stem reference parity, identity, reset/EOF, alignment, queue and
epoch recovery, reconstruction, variable callbacks and callback allocation checks.

[Windows and macOS arm64 CI](ci-validation.json) also passed the build and test
jobs, including both independent PyTorch parity tests, at source commit
`eb5891e74b6243ac8c915d6d7640bcecb22605ba`. The subsequent report update changes
only documentation and report JSON, preserving the tested graph, fixtures,
contract and runtime code.

Physical M4 parity and sustained timing are **pending for this graph**. Follow
[M4_TESTING.md](../M4_TESTING.md), then test the installed AU in Ableton and listen
to instrumental passages, quiet vocals and Other. Previous timing and hardware
reports are retained under [baseline-v0.6.1](baseline-v0.6.1/README.md) and do not
qualify these weights. Preserve the shipped v0.6.1 bundles for rollback.
