# Bundled experimental model

This is the asymmetric-window hop128 leader-cleanup endpoint: 250 cleanup updates from the retained SDR leader, 8250 updates in its complete ancestry. Training adds instrumental-only and vocal-only mixtures with deployed-source supervision. The C91 teacher provides an additional target on ordinary mixtures only, at weight 0.5. These changes affect training; the inference architecture, source gains, four-state ABI and one-worker plugin processing remain unchanged.

The candidate improves the fixed quality panel and reserved-interval confirmation. Human listening and Apple M4 runtime checks remain pending for these weights. The working rollback is `ax/hop128-5ms-teacher` at `86562b9`; its model, fixtures and [historical validation notes](working-model-notes.md) remain available.

| Identity | Value |
| --- | --- |
| ONNX SHA-256 | `b8574ac2e67bcd1df533e3fc7464c0659d4bfa6744cfbb4389c0967271594fe3` |
| ONNX size | 111,344,465 bytes |
| Checkpoint SHA-256 | `2c68804549a6285b4d15e727945b955f89aeeb1b456534c3d47398066aae60b7` |
| Tensor-state SHA-256 | `c204b0fcb9627ca7fecd287db42fb869a1ae6783a1bc24cf2d8864c3b4a565fb` |
| Architecture | `cropped1024-asymmetric256-hop128-v1` |
| Sample rate / hop | 44,100 Hz / 128 samples |
| Graph / queue delay | 128 / 128 samples at a matching host block |

The [CMake contract](../cmake/QualifiedModelContract.cmake) owns build/runtime identity, geometry and all 99 embedded metadata fields. Its historical `QUALIFIED` names are an identity lock; they do not certify M4 DAW performance. Metadata retains the export's original unqualified-host flags.

## Quality and confirmation

The [completed primary comparison](validation/leader-cleanup-quality-summary.json) uses the unchanged 14-track protocol, with two 15-second intervals per track. Full/low/SIR macro scores are **4.069079 / 3.055463 / 7.121449 dB**, improving **0.222490 / 0.283306 / 0.421043 dB** over working. Full SDR exceeds the original 4.057716 dB recovery target by 0.011363 dB; the 4.258989 dB stretch target remains unmet. The paired full-SDR interval is **[+0.151870,+0.306618] dB**. Thirteen of fourteen track means improve. The final cleanup adds 0.043984 dB over its trained parent; the full gain over working is not attributed solely to the final 250 updates.

The endpoint was [selected before confirmation](validation/leader-cleanup-primary-selection.json). Its [reserved comparison](validation/leader-cleanup-confirmation-summary.json) uses 105–120 and 135–150 seconds on the same fourteen development tracks, streaming continuously from sample zero. These intervals were excluded from this selection; they are not unseen tracks or a sealed test set. Both evaluations and the [comparison execution](validation/leader-cleanup-confirmation-execution.json) complete with exit zero and unchanged inputs.

| Reserved metric | Working | Candidate | Difference | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Full SDR | 3.889436 | 4.127920 | +0.238484 | [+0.153539,+0.335283] |
| Low SDR, 20–250 Hz | 2.816608 | 3.115930 | +0.299322 | [+0.222806,+0.378926] |
| SIR | 6.922702 | 7.385882 | +0.463180 | [+0.223430,+0.802855] |

Reserved full-SDR gains for Drums/Bass/Vocals/Other are **+0.242859 / +0.150848 / +0.304871 / +0.255358 dB**; all four low-SDR and SIR means also improve. Full SDR improves on 13/14 track means and 49/56 track/stem cells. Low SDR improves on 48/56 cells and SIR on 39/56. Intervals resample whole tracks and exclude training-seed uncertainty and repeated-selection effects.

## Spill, wanted sources and retained costs

The [complete quality review](validation/leader-cleanup-quality-review.json) includes controlled source views, active vocal gain, quiet wanted sources, absence, bass probes and all three reference models. Against working, instrumental-input Vocal output falls **3.632562 dB**, and vocal-only Other output falls **2.848040 dB**. Wanted vocal-only full/low SDR improve **3.260426 / 2.586802 dB**, and absolute vocal projection-gain error falls **0.010901**. Wanted instrumental Drums/Bass/Other averages also improve. These measurements support improved separation without establishing a human listening verdict.

Regressions remain visible:

- On primary intervals, absent-source Other output rises **1.610331 dB**. Vocal SIR falls **0.089149 dB** versus the trained parent. James May vocals lose 0.258744 dB full SDR versus working.
- In the very quiet wanted-source panel, vocal full/low SDR improve **1.832204 / 1.196138 dB** over working, but bass changes **−0.007985 / −0.061204 dB**, and the single qualifying Other case changes **−1.607919 / −1.844189 dB**. Quiet vocal estimates remain inaccurate in absolute terms.
- Skelpolu's controlled vocal-only example remains severely misrouted: vocal projection gain averages **0.08016** and full SDR is **0.55764 dB**. A lower spill average does not establish recovery of this voice.
- On reserved intervals, Meaxic vocals lose **0.429464 dB** full SDR; Traffic Experiment bass loses **0.400903 / 0.440793 dB** full/low SDR. Traffic Experiment's full track mean falls 0.025342 dB. Fergessen Other SIR falls 1.637768 dB. Absent Other output rises **0.863916 dB**, while absent Drums/Bass/Vocals means fall.

No checkpoint is reselected using the reserved results. The source-view and quiet-source evidence concerns the primary development material; it was not separately rerun on the reserved intervals.

## Runtime evidence and limits

The [export verification](validation/leader-cleanup-export-verification.json) and its [execution](validation/leader-cleanup-export-execution.json) pass all six CPU comparisons under the original tolerances, using three independent 96-hop trajectories with state, reset and EOF checks.

The [continuous native comparison](validation/leader-cleanup-native-linux-diagnostic.json) processes 3,307,648 real samples from zero, with no interior flush, one final graph flush and one queue drain. All four stems in seconds 60–75 pass: maximum absolute error **1.627207e-5** (limit 1e-4), maximum callback RMS error **8.723194e-6** (limit 1e-5). Every real sample is recovered at the 256-sample callback offset, with reconstruction error at most **1.490116e-8**. All eight partial EOF cases and deterministic reset replays pass.

The same native run **fails paced correctness and realtime timing on Linux**, with [actual exit 3](validation/leader-cleanup-native-linux-execution.json). It records 326 missed observed output boundaries, 699 queue-full drops, sequence gaps and late discards. Median inference is 9.428707 ms versus the 2.902494 ms deadline. Four CPU confirmation workers were active concurrently, so this is not a controlled timing comparison against prior models. Offline waveform agreement does not override the paced failure. Earlier Linux and user-reported M4 timing failures also remain in the record; the user's practical acceptance of the working plugin does not qualify these new weights.

The [synthetic PyTorch fixtures](../test/fixtures/cropped1024-pytorch.json) have been independently regenerated for this checkpoint, with [exit zero](validation/leader-cleanup-fixture-execution.json). Lengths remain 1, 127, 128, 129, 255, 256, 257 and 16521. The authenticated native model runs on CPU FP32, starts with zero state, flushes once and never executes ORT as an oracle. Plugin integration checks compare all four native stems and actual output buses, including variable offline callbacks and the existing recovery fade.

The complete [Linux Release build and test suite](validation/leader-cleanup-linux-plugin-validation.json) pass: **162 checks**, one expected Windows-only skip and seven opt-in tests disabled. Both PyTorch parity tests, all 99 metadata fields, reset/EOF, routing, variable callback and callback heap-traffic checks pass. Project C++ sources compile as C++20 with warnings as errors. The two initial configuration failures are retained; a fresh cache with the existing pinned Linux SDK resolves them without production-code changes.

Run `CroppedModelParityTest.*` for integration checks and `scripts/qualify-macos.sh` for a fresh native Mac build and paced timing evidence. AU/VST3 signing and installed DAW behavior must be checked on the target Mac. This source update does not replace the installed working M4 plugin.
