# Bundled model

The payload is the exact **Raw L1 +250 cropped1024** export accepted in the research listening comparison: 250 additional raw-four L1 updates from the 2000-update parent, 2250 total. No new training or graph rewriting was performed for this plugin integration.

| Identity | Value |
| --- | --- |
| ONNX SHA-256 | `4a43cf08a088938c8d0f4be5f15ce7c6aa78d82bbd9eeec952b2fad15032598e` |
| ONNX size | 114,414,055 bytes |
| Checkpoint SHA-256 | `ac46729e5e4d379b09914a6e40ae927e09089b43fd4eef219ae7e034f355da65` |
| Tensor-state SHA-256 | `a12c215810026c603a1fd394383c1646219b8b3f764ebe9c2a83856404443aa4` |
| Architecture | `ola-cropped1024-hann512-hop256-v1` |
| Sample rate / hop | 44,100 Hz / 256 samples |
| Graph / queue delay | 256 / 256 samples at the matching host block |

The [CMake contract](../cmake/QualifiedModelContract.cmake) owns build/runtime identity and interface values. `QUALIFIED` in those variable names is inherited terminology; neither the filename nor the graph's frozen metadata represents M4 DAW qualification.

## Evidence and limits

The [original export verification](validation/export-verification.json) passed six three-way comparisons of the original PyTorch model, export copy and ORT, with exact reset replays. The graph preserves the four complete deployed outputs, including Other. The original research native harness recovered all real samples at eight partial EOF lengths with one graph flush and one queue drain. Four-stem error on the audition after its continuous input lead-in was at most 9.425e-6 per sample and 4.550e-6 callback RMS.

The user reported that the bass buzzing was gone and that the labelled Actions 60–75 second audition sounded like c91. That is scoped listening acceptance. The 14-track full/low/SIR macro deltas against c91 were **-0.201273 / -0.286965 / -0.355305 dB**, each with a negative paired 95% interval. The candidate is not dataset-wide numerical equivalence to c91.

Longer Linux research timing runs had **96 missed outputs with two threads** and **100 with four threads**, each over 8192 callbacks. A shorter two-thread pass did not override those failures. The user identified Apple M4 as the deployment target and accepted ending Linux tuning. M4 parity, timing under DAW load, AU/VST3 scanning and installed-plugin listening remain target-machine checks.

Plugin integration adds an independent [synthetic PyTorch oracle](../test/fixtures/cropped1024-pytorch.json), generated directly from the authenticated checkpoint without ONNX. It verifies all four native outputs and actual Main/stem buses at lengths 1, 255, 256, 257, 511, 512, 513 and 16521, with fixed/variable offline buffers and partial stopped callbacks. The per-stem maximum-error gate is 1e-5; Main uses 1e-7 and mixture reconstruction 1e-6. The existing 64-sample recovery fade is accounted for explicitly.

Run `CroppedModelParityTest.*` for these integration checks. Use `scripts/qualify-macos.sh` for a fresh native Mac build and paced timing evidence. Passing portable tests or CI builds alone does not establish actual M4 real-time performance.

The original checkpoint and c91 research reference remain unchanged. This PR replaces only the bundled plugin graph and its deployment contract.
