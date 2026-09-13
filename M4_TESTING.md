# Model tests on M4 and M4 Pro

This revision uses the saved EMA checkpoint with two additional branch memories.
The exported graph scores 4.455188 dB SDR; its source checkpoint scores
4.465157 dB. See the [quality and validation report](model/README.md). This release keeps one
inference worker, 44.1 kHz audio, 128-sample model hops and 256 samples of total
latency with a 128-sample host buffer. The user reported successful M4 Pro testing
of PR #13; released-AU testing on M4 and formal paced timing evidence remain pending.
The model SHA-256 begins `d2945742d27f`. Results from the previous attention
graph (`e354d24bfa0f`) or C204 must be recorded separately.

## Test the released AU

Download `StemgenRT-macOS-AU.zip` from the
[v0.4.0 release](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/tag/v0.4.0).
Quit the DAW, extract the archive and replace the complete `StemgenRT.component`
bundle in `~/Library/Audio/Plug-Ins/Components/`. Keep the previous bundle outside
the plugin directory for rollback. Restart the DAW and use the audition checks
below. The bundle includes the model and ONNX Runtime; a source build is optional.

## Build and collect machine evidence

Use a fresh checkout on each Mac because the qualification script creates and
authenticates its own dependencies and Release build. Install Xcode Command
Line Tools, CMake, Ninja and Git LFS first. From a native arm64 Terminal:

```bash
git clone --branch v0.4.0 https://github.com/sweetspotsoundsystem/stemgen-rt.git stemgen-rt-branch-memory
cd stemgen-rt-branch-memory
git lfs pull
./scripts/qualify-macos.sh ../stemgenrt-branch-memory-m4-evidence
```

On the M4 Pro, use a distinct output name such as
`../stemgenrt-branch-memory-m4-pro-evidence`. The evidence path must be new and
outside the checkout. Use AC power, turn off Low Power Mode, close the DAW and
keep background work comparable between runs.

The script builds AU/VST3, runs correctness tests and bundle checks, records
native inference controls, and runs 10,000 paced callbacks. Its thread sweep is
a diagnostic; production inference stays at one worker. A failed deadline test
is useful evidence: preserve the whole output directory and terminal result.

## Audition in the DAW

After the build and correctness checks complete, use the normal bundle installer:

```bash
./scripts/install-plugins.sh --release
```

Set the DAW to 44.1 kHz with a 128-sample buffer. Check Main, Drums, Bass, Other
and Vocals on the same material used for C204, including instrumental passages,
quiet vocals, low bass and transients. Exercise playback start/stop, seeks and
loop boundaries. Record the plugin's underrun count and whether you hear dropouts
or unwanted vocal spill. Compare both Macs under the same session load.

The average quality gain has local regressions, listed in
[model/README.md](model/README.md). Save the evidence directories and listening
notes with the model/commit identity so timings can be attributed to this build.

## Return to C204

The C204 baseline remains available at commit
`6fc2382` (release v0.3.0). Keep its installer/build, or build that
commit in a separate checkout and run its `scripts/install-plugins.sh --release`
to restore the complete bundles. The installer replaces whole bundles; avoid
copying files into an existing plugin bundle.

The previous attention candidate is also available at `c848050` for comparison.
