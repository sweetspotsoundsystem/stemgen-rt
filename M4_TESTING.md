# Attention model tests on M4 and M4 Pro

The deployment graph scores 4.288488 dB SDR on the development panel; its source
attention checkpoint scores 4.288099 dB. This candidate keeps one
inference worker, 44.1 kHz audio, 128-sample model hops and 256 samples of total
latency with a 128-sample host buffer. It has no target-Mac timing result yet.
Linux correctness passes; Linux/WSL2 timing tests fail, including 2,925 missed
output boundaries in 10,000 measured callbacks. See [the evidence](model/README.md#timing-limits).

## Build and collect machine evidence

Use a fresh checkout on each Mac because the qualification script creates and
authenticates its own dependencies and Release build. Install Xcode Command
Line Tools, CMake, Ninja and Git LFS first. From a native arm64 Terminal:

```bash
git clone --branch ax/best-model-m4-test https://github.com/sweetspotsoundsystem/stemgen-rt.git stemgen-rt-attention
cd stemgen-rt-attention
git lfs pull
./scripts/qualify-macos.sh ../stemgenrt-attention-m4-evidence
```

On the M4 Pro, use a distinct output name such as
`../stemgenrt-attention-m4-pro-evidence`. The evidence path must be new and
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

The PR leaves the C204 release available at commit
`6fc2382` (the base of this test branch). Keep its installer/build, or build that
commit in a separate checkout and run its `scripts/install-plugins.sh --release`
to restore the complete bundles. The installer replaces whole bundles; avoid
copying files into an existing plugin bundle.
