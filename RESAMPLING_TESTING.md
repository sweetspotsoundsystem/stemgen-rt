# Testing host sample-rate conversion

This branch enables **44.1, 48, 88.2, 96, 176.4 and 192 kHz** sessions. The
model runs at 44.1 kHz. Higher-rate input is converted before inference, and
Drums, Bass and Vocals are converted back on the inference worker. Other is
reconstructed from native Main minus those three stems.
At 44.1 kHz both converters bypass exactly.

Use this PR's successful **CMake** workflow artifacts for macOS AU/VST3 or
Windows VST3. The 0.6.0 release downloads still accept only 44.1 kHz. For a
local build, follow the README's build commands on `codex/host-rate-resampling`.
Replace the complete plugin bundle and restart the DAW; keep the previous bundle
available for comparison.

## Expected latency

The editor and DAW receive the full delay, including resampling and scheduling.
For a prepared **128-sample** host buffer:

| Session rate | Reported samples | Reported milliseconds |
| --- | ---: | ---: |
| 44.1 kHz | 256 | 5.80 |
| 48 kHz | 702 | 14.63 |
| 88.2 kHz | 947 | 10.74 |
| 96 kHz | 1276 | 13.29 |
| 176.4 kHz | 2021 | 11.46 |
| 192 kHz | 2422 | 12.61 |

At **48 kHz / 512 samples**, expect **958 samples / 19.96 ms**. The resampling
filters account for 167 host samples (3.48 ms) at 48 kHz; the rest is graph and
scheduling delay. Changing the buffer can change the scheduling reserve, so
latency does not simply scale with the sample rate.

## M4 playback and export checks

1. Start with the same song at 44.1 kHz / 128 samples to check the baseline.
2. Run a 48 kHz session at 128 samples for several minutes, then try 256 and
   512. Record the DAW, machine, buffer, playback duration, displayed latency
   and fallback count before/after. Note CPU readings and any audible gaps.
3. Stop/start, seek, loop and change the host rate between 44.1 and 48 kHz.
   Let the DAW reprepare the plugin after a rate/buffer change. Listen for stale
   audio or bursts and check that the latency display updates.
4. Route Main, Drums, Bass, Other and Vocals separately. Sum only the four
   stems and compare with Main using delay compensation; they should reconstruct
   the same audio. Adding Main to the stem sum doubles the mix.
5. Export a short passage with a clear ending, including the reported tail.
   Check the last notes and compare offline export with playback.
6. Repeat at higher session rates you use, starting with 96 kHz.

Main retains the full native input bandwidth. The model path uses a 20 kHz
passband and rolls off before 22.05 kHz; Other carries the residual needed to
reconstruct Main. The linked confidence/recovery fades still apply.

Automated tests check numerical behavior and scheduling, with worker results
made ready between callbacks where needed. They do not measure real-time M4
headroom. Physical-machine timing and listening results for resampling remain
pending; the earlier zero-fallback report applies to 44.1 kHz.
