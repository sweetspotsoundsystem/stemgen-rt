# Final M4 DAW check

Run this on the exact launch candidate. Earlier stable Ableton playback is useful
evidence, but does not cover the new JUCE build. This takes about one hour total,
30 minutes in Logic (AU) and 30 minutes in Ableton (VST3), plus setup.

1. Record the candidate commit, archive SHA-256, macOS and DAW versions, M4 model,
   audio interface/driver, power mode and background workload. Install the complete
   candidate bundles as in [the installation guide](mac-installation.md).
2. Set the session to **44.1 kHz, 128 samples**. Use representative music you have
   permission to test. Confirm all four stems and Main route as expected. Main is
   delayed dry input; listening only to Main does not demonstrate separation.
3. Play for five minutes with one instance. After the first two seconds, record
   the UI's latency and Fallback sample count. Expected reported latency is
   **256 samples / 5.80 ms**. Record counters again after the segment.
4. Add a second instance on a separate active track and play for twenty minutes.
   Record both counters at the start and end. Listen for missing stems, crackles,
   dropouts, stuck sound and unexpected changes in routing.
5. During five further minutes, loop a short region, seek repeatedly, stop/start,
   bypass/unbypass, and save/reopen the session. Capture counters around each
   transition; they may reset when the host prepares the processor. Distinguish
   transition behavior from uninterrupted playback. Report any audible dropout
   or increased Fallback count with its exact trigger and which instance.
6. Separately try 48 kHz: the visible status must say separation is unavailable
   and request 44.1 kHz. Return to 44.1 kHz and confirm stems recover. Then check
   64/256/512/1024 buffers: expected PDC is 320/384/640/1152 samples. Restore 128.
7. Remove/reinsert the plugin and reopen the host. Confirm no freeze/crash, then
   repeat the same thirty-minute sequence in the other DAW.

Save a text report with the fields above and one row per segment: DAW, instance
count, duration, initial/final fallback counters, reset/seek events, and listening
observations. Include screenshots of the status/counters when convenient. Report
failures too; a successful AU test does not substitute for the VST3 test.
Passing means no crash/freeze, correct status and recovery, and no increasing
fallback counter or audible dropout during the steady playback segments. Any
transport anomaly needs inspection before claiming that scenario works.

Do not run a synthetic two-hour test for this handoff. This check specifically
fills the real-host, transport and multiple-instance evidence gap.
