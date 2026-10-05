# Debugging a Windows crash or hang

[Issue #29](https://github.com/sweetspotsoundsystem/stemgen-rt/issues/29) reports
Ableton Live 12.4.5 crashing when StemgenRT is inserted into an audio track on
Windows 10, with a second report on Windows 11. Those reports do not identify the
plugin version. A separate v0.6.2 reproduction now has hang snapshots and a
pluginval report; its analysis is below.

## Automated coverage

The existing C++ tests link the processor directly. Windows CI now also validates
the distributable ZIP with [pluginval](https://github.com/Tracktion/pluginval):

- Extract into a separate directory and verify the expected model identity,
  x86-64 binaries, manifest and bundled runtime against the build SDK.
- Load the VST3 through a host, with editor tests enabled, at 44.1/48/96 kHz and
  64/128/256/512/1024/4096 samples per callback. Only 44.1 kHz enables separation; the
  other rates exercise safe fallback.
- Fail CI on a validation error or crash. Keep logs, binary hashes and the exact
  Release VST3 PDB in a diagnostics artifact, including after validation failure.

This covers packaging and host lifecycle compatibility. It does not establish
Ableton compatibility, sustained real-time performance or a fix for this report.
The processor regression also simulates a host that deactivates/reactivates in
response to every latency notification, at 128 and 4096 samples. It requires the
restart requests to settle, and checks actual buffer/rate changes still notify.
Symbols from a different build cannot reliably decode an existing release dump.

To validate locally, extract pluginval v1.0.4 for Windows, install CMake and use
PowerShell 7 with a fresh evidence directory:

```powershell
./scripts/validate-windows-package.ps1 `
  -PluginArchive 'C:\Downloads\StemgenRT-Windows-VST3.zip' `
  -PluginvalPath 'C:\Tools\pluginval.exe' `
  -OutputDirectory 'C:\Temp\StemgenRT-validation-1'
```

Use this script from the source revision matching the package, because it checks
the model against that revision's contract. For a local build, also supply
`-ExpectedOrtDll './libs/onnxruntime/lib/onnxruntime.dll'` to verify DLL identity.
The script leaves evidence in the chosen directory and does not install a plugin.

## Reproduce in Ableton

Install the **same complete VST3 bundle** that crashed, restart Live and use a
blank Set with one stereo audio track. First try inserting the plugin without
playback, then starting playback. Record the plugin version/archive hash, Windows
and Live versions, CPU, audio driver, sample rate, buffer size and exact trigger.
Use 44.1 kHz/128 samples for the initial separation test, then the reported setup.

Capture the failure with [Microsoft ProcDump](https://learn.microsoft.com/en-us/sysinternals/downloads/procdump).
With Live running, find its PID in Task Manager's Details tab. Replace `1234`
below with that PID, create the dump directory, then insert the plugin:

```powershell
New-Item -ItemType Directory -Path 'C:\Temp\StemgenRT-dumps' -Force
./procdump64.exe -ma -e 1234 'C:\Temp\StemgenRT-dumps'
```

`-e` captures an unhandled exception; `-ma` includes process memory. A full dump
can contain loaded project/audio data, so use an empty Set and share it privately.
For a freeze, use `-h` instead of `-e`. If Live remains hung, take a second manual
snapshot after about a minute to distinguish a transient wait from a persistent
failure:

```powershell
./procdump64.exe -ma -h 1234 'C:\Temp\StemgenRT-dumps'
# Run separately while Live is still hung:
./procdump64.exe -mm 1234 'C:\Temp\StemgenRT-dumps\Live-persistent-hang.dmp'
```

The smaller dump may omit heap state; keep the full dump locally for further
analysis. These commands inspect the existing process without changing the
installed plugin.

If Live handles the exception and no dump appears, attach Visual Studio/WinDbg
and break on the relevant exception, or use ProcDump's first-chance option after
checking which exception Live reports. Plugin scanning can use a different
process; attach to that process if the crash occurs during scanning.

Open the dump in Visual Studio or WinDbg with the matching plugin binary and PDB.
Inspect the exception code, faulting module and all thread stacks. This should
distinguish failure during DLL loading, processor construction, model preparation,
editor creation or teardown. Then add a focused regression for the confirmed
trigger and rerun both packaged validation and the Ableton reproduction.

## v0.6.2 Ableton insertion hang: evidence and candidate fix

On 2026-10-04, a user reproduced an insertion hang in Live 12.3 on Windows 11
Pro (build 26200), Ryzen 5 5500, MME/DirectX, 44.1 kHz/4096 samples, with no input
device. The Set had an empty audio track and transport was stopped. The editor
did not finish opening. Live at 128 samples and playback remain untested.

The installed binary and pluginval-tested binary both match the shipped v0.6.2
VST3 SHA256:

```text
c3c8a935abfbadf66874f84412cdc5287d1b190f4d35ac7c31d320caa2593ad5
```

Both debugger stack listings, about one minute apart, place Live's main thread
in `SleepEx`, called from StemgenRT RVA `0x1ab28f`, with the caller at RVA
`0x45bfaf`. Inspection of that exact binary identifies the caller as the
`prepareToPlay()` inference-warmup polling loop: it checks request state 4
(`Processed`), a two-second steady-clock deadline and a one-millisecond sleep.
The next frame, RVA `0x19fcc3`, is JUCE's VST3 preparation wrapper. The nearest
export label `InitDll` is not the source function for this address.

The captured stacks locate the wait but do not prove whether this is one blocked
prepare or repeated preparations. The original pluginval 1.0.4 strictness-5 test
passed with 64/128/256/512/1024-sample blocks; it did not cover 4096. A pluginval
pass also does not assert that the separation engine is active. Both StemgenRT's
`onnxruntime.dll` and Live's `onnxruntime_abl.dll` were loaded; their presence
alone does not establish a conflict. The full dumps have not been independently
inspected, only the attached reports and debugger output.

The source has a reproducible latency-notification loop: successful preparation
reports nonzero PDC, while `releaseResources()` reports zero PDC. A host that
restarts the plugin after latency changes keeps receiving new changes on each
deactivate/reactivate cycle. A bounded host simulation failed to settle with the
old code at both 128 and 4096 samples. The candidate fix preserves the host's
last advertised latency during deactivation while disabling active inference.
The next preparation still reports a changed reserve or zero on startup failure
or unsupported configuration. JUCE's VST3 wrapper forwards these notifications
as `kLatencyChanged`; [a JUCE maintainer also describes the resulting host
deactivation/reactivation](https://forum.juce.com/t/ableton-vst3-96khz-calling-preparetoplay-for-every-buffer/48140/4).

This fixes the simulated lifecycle defect. Confirmation for the reported Windows
hang requires installing the candidate's complete bundle and repeating the same
Live setup, then testing 128 samples. If it still hangs, use the candidate's
matching PDB to inspect the worker and main-thread stacks, and check whether
preparation is being re-entered before changing inference scheduling.
