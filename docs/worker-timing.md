# Optional worker timing

This schema is a single-worker diagnostic. Set
`STEMGENRT_PACED_INDEPENDENT_WORKERS=1` when enabling `STEMGENRT_TRACE_WORKER=1`.
Three-worker playback rejects this trace because one calling-thread CPU clock
cannot account for both helper threads. Normal 0.8.0 playback uses three workers
without tracing.

`setWorkerTimingTrace` attaches bounded storage while the inference worker is stopped. Normal playback leaves it null. The worker samples clocks only with this diagnostic enabled; it performs no per-hop file I/O or allocation. The paced test writes CSV files after joining the worker. At 620157 measured callbacks plus warmup, callback and worker storage must fit the existing 128 MiB cap.

Schema 2 adds calling-thread CPU nanoseconds and timestamps immediately around `OrtApi::Run`. macOS/Linux use `clock_gettime(CLOCK_THREAD_CPUTIME_ID)`; Windows uses `GetThreadTimes`. An unavailable reading is -1. Process CPU remains explicitly separate. Windows thread CPU can have coarse accounting resolution.

The intervals are:

- Pre-ORT: worker run start through ORT start, including input validation and packing.
- ORT: the runtime `Run` call, including runtime dispatch and kernels.
- Post-ORT: ORT finish through worker run finish, including output validation, copying and persistent-state updates.

The CPU reads enclose the corresponding wall reads. Pre/post intervals include diagnostic sampling overhead. Wall minus thread CPU can be slightly negative; never clamp it into a claim of exact time off CPU. Clock APIs and profilers change execution cost. These traces cannot qualify timing or establish a speedup.

The diagnostic worker is named `StemgenRT inference`. `STEMGENRT_TRACE_CLOCK` records a system UTC reading between two steady-clock readings, relative to callback zero. An event at steady time E has initial UTC bounds `[UTC + E - after, UTC + E - before]`. This is an approximate bridge for finding a region in Instruments; later wall-clock adjustments are not tracked.

Use Apple's [Audio System Trace workflow](https://developer.apple.com/documentation/audiotoolbox/analyzing-audio-performance-with-instruments) and inspect Thread State Trace, System Load, Virtual Memory Trace and Thermal State around a late hop. Confirm whether the worker was running, runnable, or blocked before attributing the delay. The synthetic paced host does not supply a DAW Audio Workgroup or exercise Core Audio I/O. Installed AU/VST3 capture and fresh uninstrumented M4/M4 Pro tests remain separate requirements.
