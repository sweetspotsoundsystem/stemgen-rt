# Changelog

## 0.8.0 (unreleased)

- Run Bass on the existing queue worker and Drums/Vocals on two persistent
  helpers by default, both in source builds and CI packages. Keep one ORT thread
  per session and retain the single-worker diagnostic control.
- Add macOS playback scheduling and AU workgroup handling; use Windows atomic
  helper wakeups, a high-resolution worker timer and MMCSS Pro Audio scheduling.
  Model weights, streaming states, host IDs and reported latency are unchanged.
- Validate and stage Windows installations before replacing an existing bundle.
  Verify all copied files and restore the previous bundle if replacement fails.
- Include the pinned ONNX Runtime SDK's license and third-party notices in
  plugin bundles and verify their presence during packaging and installation.
- Keep legacy single-worker tracing comparisons explicit, and require three
  workers in default-path Mac qualification.
- Bundle the current matching model/training/export source, including partition
  export validation, and distinguish current runtime evidence from historical
  0.7.0 results.

M4 sustained and M1 Pro quick standalone comparisons passed. The tested Ryzen
5 5500 still had playback underruns at both 128- and 256-sample host buffers;
Windows is not qualified for zero-underrun playback on that machine. The M4
Ableton stability report is a user observation. See
[platform evidence](docs/parallel-workers.md) for exact exposure and limits.
