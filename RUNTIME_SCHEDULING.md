# Runtime scheduling candidate

This branch changes the existing StemgenRT playback worker. It retains the
released graph, weights, 13 states, controls, host IDs, residual Other and
256-sample latency at 128 samples / 44.1 kHz. It is a candidate awaiting sustained
physical-M4/M4 Pro and installed-AU/VST3 qualification.

On macOS, a worker prepared for real-time playback requests a Mach
time-constraint policy with a 128/44100-second period and 2.7 ms computation
estimate. Initial readback failure rejects worker startup during preparation.
The final readback remains available after shutdown. An offline preparation
uses user-interactive QoS. Changes to the host's offline flag take effect at the
next preparation; they do not trigger an OS policy change from processBlock.
Windows and Linux retain their existing priority mechanisms. ORT still uses
one sequential worker with intra/inter-op counts of one and spinning disabled.

The JUCE AU wrapper supplies changes through audioWorkgroupContextChanged.
The callback retains the handle into one of eight preallocated slots and
publishes atomically. It does not join, leave, release an owned handle, allocate,
lock, wait, or wake the worker. The worker consumes the latest published context,
leaves the old group before releasing it, joins the new group, and retires older
handles. It leaves before policy restoration and thread exit. A retained context
survives stop/start because hosts need not repeat an unchanged notification.

If all eight slots are occupied, the callback records an overflow and publishes
an invalidation. The worker leaves obsolete membership and continues with its
real-time policy until a subsequent host context update is accepted. Hosts that
provide no workgroup, including the VST3 path, use the real-time policy alone.
Offline workers retain context changes but do not join a group.

Workgroup status reports `JoinRequested`, not verified native membership:
JUCE 8.0.6 exposes token-provider presence even when a native join fails. Native
macOS lifecycle tests independently probe attachment and detachment, cancellation,
restart and separate instances. The callback allocation guard counts C++
new/delete; it does not intercept allocations inside system libraries. Host
integration follows [Apple's AU auxiliary-thread guidance](https://developer.apple.com/documentation/audiotoolbox/adding-audio-unit-auxiliary-real-time-threads-to-audio-workgroups).

The paced test uses the actual playback default when
`STEMGENRT_PACED_WORKER_REALTIME` is unset. Explicit 0/1 values remain diagnostic
QoS/real-time overrides, and the summary distinguishes them. `qualify-macos.sh`
requires the default worker path and a real-time synthetic callback. The older
comparison/long-soak runner intentionally requests explicit diagnostic policies.
Neither synthetic callback tests nor test-created workgroups qualify installed
DAW behavior. Long playback, offline rendering, transport changes, multiple
instances, actual AU host membership and sustained deadlines remain release gates.
