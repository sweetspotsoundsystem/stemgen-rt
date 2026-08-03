# Full c191 correction listening model

This model is for a controlled listening test only. It is not production
qualified.

- Source graph SHA-256: `07557c7756815c0a84960c02faed4becd31413b53e1e4bb5b28177f2d6a97159`
- Output graph SHA-256: `6e817d7a09832072f0d7df4a3cbec0a798804efa4a88fbaabea14a1f70fdcf50`
- Output size: 114,646,323 bytes
- Source checkpoint SHA-256: `a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b`
- Changed initializer: `runtime.head.output_projection.weight`, shape `[4,8,1]`
- Previous projection: Drums `7/128`, Bass `16/128`
- Audition projection: Drums `128/128`, Bass `128/128`
- Previous projection tensor SHA-256: `ea9686f34466d37cd275c974bc7c33e82b40a5524d0b0ba1c2affb1b55583801`
- Audition projection tensor SHA-256: `7f3b594e1ea06b6b5f4fe9f188181e908be249b1c918dc782d204cb7bc2f338b`

The materializer authenticated the source graph, c191 checkpoint, and c193
selected head; proved all 23 non-projection head tensors bit-identical; and
preserved the eight-input/eight-output, seven-state current-chunk ABI. Across
24 deterministic random and 50 Hz stateful hops, eager PyTorch and ONNX agreed
within `9.23872e-7` maximum absolute error. Reset replay was bit-exact, every
state family was live, all outputs were finite, and mixture reconstruction
error was `3.72529e-9`.

The full correction previously failed strict experimental boundary and safety
guards. This branch exists to answer one listening question: does restoring
the intended correction materially improve the audible kick/bass distortion?
Do not promote it without a new quality and safety qualification.
