# Full c191 correction lightweight qualification

This is the exact full-strength c191 step-128 Drums/Bass correction candidate.
The frozen 12-track electronic holdout passed all six agreed quality gates. It
is a production candidate pending one exact-artifact target-Mac 10,000-callback
paced test; do not promote it until that test passes.

- Candidate: `c191-step128-full-correction-128-over-128`
- Candidate ONNX SHA-256: `370d0a8971b405bd9c7f49928ccdea66e5b28fb028f6f5425c9c1ba5dc162f91`
- Candidate ONNX size: 114,646,796 bytes
- Candidate receipt SHA-256: `c9246d27cf9cd1fff69d50b4fbb6d651a009fd1af563ad07025be024ad56b030`
- Listened-to audition SHA-256: `6e817d7a09832072f0d7df4a3cbec0a798804efa4a88fbaabea14a1f70fdcf50`
- c212 scaled source graph SHA-256: `07557c7756815c0a84960c02faed4becd31413b53e1e4bb5b28177f2d6a97159`
- Source checkpoint SHA-256: `a80f65f1f475815181306ef6366d077ce6fa21eebd423994744b2d6e7d2f9f5b`
- Changed initializer versus c212: `runtime.head.output_projection.weight`,
  shape `[4,8,1]`
- Previous projection: Drums `7/128`, Bass `16/128`
- Candidate projection: Drums `128/128`, Bass `128/128`
- Candidate projection tensor SHA-256: `7f3b594e1ea06b6b5f4fe9f188181e908be249b1c918dc782d204cb7bc2f338b`

The metadata seal did not change computation. The candidate and listened-to
audition have byte-identical computational GraphProto data, a byte-identical
metadata-cleared ModelProto, and all 97 initializer TensorProto values in the
same order. Their shared computational graph SHA-256 is
`ddb6f3733f9e65ff60fa0121bbaade071805fbbb4874babffa11789ef27ca6ec`.

The original materializer authenticated the source graph, c191 checkpoint,
and c193 selected head; proved all 23 non-projection head tensors bit-identical;
and preserved the eight-input/eight-output, seven-state current-chunk ABI.
Across 24 deterministic random and 50 Hz stateful hops, eager PyTorch and ONNX
agreed within `9.23872e-7` maximum absolute error. Reset replay was bit-exact,
every state family was live, all outputs were finite, and mixture
reconstruction error was `3.72529e-9`.

## Electronic holdout

The c213 evaluator freshly reproduced c91 exactly, evaluated full c191 on the
same frozen 12 electronic tracks, and compared it with the sealed scaled-c193
result. All six lightweight gates passed:

| Gate versus c91 | Candidate | Delta | Floor |
| --- | ---: | ---: | ---: |
| Aggregate SI-SDR | -0.9194 dB | +0.1813 dB | -0.5 dB |
| Aggregate low-band SI-SDR | -4.2363 dB | +0.2028 dB | -0.5 dB |
| Drums SI-SDR | 5.1927 dB | +0.2585 dB | -1.0 dB |
| Drums low-band SI-SDR | 6.3051 dB | +0.0593 dB | -1.0 dB |
| Bass SI-SDR | -0.6131 dB | -0.1722 dB | -1.0 dB |
| Bass low-band SI-SDR | -0.2936 dB | -0.2420 dB | -1.0 dB |

The complete diagnostic tradeoffs versus c91 were:

| Scope | SI-SDR delta | Low-band delta | Projection-SIR delta |
| --- | ---: | ---: | ---: |
| Aggregate | +0.1813 dB | +0.2028 dB | -0.1989 dB |
| Drums | +0.2585 dB | +0.0593 dB | -1.0135 dB |
| Bass | -0.1722 dB | -0.2420 dB | -0.2893 dB |
| Vocals | +0.4418 dB | +0.2159 dB | +0.7844 dB |
| Other | +0.1972 dB | +0.7780 dB | -0.2772 dB |

Full c191 and scaled c193 were effectively tied on aggregate metrics: full
c191 changed SI-SDR by `+0.00008 dB`, low-band SI-SDR by `+0.00019 dB`, and
projection SIR by `-0.00605 dB`. The complete per-stem comparison lives in
`FULL_CORRECTION_CANDIDATE.json`.

- c213 evaluation SHA-256: `c4f16bb90b9784fa0b34912c344e0899bb5e9a4417fab4f29c7ecf22b41048fd`
- c213 gate SHA-256: `ed49d9ed8882afa543992a12d11657274d3e7fcf78d7fe12fa9289921c5826ae`
- c213 terminal seal SHA-256: `113020dd84e065f77ca734796e446ce57f82ce49bf36470ffbf31492fbd3eed2`

The user heard the compute-identical audition as “much better,” though not
perfect, with no clicks and stable operation. That listening result transfers
to this metadata-only candidate. The remaining qualification step is the
exact candidate hash on the target Mac; no retraining, full14 rerun, thread
sweep, exhaustive boundary campaign, or new exporter campaign is required.
