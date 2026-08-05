# c214 hop-256 listening candidate

This is the unpromoted c214 step-100,000 candidate with the exact c191
step-128 causal output correction adapted to its 256-sample parent stream.

## Identity

- ONNX SHA-256: `8262f56503f1f8acf2cdb105822db474de2ec546b5f9dd0521bdc6b06b3a02ab`
- ONNX size: 111,372,674 bytes
- Deploy-checkpoint SHA-256:
  `a348fede94871af5255f8bb945cd0643cc2d7694bf41e7ab78456fe5924e62fa`
- Export qualification SHA-256:
  `a07cbf17bdc7a8653339094fae43e22ce4eaa8432312328b253221ac750dda4f`
- Audition-export source seal SHA-256:
  `ba96c968b58d82974b06b807f4a2f75a7f1b908ada7e151a3a4a129af1ca4449`

## Checked-export result

The CPU-only 64-hop eager/ONNX comparison passed with per-output tolerances:

| Output | Maximum absolute error | Limit |
| --- | ---: | ---: |
| `separated_chunk` | `2.3841858e-7` | `2e-5` |
| `next_analysis_history` | `0` | `2e-5` |
| `next_fusion_hidden` | `6.4432621e-5` | `1e-4` |
| `next_emitted_db_history` | `1.9744039e-7` | `2e-5` |

Maximum mixture-reconstruction error was `7.4505806e-9`. Reset replay was
bit-exact; every state was live; all outputs and states were finite.

The separate fusion-state limit reflects bounded accumulation in an opaque
recurrent state. Audio and waveform-history outputs retain the stricter
`2e-5` limit. Tests prove both the strict audio rejection threshold and the
fusion-state `1e-4` boundary.

## Promotion status

Listening only. The 512-hop qualified fallback remains authoritative. c214
still requires completed full quality qualification, target-Mac thread/timing
tests at the 5.80499 ms deadline, and explicit human listening approval.
