# c236 audition handoff

This detached worktree is the authenticated, unpromoted c236 audition tree. It
is based exactly on stemgen-rt commit
`8b0f2a041aaf546637f89559324f4c6d362611c1`; the checked c236 ONNX and final
receipt-bound contract are installed. No 512-hop qualification was transferred.
It may now be built and auditioned, but must not be promoted or presented as
production-ready before target-Mac qualification and listening approval.

## Qualified audition ABI

- sample rate: 44,100 Hz
- graph hop and host-visible PDC: 256 samples
- analysis window: 1,024 past-plus-current samples
- inputs: `audio_chunk [1,2,256]`, `analysis_history [1,2,768]`,
  `fusion_hidden [2,1,1000]`, `emitted_db_history [1,4,2048]`
- outputs: `separated_chunk [1,4,2,256]`,
  `next_analysis_history [1,2,768]`, `next_fusion_hidden [2,1,1000]`,
  `next_emitted_db_history [1,4,2048]`
- reset: positive float32 zero for all three state tensors
- alignment: valid current-input-chunk output on callback zero; no pre-roll,
  future context, output-delay hop, or flush
- source order: Drums, Bass, Vocals, Other; Other is re-derived from the raw
  aligned mixture after inference
- runtime scheduling: one asynchronous queue hop; the macOS automatic ORT
  intra-op cap remains two threads

## Activated artifact identity

The contract-last substitution completed and replay-authenticated successfully:

- model: `6dd58f05ee6bdf4beadb5df24849320e4f3dec4dbd0c7767fedd0dde2b557a02`
  (111,374,870 bytes)
- checked-export receipt: `a00f2a6c764be9cfb902b871607f40b93b18ed81e39fb1cd10848fe3ded36917`
- final-chain receipt: `04bc07022394d47f403491551bd49428b154b5f1c7006449d10dca700af1b16e`
- qualification-v2: `25456eafb49b3ef428ae860eff927fe548e160107fdbd9fafd688fd6ead96449`
- active contract: `2ebae98946584d12e4337172df9aeffcadcc43b91e900eae3470982582188f59`
- substitution receipt: `9e498f8d056bfd73db8083573b0bf7f9f2ec92799f6ab9edc52c7f9d70f606c6`
- selected update: 100,000; run UUID:
  `024ef8bb-c0f2-4663-ae60-03c4fc4a2b9e`

The checked export ran against pinned ONNX Runtime 1.26.0, matching the target
Mac runtime. Do not weaken the configure-time or runtime identity checks.

## Atomic receipt and model substitution

The checked handoff is installed by `scripts/substitute-c236-audition.py`.
With no arguments (or with `--source-check`) the tool is deliberately inert:
it prints its plan without opening the model, completion bundle, or recovery
chain and without changing this worktree.

Real substitution is possible only in this exact detached workspace and needs
two independently reviewed, nonzero trust anchors: the SHA-256 of the canonical
checked-export receipt and the SHA-256 of the canonical final-chain receipt.
The tool accepts no caller-supplied qualification path. It authenticates the
canonical export completion, obtains the checked quartet from that completion,
obtains the final chain from the authenticated export authority, and derives
qualification-v2 from that final-chain receipt.

```bash
python3 scripts/substitute-c236-audition.py

python3 scripts/substitute-c236-audition.py --apply \
  --export-completion \
    /home/axel/autoresearch/codex/HS-TasNet-latency11-v1-state/artifacts/c236-terminal-recovery-checked-onnx-export-v3/export-complete \
  --expected-export-receipt-sha256 a00f2a6c764be9cfb902b871607f40b93b18ed81e39fb1cd10848fe3ded36917 \
  --expected-final-chain-receipt-sha256 04bc07022394d47f403491551bd49428b154b5f1c7006449d10dca700af1b16e
```

Do not copy hashes out of the same receipt during the apply invocation and call
that independent review. The two values are intentional human authorization
boundaries. The tool holds a root-scoped writer lock, rejects links and
non-exact inventories, retains authenticated file descriptors, stages copied
bytes through unnamed Linux `O_TMPFILE` inodes that become visible only after
their content is complete and durable, retires the three legacy 512-hop
evidence files, and activates the result by exchanging
`cmake/QualifiedModelContract.cmake` last. It writes a
new external `model/C236_AUDITION_SUBSTITUTION.json` plus exact sidecar; the
checked-export receipt itself remains outside plugin resources and its SHA is
not embedded in ONNX metadata.

An ordinary error is rolled back to the exact pending tree. If the process or
machine stops after the durable journal is published, do not edit either tree
state or `.c236-substitution-txn`; run explicit hash-inferred recovery:

```bash
python3 scripts/substitute-c236-audition.py --recover
```

Recovery accepts only the sealed before/after identities in the transaction
journal and refuses unknown live bytes without cleanup. Cleanup first retires
the authenticated transaction directory atomically, then preserves its journal
as a root-level marker until the exact terminal tree and empty retired directory
have both been revalidated; a stop at any cleanup boundary is resumable with
`--recover`. A successful apply is still an unpromoted audition candidate; it
does not build, run, listen, commit, push, or claim target-Mac qualification.

The terminal recovery chain published and independently verified all four final files:

- `model.onnx`
- `model.onnx.sha256`
- `model.onnx.export.json`
- `model.onnx.export.json.sha256`

The substitution replay recomputes both file hashes and checks this exact
export-receipt envelope:

- `schema_version == 3`
- `kind == hs_tasnet_c236_recovery_checked_onnx_export_v3`
- `status == pass`
- `deployment_status == qualified_unpromoted_listening_candidate`
- `gate_pass == true`, `cpu_only == true`, and
  `promotion_performed == false`
- `export.file_sha256` and `export.bytes` match the received `model.onnx`, and
  `export.qualified_publication_snapshot.{file_sha256,bytes}` match them too
- `abi.host_visible_pdc_samples == 256`, `abi.analysis_window_samples == 1024`,
  the exact four-input/four-output ABI above, and `abi.flush_required == false`
- 64-hop bounded eager/ORT parity, reset replay, exact residual Other on every
  checked ONNX hop, and liveness of all three states
- `terminal_authority.terminal_recovery_chain.qualification_v2` has a nonzero
  SHA-256 and `candidate_status == qualified_by_declared_quality_budget`
- a terminal-selected update and a materialization receipt authenticated by the
  export authority; do not substitute a provisional training checkpoint

The qualification record named above is the post-training evaluation result
bound into the final recovery chain. It is distinct from the export receipt's
own schema-2 `hs_tasnet_c236_64_hop_eager_onnx_qualification_v2` graph check;
both must pass.

The tool made the following substitutions as one contract-last transaction:

1. Verify both sidecars independently, ensure the model is a regular,
   self-contained file named `model.onnx`, and reject symlinks or
   `model.onnx.data`.
2. Replace `model/model.onnx` with those already-checked bytes and retire the
   tracked 512-hop evidence in the same atomic transaction.
3. In `cmake/QualifiedModelContract.cmake`, replace only the fail-closed fields
   below. Values come from the receipt unless otherwise noted:

| Contract field | Receipt/artifact source |
| --- | --- |
| `STEMGENRT_QUALIFIED_MODEL_SHA256` | `export.file_sha256`, `export.qualified_publication_snapshot.file_sha256`, and the independently recomputed model SHA-256 |
| `STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE` | `export.bytes`, `export.qualified_publication_snapshot.bytes`, and the independently observed byte size |
| `STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SHA256` | Independently recomputed canonical SHA-256 of `model.onnx.export.json`, equal to `model.onnx.export.json.sha256`; evidence-only, never ONNX metadata |
| `STEMGENRT_QUALIFIED_C236_SELECTED_UPDATE` | `terminal_authority.selected_update` |
| `STEMGENRT_QUALIFIED_C236_RUN_UUID` | `terminal_authority.materialized_candidate.run_uuid` |
| `STEMGENRT_QUALIFIED_C236_CONTRACT_IDENTITY_SHA256` | `terminal_authority.materialized_candidate.contract_identity_sha256` |
| `STEMGENRT_QUALIFIED_C236_STATIC_IDENTITY_SHA256` | `terminal_authority.materialized_candidate.static_identity_sha256` |
| `STEMGENRT_QUALIFIED_C236_SOURCE_CHECKPOINT_SHA256` | `terminal_authority.materialized_candidate.source_checkpoint_sha256` |
| `STEMGENRT_QUALIFIED_C236_MATERIALIZED_ARTIFACT_SHA256` | `terminal_authority.materialized_candidate.candidate_artifact.sha256` |
| `STEMGENRT_QUALIFIED_C236_MATERIALIZATION_RECEIPT_SHA256` | `evaluator_coordination.materialization_receipt_sha256` |
| `STEMGENRT_QUALIFIED_C236_MODEL_STATE_SHA256` | `runtime_identity.model_state_sha256` |
| `STEMGENRT_QUALIFIED_C236_COMPOSITE_RUNTIME_STATE_SHA256` | `runtime_identity.composite_runtime_state_sha256` |
| `STEMGENRT_QUALIFIED_C236_RECOVERY_RECEIPT_SHA256` | `terminal_authority.terminal_run.terminal_recovery.recovery_receipt.sha256` |
| `STEMGENRT_QUALIFIED_C236_FINAL_CHAIN_RECEIPT_SHA256` | `terminal_authority.terminal_recovery_chain.final_chain_bundle.receipt.sha256` |
| `STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_SHA256` | `terminal_authority.terminal_recovery_chain.qualification_v2.sha256` |
| `STEMGENRT_QUALIFIED_C236_QUALIFICATION_V2_CANDIDATE_STATUS` | `terminal_authority.terminal_recovery_chain.qualification_v2.candidate_status` |
| `STEMGENRT_QUALIFIED_C236_EVALUATION_RESULT_BOUND` | Literal `true`, only after the nonzero qualification hash and accepted status above are verified |

The exact fixed source authorities already frozen into this draft are:

- materializer: `5da7c0d97faeabdfb87fbdf3d37e5a2af2847f8a1595fc0467b01a14ea59ca2d`
- shared publication authority: `6fc64af8968199e926f529ac9626f20c5f9fd6c28bcea02ae426353ae19246d2`
- recovery-receipt authority: `fcb8edac6e94512d9be3b98d225541cf08458a3f3e015a139f36ac774bb76aa8`
- terminal-chain authority: `e8d23b5e78f36482e5f60b2e0d4b5b40d5cb91a242882f26a4f85f362a376cc9`
- recovery contract: `a1125772696b9f05d699c9de2f92caaa92cfaa1a58d74b80713ea6d021db4501`

The c236 family, those authorities, exact-c191 payload/head hashes, candidate
kind, ABI, receipt schema/kind/status literals, and metadata literals must not
be relaxed. Every final dynamic hash must match the embedded ONNX metadata,
except `STEMGENRT_QUALIFIED_EXPORT_RECEIPT_SHA256`: embedding that receipt hash
would create a self-reference cycle because the receipt already binds the ONNX
hash. Do not confuse it with the earlier materialization-receipt hash.

4. Validate the staged final contract through
   `test/cmake/C236DraftContractTest.cmake` in final phase. The test accepts
   exact receipt-derived values as read-only overrides, asserts the accepted
   qualification status and `evaluation_result_bound == true`, and preserves
   the same one-field negative mutation coverage in pending and final phases.
5. Run the source/static test, configure/build tests, and check that the runtime
   accepts every metadata entry but rejects one-at-a-time mutations of names,
   shapes, types, state count, alignment, residual index, artifact identity,
   export-evidence identity, and c236/c191 lineage.

The export receipt and its SHA sidecar are qualification evidence, not plugin
resources and not ONNX metadata. Preserve them outside the bundle and record
their exact hash in the audition report.

## Tests before listening

The pre-install transaction suite passed 138/138 before activation. It uses the
pending contract as a synthetic fixture template and therefore is not a
post-install command. For the activated tree, run the contract and source checks:

```bash
cmake -DC236_CONTRACT_PHASE=final \
  -DC236_MODEL_FILE="$PWD/model/model.onnx" \
  -DC236_EXPORT_RECEIPT_FILE=/path/to/model.onnx.export.json \
  -P test/cmake/C236DraftContractTest.cmake
git diff --check
```

The Python suite uses disposable synthetic repositories only. It covers both
mandatory trust anchors, exact authority and metadata mutations, every quartet
member as a symlink/hardlink/FIFO, all contract-last mutation boundaries under
caught exceptions and `SIGKILL`, explicit recovery, unknown-state refusal, and
concurrent-writer exclusion. Its hidden fixture switches are not a real apply
interface and are rejected for this canonical workspace.

After the atomic substitution, run the normal unit suite and the release build.
Confirm callback-zero validity, state progression/reset, gap/epoch handling,
out-of-bounds guards, generic block scheduling, 256-sample PDC, complete-Other
fallback, and exact reconstruction. Do not claim CPU qualification from Linux.

## macOS audition and qualification

On the separate Apple Silicon Mac, use the pinned official ONNX Runtime 1.26.0
and first run `git lfs pull --include=model/model.onnx`. Confirm that the model
is 111,374,870 bytes with SHA-256 `6dd58f05ee6bdf4beadb5df24849320e4f3dec4dbd0c7767fedd0dde2b557a02`,
and record the independently authenticated external checked-export receipt SHA
`a00f2a6c764be9cfb902b871607f40b93b18ed81e39fb1cd10848fe3ded36917`.
The receipt itself is intentionally not a plugin resource. Build and install
the exact received tree, listen specifically for kick/sub buzzing and Other
leakage, and record the commit plus all artifact hashes. If the listening
candidate is accepted, run:

```bash
build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep

STEMGENRT_QUALIFICATION_CALLBACKS=10000 \
  build-release/test/AudioPluginTest --gtest_also_run_disabled_tests \
  --gtest_filter=RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenRealtimePaced
```

The paced summary must identify 256 callback samples, 44.1 kHz, the 5.80 ms
deadline, applied worker priority, finite distinct stems, reconstruction error
at or below `1e-6`, and zero deadline misses, underruns, queue/ring drops, and
unsafe callbacks. The thread sweep may confirm or replace the preserved
two-thread macOS default; any change requires rerunning the paced test.
