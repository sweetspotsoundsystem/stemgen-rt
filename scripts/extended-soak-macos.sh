#!/bin/bash
# Supplementary two-by-30-minute synthetic-host measurement on a native Mac.
# Uses an existing Release test build. It does not install or publish a plugin.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash run_latency58_macos_extended_soak.sh STEMGENRT_SOURCE NEW_OUTPUT_DIRECTORY

Run the existing paced callback/worker test twice, each with at least 30 minutes
of measured callbacks. Requires native arm64 macOS and an existing Release
AudioPluginTest build. Preserve the build's correctness/identity evidence too.
The raw logs retain startup and measured fallback counters separately.
This synthetic-host test does not establish installed-AU or DAW acceptance.
EOF
    exit 0
fi
[[ $# -eq 2 ]] || { echo "Expected source and new output directories" >&2; exit 2; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
    echo "This measurement requires a native arm64 Mac; no timing test was run." >&2
    exit 2
}
SOAK_SOURCE="$(cd "$1" && pwd -P)"
[[ ! -e "$2" && ! -L "$2" ]] || { echo "Output already exists" >&2; exit 2; }
SOAK_PARENT="$(cd "$(dirname "$2")" && pwd -P)"
SOAK_OUTPUT="$SOAK_PARENT/$(basename "$2")"
SOAK_BINARY="$SOAK_SOURCE/build-release/test/AudioPluginTest"
[[ -x "$SOAK_BINARY" ]] || { echo "Release AudioPluginTest is missing" >&2; exit 2; }
[[ -z "$(git -C "$SOAK_SOURCE" status --porcelain=v1 --untracked-files=no)" ]] || {
    echo "Commit the measured source changes first so the run has an exact source identity." >&2
    exit 2
}
[[ "$(lipo -archs "$SOAK_BINARY")" == arm64 ]] || {
    echo "Expected the native arm64 Release test binary" >&2; exit 2;
}

# Read identity and geometry from the same authoritative contract as packaging.
SOAK_CONTRACT="$(cd "$SOAK_SOURCE" && cmake -P cmake/PrintModelContract.cmake 2>&1)"
SOAK_MODEL_SHA=""
SOAK_MODEL_BYTES=""
SOAK_RATE=""
SOAK_BLOCK=""
SOAK_PDC=""
while IFS='=' read -r key value; do
    case "$key" in
        model_sha256) SOAK_MODEL_SHA="$value" ;;
        model_bytes) SOAK_MODEL_BYTES="$value" ;;
        sample_rate) SOAK_RATE="$value" ;;
        callback_samples) SOAK_BLOCK="$value" ;;
        pdc_samples) SOAK_PDC="$value" ;;
        *) echo "Unrecognized model contract field: $key" >&2; exit 2 ;;
    esac
done <<< "$SOAK_CONTRACT"
[[ "$SOAK_RATE" == 44100 && "$SOAK_BLOCK" == 128 && "$SOAK_PDC" == 256 ]] || {
    echo "The fixed hop128/44.1 kHz/256-sample contract changed" >&2; exit 2;
}
[[ "$SOAK_MODEL_SHA" =~ ^[0-9a-f]{64}$ && "$SOAK_MODEL_BYTES" =~ ^[0-9]+$ ]] || exit 2
[[ "$(shasum -a 256 "$SOAK_SOURCE/model/model.onnx" | awk '{print $1}')" == "$SOAK_MODEL_SHA" ]] || exit 2
[[ "$(stat -f '%z' "$SOAK_SOURCE/model/model.onnx")" == "$SOAK_MODEL_BYTES" ]] || exit 2
SOAK_CALLBACKS=$(((1800 * SOAK_RATE + SOAK_BLOCK - 1) / SOAK_BLOCK))
[[ "$SOAK_CALLBACKS" == 620157 ]] || exit 2
umask 077
mkdir "$SOAK_OUTPUT"
printf '%s\n' "$SOAK_CONTRACT" > "$SOAK_OUTPUT/model-contract.txt"
git -C "$SOAK_SOURCE" rev-parse HEAD > "$SOAK_OUTPUT/source-commit.txt"
{
    date -u '+observed_utc=%Y-%m-%dT%H:%M:%SZ'
    sw_vers
    sysctl machdep.cpu.brand_string hw.physicalcpu hw.logicalcpu
    printf 'measured_callbacks_per_run=%s\nrepetitions=2\nworker_trace=disabled\n' "$SOAK_CALLBACKS"
    printf 'scope=synthetic_host_not_installed_AU_or_DAW\n'
} > "$SOAK_OUTPUT/machine-and-scope.txt"
(
    cd "$SOAK_SOURCE"
    shasum -a 256 build-release/test/AudioPluginTest model/model.onnx \
        cmake/QualifiedModelContract.cmake test/source/RealtimeStemSanityTest.cpp \
        test/source/PacedQualificationTrace.h plugin/source/InferenceQueue.cpp \
        plugin/source/OnnxRuntime.cpp libs/onnxruntime/lib/libonnxruntime.dylib
) > "$SOAK_OUTPUT/inputs.sha256"
SOAK_RESULT=0
for SOAK_REPETITION in 1 2; do
    SOAK_PREFIX="$SOAK_OUTPUT/repetition-$SOAK_REPETITION"
    pmset -g therm > "$SOAK_PREFIX-thermal-before.txt" 2>&1 || true
    pmset -g custom > "$SOAK_PREFIX-power-before.txt" 2>&1 || true
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$SOAK_PREFIX-started.txt"
    SOAK_CODE=0
    (
        cd "$SOAK_SOURCE"
        env STEMGENRT_QUALIFICATION_CALLBACKS="$SOAK_CALLBACKS" \
            STEMGENRT_TRACE_WORKER=0 STEMGENRT_PACED_ORT_THREADS=0 \
            "$SOAK_BINARY" --gtest_also_run_disabled_tests \
            --gtest_filter=RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenAsyncRealtimePaced \
            "--gtest_output=xml:$SOAK_PREFIX.xml"
    ) > "$SOAK_PREFIX.log" 2>&1 || SOAK_CODE=$?
    printf '%s\n' "$SOAK_CODE" > "$SOAK_PREFIX-exit-code.txt"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$SOAK_PREFIX-finished.txt"
    pmset -g therm > "$SOAK_PREFIX-thermal-after.txt" 2>&1 || true
    # Exact count/geometry and actual exit are required; a missing/skipped
    # gtest or partial log must never turn into a passing long soak.
    SOAK_SUMMARY="$(awk '/^STEMGENRT_QUALIFICATION_SUMMARY /{print}' "$SOAK_PREFIX.log")"
    SOAK_PHASES="$(awk '/^STEMGENRT_PACED_FAILURE_PHASES /{print}' "$SOAK_PREFIX.log")"
    if [[ "$SOAK_CODE" -ne 0 || "$SOAK_SUMMARY" == *$'\n'* || "$SOAK_PHASES" == *$'\n'* ||
          "$SOAK_SUMMARY" != *" status=pass "* ||
          "$SOAK_SUMMARY" != *" measured_callbacks=$SOAK_CALLBACKS "* ||
          "$SOAK_SUMMARY" != *" warmup_callbacks=100 "* ||
          "$SOAK_SUMMARY" != *" ort_intra_op_threads=1 "* ||
          "$SOAK_SUMMARY" != *" ort_intra_op_threads_override=0 "* ||
          "$SOAK_SUMMARY" != *" worker_trace=disabled "* ||
          "$SOAK_SUMMARY" != *" callback_samples=128 sample_rate=44100 pdc_samples=256 "* ||
          "$SOAK_PHASES" != *" measured_underrun_samples=0 "* ||
          "$SOAK_PHASES" != *" measured_due_boundary_misses=0 "* ]]; then
        SOAK_RESULT=1
    fi
    printf 'repetition=%s exit_code=%s\n' "$SOAK_REPETITION" "$SOAK_CODE"
    printf '%s\n%s\n' "$SOAK_PHASES" "$SOAK_SUMMARY"
done
(
    cd "$SOAK_SOURCE"
    shasum -a 256 -c "$SOAK_OUTPUT/inputs.sha256"
) > "$SOAK_OUTPUT/inputs-after-check.txt" 2>&1 || SOAK_RESULT=1
if [[ "$SOAK_RESULT" -eq 0 ]]; then
    printf 'synthetic_host_soak_passed\n' > "$SOAK_OUTPUT/status.txt"
else
    printf 'synthetic_host_soak_failed_or_incomplete\n' > "$SOAK_OUTPUT/status.txt"
fi
exit "$SOAK_RESULT"
