#!/bin/bash
# Fail-closed target-Mac machine qualification for the hardened cropped1024 runtime.
# This script never installs, publishes, tags, or promotes a plug-in.

set -euo pipefail

EXPECTED_ORT_VERSION="1.26.0"
EXPECTED_ORT_ARCHIVE_BYTES="31717869"
EXPECTED_ORT_ARCHIVE_SHA256="7a1280bbb1701ea514f71828765237e7896e0f2e1cd332f1f70dbd5c3e33aca3"
EXPECTED_ORT_ARCHIVE_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-osx-arm64-1.26.0.tgz"
EXPECTED_CPM_SHA256="78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791"
EXPECTED_JUCE_COMMIT="51a8a6d7aeae7326956d747737ccf1575e61e209"
EXPECTED_GOOGLETEST_COMMIT="6910c9d9165801d8827d628cb72eb7ea9dd538c5"
EXPECTED_CALLBACKS="10000"
EXPECTED_WARMUP_CALLBACKS="100"
MAX_POST_WARMUP_RSS_GROWTH_KIB="131072"
MAX_END_RSS_GROWTH_KIB="8192"

usage() {
    cat <<'EOF'
Usage: ./scripts/qualify-macos.sh NEW_EVIDENCE_DIRECTORY

Runs a fresh arm64/macOS 14 Release build, the complete test suite, strict AU
and VST3 bundle verification, direct ORT controls, and the 10,000-callback
paced asynchronous plug-in test. The evidence directory must not already
exist and must be outside the source tree.

This is machine evidence only. A passing run remains unpromoted until the
installed AU/VST3 passes DAW testing and the user explicitly approves the
blind listening comparison.
EOF
}

fail() {
    printf '%s\n' "STEMGENRT_MAC_QUALIFICATION: FAIL: $*" >&2
    exit 1
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

[[ "$(uname -s)" == "Darwin" ]] ||
    fail "native macOS is required; WSL/Linux evidence cannot qualify the plug-in"
[[ "$(uname -m)" == "arm64" ]] ||
    fail "a native Apple Silicon arm64 shell is required"

for command_name in awk cmake codesign cp curl date find git grep lipo ninja otool pmset ps shasum sort sw_vers sysctl xcodebuild xcrun; do
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "required command is unavailable: $command_name"
done
MACOS_MAJOR="$(sw_vers -productVersion | awk -F. '{print $1}')"
[[ "$MACOS_MAJOR" =~ ^[0-9]+$ && "$MACOS_MAJOR" -ge 14 ]] ||
    fail "macOS 14 or newer is required"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
CONTRACT_VALUES="$(cmake -P cmake/PrintModelContract.cmake 2>&1)" ||
    fail "model contract could not be read"
while IFS='=' read -r key value; do
    case "$key" in
        model_sha256) EXPECTED_MODEL_SHA256="$value" ;;
        model_bytes) EXPECTED_MODEL_BYTES="$value" ;;
        sample_rate) EXPECTED_SAMPLE_RATE="$value" ;;
        callback_samples) EXPECTED_CALLBACK_SAMPLES="$value" ;;
        pdc_samples) EXPECTED_PDC_SAMPLES="$value" ;;
        *) fail "unexpected model contract field: $key" ;;
    esac
done <<< "$CONTRACT_VALUES"

EVIDENCE_ARGUMENT="$1"
case "$EVIDENCE_ARGUMENT" in
    ""|-*) fail "evidence directory must be a nonempty path not beginning with '-'" ;;
esac
EVIDENCE_PARENT="$(dirname "$EVIDENCE_ARGUMENT")"
EVIDENCE_NAME="$(basename "$EVIDENCE_ARGUMENT")"
[[ -d "$EVIDENCE_PARENT" ]] ||
    fail "evidence-directory parent does not exist: $EVIDENCE_PARENT"
EVIDENCE_PARENT="$(cd "$EVIDENCE_PARENT" && pwd -P)"
EVIDENCE_DIR="$EVIDENCE_PARENT/$EVIDENCE_NAME"
[[ ! -e "$EVIDENCE_DIR" && ! -L "$EVIDENCE_DIR" ]] ||
    fail "refusing to reuse or overwrite evidence: $EVIDENCE_DIR"
EVIDENCE_SIDECAR="$EVIDENCE_DIR.SHA256SUMS.sha256"
[[ ! -e "$EVIDENCE_SIDECAR" && ! -L "$EVIDENCE_SIDECAR" ]] ||
    fail "refusing to overwrite evidence sidecar: $EVIDENCE_SIDECAR"
case "$EVIDENCE_DIR/" in
    "$PROJECT_ROOT/"*) fail "evidence directory must be outside the source tree" ;;
esac

[[ ! -e build-release && ! -L build-release ]] ||
    fail "build-release already exists; qualification requires a fresh source/build tree"
[[ ! -e libs && ! -L libs ]] ||
    fail "libs already exists; qualification requires freshly authenticated dependencies"

umask 077
mkdir "$EVIDENCE_DIR"
mkdir "$EVIDENCE_DIR/logs"
QUALIFICATION_START_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
QUALIFICATION_START_EPOCH="$(date '+%s')"
QUALIFICATION_COMPLETE=0
qualification_exit() {
    local status=$?
    trap - EXIT
    if [[ "$QUALIFICATION_COMPLETE" -ne 1 ]]; then
        printf '%s\n' "status=failed_or_interrupted" > "$EVIDENCE_DIR/STATUS"
    fi
    exit "$status"
}
trap qualification_exit EXIT

unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS OBJCFLAGS ARCHFLAGS SDKROOT \
    DEVELOPER_DIR \
    MACOSX_DEPLOYMENT_TARGET CMAKE_TOOLCHAIN_FILE CPM_SOURCE_CACHE \
    CODESIGN_ALLOCATE DYLD_INSERT_LIBRARIES DYLD_LIBRARY_PATH

HANDOFF_MANIFEST_SHA256="none"
HANDOFF_ARCHIVE_SHA256="none"
if [[ -f HANDOFF_CONTENTS.sha256 && ! -L HANDOFF_CONTENTS.sha256 ]]; then
    HANDOFF_ARCHIVE_SHA256="${STEMGENRT_HANDOFF_ARCHIVE_SHA256:-}"
    [[ "$HANDOFF_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
        fail "sealed handoff requires STEMGENRT_HANDOFF_ARCHIVE_SHA256 from the externally verified SHA256SUMS"
    if ! shasum -a 256 -c HANDOFF_CONTENTS.sha256 \
        > "$EVIDENCE_DIR/logs/handoff-manifest-check.log" 2>&1; then
        tail -n 80 "$EVIDENCE_DIR/logs/handoff-manifest-check.log" >&2 || true
        fail "sealed source handoff manifest verification failed"
    fi
    cp HANDOFF_CONTENTS.sha256 "$EVIDENCE_DIR/input-handoff-contents.sha256"
    cp SOURCE_PROVENANCE.txt "$EVIDENCE_DIR/input-source-provenance.txt"
    printf '%s  cropped1024-11ms-source.tar.gz\n' "$HANDOFF_ARCHIVE_SHA256" \
        > "$EVIDENCE_DIR/input-handoff-archive.sha256"
    HANDOFF_MANIFEST_SHA256="$(shasum -a 256 HANDOFF_CONTENTS.sha256 | awk '{print $1}')"
    printf '%s\n' "sealed_archive_manifest" > "$EVIDENCE_DIR/source-provenance.txt"
elif [[ -d .git || -f .git ]] && command -v git >/dev/null 2>&1; then
    [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] ||
        fail "source tree is dirty; qualify a clean checkout or a sealed handoff archive"
    git rev-parse --verify HEAD > "$EVIDENCE_DIR/source-provenance.txt"
else
    fail "source is neither a clean Git checkout nor a verified sealed handoff archive"
fi

MODEL_PATH="$PROJECT_ROOT/model/model.onnx"
if [[ -d .git || -f .git ]] && command -v git >/dev/null 2>&1; then
    if command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1; then
        git lfs pull --include=model/model.onnx \
            > "$EVIDENCE_DIR/logs/git-lfs-pull.log" 2>&1
    fi
    git rev-parse HEAD > "$EVIDENCE_DIR/git-head.txt" 2>/dev/null || true
    git status --porcelain=v1 > "$EVIDENCE_DIR/git-status.txt" 2>/dev/null || true
    git diff --binary > "$EVIDENCE_DIR/source.patch" 2>/dev/null || true
fi

[[ -f "$MODEL_PATH" && ! -L "$MODEL_PATH" ]] ||
    fail "model/model.onnx must be a regular file"
MODEL_BYTES="$(wc -c < "$MODEL_PATH" | tr -d '[:space:]')"
MODEL_SHA256="$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')"
[[ "$MODEL_BYTES" == "$EXPECTED_MODEL_BYTES" ]] ||
    fail "model byte count changed: $MODEL_BYTES"
[[ "$MODEL_SHA256" == "$EXPECTED_MODEL_SHA256" ]] ||
    fail "model SHA-256 changed: $MODEL_SHA256"

# Bind every source input before dependency download or build output exists.
if [[ -d .git || -f .git ]] && command -v git >/dev/null 2>&1; then
    git ls-files -co --exclude-standard | LC_ALL=C sort |
        while IFS= read -r relative_path; do
            [[ -n "$relative_path" && -f "$relative_path" ]] || continue
            shasum -a 256 "$relative_path"
        done > "$EVIDENCE_DIR/source-manifest.sha256"
else
    find . \( -type f -o -type l \) \
        ! -path './libs/*' ! -path './build/*' ! -path './build-release/*' \
        ! -path './.git/*' -print | LC_ALL=C sort |
        while IFS= read -r relative_path; do
            shasum -a 256 "$relative_path"
        done > "$EVIDENCE_DIR/source-manifest.sha256"
fi

{
    printf 'uname='; uname -a
    sw_vers
    printf 'translated='; sysctl -in sysctl.proc_translated 2>/dev/null || printf '0\n'
    printf 'hardware_model='; sysctl -n hw.model
    printf 'cpu_brand='; sysctl -n machdep.cpu.brand_string
    printf 'logical_cpus='; sysctl -n hw.logicalcpu
    printf 'physical_cpus='; sysctl -n hw.physicalcpu
    printf 'memory_bytes='; sysctl -n hw.memsize
    printf 'cmake='; cmake --version | head -n 1
    printf 'ninja='; ninja --version
    printf 'xcode='; xcodebuild -version | tr '\n' ' '; printf '\n'
    printf 'clang='; xcrun clang --version | head -n 1
    printf 'macos_sdk_path='; xcrun --sdk macosx --show-sdk-path
    printf 'macos_sdk_version='; xcrun --sdk macosx --show-sdk-version
} > "$EVIDENCE_DIR/host.txt"

capture_power_state() {
    local phase="$1"
    local battery_path="$EVIDENCE_DIR/power-${phase}.txt"
    local settings_path="$EVIDENCE_DIR/power-settings-${phase}.txt"
    local thermal_path="$EVIDENCE_DIR/thermal-${phase}.txt"
    local foundation_thermal_path="$EVIDENCE_DIR/foundation-thermal-${phase}.txt"

    pmset -g batt > "$battery_path"
    pmset -g > "$settings_path"
    pmset -g therm > "$thermal_path" 2>&1 || true
    xcrun swift -e \
        'import Foundation; let p = ProcessInfo.processInfo; print(p.thermalState.rawValue, p.isLowPowerModeEnabled ? 1 : 0)' \
        > "$foundation_thermal_path"

    grep -q 'AC Power' "$battery_path" ||
        fail "target Mac must remain connected to AC power ($phase)"
    if grep -Eq '^[[:space:]]*lowpowermode[[:space:]]+' "$settings_path"; then
        awk '$1 == "lowpowermode" && $2 != 0 { exit 1 }' "$settings_path" ||
            fail "Low Power Mode must be disabled ($phase)"
    fi
    awk '
        /(CPU_Speed_Limit|Scheduler_Limit)/ {
            value = $0
            sub(/^.*=/, "", value)
            gsub(/[[:space:]]/, "", value)
            if (value ~ /^[0-9]+$/ && (value + 0) < 100) exit 1
        }
        ' "$thermal_path" ||
        fail "macOS reported CPU or scheduler thermal throttling ($phase)"
    [[ "$(awk '{ print $1; exit }' "$foundation_thermal_path")" == "0" ]] ||
        fail "Foundation reported a non-nominal thermal state ($phase)"
    [[ "$(awk '{ print $2; exit }' "$foundation_thermal_path")" == "0" ]] ||
        fail "Foundation reported Low Power Mode enabled ($phase)"
}

capture_power_state before

run_stage() {
    local stage_name="$1"
    shift
    local log_path="$EVIDENCE_DIR/logs/${stage_name}.log"
    printf '%s\n' "Running $stage_name..."
    if ! "$@" > "$log_path" 2>&1; then
        tail -n 120 "$log_path" >&2 || true
        fail "$stage_name failed; retained log: $log_path"
    fi
}

grep -Fq "VERSION=\"$EXPECTED_ORT_VERSION\"" scripts/download-onnxruntime.sh ||
    fail "download helper ONNX Runtime version changed"
grep -Fq "ARCHIVE_BYTES=\"$EXPECTED_ORT_ARCHIVE_BYTES\"" scripts/download-onnxruntime.sh ||
    fail "download helper ONNX Runtime archive size changed"
grep -Fq "ARCHIVE_SHA256=\"$EXPECTED_ORT_ARCHIVE_SHA256\"" scripts/download-onnxruntime.sh ||
    fail "download helper ONNX Runtime archive hash changed"
run_stage download-onnxruntime ./scripts/download-onnxruntime.sh
[[ "$(tr -d '[:space:]' < libs/onnxruntime/VERSION_NUMBER)" == "$EXPECTED_ORT_VERSION" ]] ||
    fail "ONNX Runtime version mismatch"
{
    printf 'version=%s\n' "$EXPECTED_ORT_VERSION"
    printf 'archive_url=%s\n' "$EXPECTED_ORT_ARCHIVE_URL"
    printf 'archive_bytes=%s\n' "$EXPECTED_ORT_ARCHIVE_BYTES"
    printf 'archive_sha256=%s\n' "$EXPECTED_ORT_ARCHIVE_SHA256"
} > "$EVIDENCE_DIR/onnxruntime-download.txt"
shasum -a 256 libs/onnxruntime/VERSION_NUMBER \
    libs/onnxruntime/include/onnxruntime_c_api.h \
    libs/onnxruntime/lib/libonnxruntime.dylib \
    > "$EVIDENCE_DIR/onnxruntime-sdk.sha256"

run_stage configure cmake --preset release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cp build-release/CMakeCache.txt "$EVIDENCE_DIR/cmake-cache.txt"
CPM_SHA256="$(shasum -a 256 libs/cpm/CPM_0.40.8.cmake | awk '{print $1}')"
JUCE_COMMIT="$(git -C libs/juce rev-parse HEAD)"
GOOGLETEST_COMMIT="$(git -C libs/googletest rev-parse HEAD)"
[[ "$CPM_SHA256" == "$EXPECTED_CPM_SHA256" ]] ||
    fail "CPM.cmake source hash changed: $CPM_SHA256"
[[ "$JUCE_COMMIT" == "$EXPECTED_JUCE_COMMIT" ]] ||
    fail "JUCE source commit changed: $JUCE_COMMIT"
[[ "$GOOGLETEST_COMMIT" == "$EXPECTED_GOOGLETEST_COMMIT" ]] ||
    fail "GoogleTest source commit changed: $GOOGLETEST_COMMIT"
[[ -z "$(git -C libs/juce status --porcelain=v1 --untracked-files=no)" ]] ||
    fail "JUCE tracked source changed during configure"
[[ -z "$(git -C libs/googletest status --porcelain=v1 --untracked-files=no)" ]] ||
    fail "GoogleTest tracked source changed during configure"
{
    printf 'cpm_sha256=%s\n' "$CPM_SHA256"
    printf 'juce_commit=%s\n' "$JUCE_COMMIT"
    printf 'googletest_commit=%s\n' "$GOOGLETEST_COMMIT"
} > "$EVIDENCE_DIR/dependency-sources.txt"
run_stage build cmake --build --preset release --parallel
run_stage ctest ctest --preset release
run_stage verify-bundles cmake --build --preset release \
    --target StemgenRT_VerifyMacBundles

TEST_BINARY="$PROJECT_ROOT/build-release/test/AudioPluginTest"
[[ -x "$TEST_BINARY" ]] || fail "Release test binary is missing: $TEST_BINARY"
REQUIRED_E2E_FILTER="CroppedModelParityTest.*:InferenceQueueTest.ResetDuringInFlightRunDiscardsStaleOutputAndKeepsOnePreroll:InferenceQueueTest.RepeatedResetPublicationRacesDoNotLoseCurrentEpochOrStall:ProcessBlockTest.ProducesNonSilentOutputWithLoadedModel:AudioQualityTest.NonFiniteInputResetsAndProducesFiniteLosslessOutput:TransportUnderrunE2ETest.StoppedTransportDoesNotReportUnderrunsAndPlaybackRecovers"
run_stage required-e2e "$TEST_BINARY" --gtest_filter="$REQUIRED_E2E_FILTER"
if grep -q '\[  SKIPPED \]' "$EVIDENCE_DIR/logs/required-e2e.log"; then
    fail "a required reset/output/transport E2E test was skipped"
fi
run_stage direct-ort-control "$TEST_BINARY" --gtest_also_run_disabled_tests \
    --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuPerHop
run_stage thread-sweep "$TEST_BINARY" --gtest_also_run_disabled_tests \
    --gtest_filter=OrtStreamingRuntimeTest.DISABLED_BenchmarkStatefulCpuIntraOpThreadSweep

capture_power_state pre-soak
PACED_LOG="$EVIDENCE_DIR/logs/paced-async-10000.log"
RSS_LOG="$EVIDENCE_DIR/paced-rss-kib.tsv"
: > "$RSS_LOG"
printf '%s\n' "Running paced-async-10000 (about 117 seconds)..."
env STEMGENRT_QUALIFICATION_CALLBACKS="$EXPECTED_CALLBACKS" \
    "$TEST_BINARY" --gtest_also_run_disabled_tests \
    --gtest_filter=RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenAsyncRealtimePaced \
    > "$PACED_LOG" 2>&1 &
PACED_PID=$!
RSS_SAMPLE=0
while kill -0 "$PACED_PID" 2>/dev/null; do
    RSS_KIB="$(ps -o rss= -p "$PACED_PID" 2>/dev/null | awk '{print $1}')"
    if [[ "$RSS_KIB" =~ ^[0-9]+$ ]]; then
        printf '%s\t%s\n' "$RSS_SAMPLE" "$RSS_KIB" >> "$RSS_LOG"
    fi
    RSS_SAMPLE=$((RSS_SAMPLE + 1))
    sleep 1
done
if wait "$PACED_PID"; then
    PACED_STATUS=0
else
    PACED_STATUS=$?
fi
if [[ "$PACED_STATUS" -ne 0 ]]; then
    tail -n 160 "$PACED_LOG" >&2 || true
    fail "paced asynchronous qualification failed"
fi
capture_power_state after

SUMMARY_COUNT="$(grep -c '^STEMGENRT_QUALIFICATION_SUMMARY ' "$PACED_LOG" || true)"
[[ "$SUMMARY_COUNT" == "1" ]] ||
    fail "paced log must contain exactly one qualification summary"
PACED_SUMMARY="$(grep '^STEMGENRT_QUALIFICATION_SUMMARY ' "$PACED_LOG")"
[[ "$PACED_SUMMARY" == *" status=pass "* ]] ||
    fail "paced summary did not pass"
[[ "$PACED_SUMMARY" == *" measured_callbacks=${EXPECTED_CALLBACKS} "* ]] ||
    fail "paced summary measured callback count changed"
[[ "$PACED_SUMMARY" == *" warmup_callbacks=${EXPECTED_WARMUP_CALLBACKS} "* ]] ||
    fail "paced summary warmup callback count changed"
[[ "$PACED_SUMMARY" == *" callback_samples=${EXPECTED_CALLBACK_SAMPLES} "* ]] ||
    fail "paced summary callback size changed"
[[ "$PACED_SUMMARY" == *" sample_rate=${EXPECTED_SAMPLE_RATE} "* ]] ||
    fail "paced summary sample rate changed"
[[ "$PACED_SUMMARY" == *" pdc_samples=${EXPECTED_PDC_SAMPLES} "* ]] ||
    fail "paced summary PDC changed"
[[ "$PACED_SUMMARY" == *" due_boundary_misses=0 "* ]] ||
    fail "paced summary contains a due-boundary miss"
[[ "$PACED_SUMMARY" == *" worker_priority=applied "* ]] ||
    fail "inference worker priority was not applied"
[[ "$PACED_SUMMARY" == *" callback_priority=applied "* ]] ||
    fail "synthetic host callback priority was not applied"

RSS_ROWS="$(wc -l < "$RSS_LOG" | tr -d '[:space:]')"
[[ "$RSS_ROWS" -ge 20 ]] || fail "insufficient RSS samples during paced test"
awk -v maximum_growth="$MAX_POST_WARMUP_RSS_GROWTH_KIB" \
    -v maximum_end_growth="$MAX_END_RSS_GROWTH_KIB" '
    NR == 6 { baseline = $2; maximum = $2 }
    NR >= 6 { if ($2 > maximum) maximum = $2; final = $2 }
    END {
      if (baseline == "" || maximum == "" || final == "") exit 2
      peak_growth = maximum - baseline
      end_growth = final - baseline
      printf("baseline_kib=%d\npeak_kib=%d\nend_kib=%d\npeak_growth_kib=%d\nend_growth_kib=%d\n",
             baseline, maximum, final, peak_growth, end_growth)
      if (peak_growth > maximum_growth || end_growth > maximum_end_growth) exit 1
    }
    ' "$RSS_LOG" > "$EVIDENCE_DIR/memory-summary.txt" ||
    fail "paced process memory growth exceeded the declared gates"

AU_BUNDLE="$PROJECT_ROOT/build-release/plugin/AudioPlugin_artefacts/Release/AU/StemgenRT.component"
VST3_BUNDLE="$PROJECT_ROOT/build-release/plugin/AudioPlugin_artefacts/Release/VST3/StemgenRT.vst3"
for bundle in "$AU_BUNDLE" "$VST3_BUNDLE"; do
    [[ -d "$bundle" ]] || fail "verified bundle is missing: $bundle"
    cmake -DSTEMGENRT_BUNDLE_PATH="$bundle" -DSTEMGENRT_VERIFY_ONLY=ON \
        -P "$PROJECT_ROOT/cmake/SignAndVerifyAppleBundle.cmake" \
        >> "$EVIDENCE_DIR/logs/verify-bundles-final.log" 2>&1 ||
        fail "final bundle verification failed: $bundle"
done

if ! (
    cd "$PROJECT_ROOT"
    shasum -a 256 -c "$EVIDENCE_DIR/source-manifest.sha256"
) > "$EVIDENCE_DIR/logs/source-manifest-final-check.log" 2>&1; then
    tail -n 80 "$EVIDENCE_DIR/logs/source-manifest-final-check.log" >&2 || true
    fail "source bytes changed during qualification"
fi

shasum -a 256 "$TEST_BINARY" \
    "$AU_BUNDLE/Contents/MacOS/StemgenRT" \
    "$AU_BUNDLE/Contents/Frameworks/libonnxruntime.dylib" \
    "$AU_BUNDLE/Contents/Resources/model.onnx" \
    "$VST3_BUNDLE/Contents/MacOS/StemgenRT" \
    "$VST3_BUNDLE/Contents/Frameworks/libonnxruntime.dylib" \
    "$VST3_BUNDLE/Contents/Resources/model.onnx" \
    > "$EVIDENCE_DIR/built-artifacts.sha256"

SOURCE_MANIFEST_SHA256="$(shasum -a 256 "$EVIDENCE_DIR/source-manifest.sha256" | awk '{print $1}')"
QUALIFICATION_END_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
QUALIFICATION_END_EPOCH="$(date '+%s')"
QUALIFICATION_ELAPSED_SECONDS=$((QUALIFICATION_END_EPOCH - QUALIFICATION_START_EPOCH))
{
    printf '%s\n' \
        "status=machine_pass_listening_pending_unpromoted" \
        "start_utc=$QUALIFICATION_START_UTC" \
        "end_utc=$QUALIFICATION_END_UTC" \
        "elapsed_seconds=$QUALIFICATION_ELAPSED_SECONDS" \
        "model_sha256=$MODEL_SHA256" \
        "model_bytes=$MODEL_BYTES" \
        "onnxruntime_version=$EXPECTED_ORT_VERSION" \
        "onnxruntime_archive_sha256=$EXPECTED_ORT_ARCHIVE_SHA256" \
        "cpm_sha256=$CPM_SHA256" \
        "juce_commit=$JUCE_COMMIT" \
        "googletest_commit=$GOOGLETEST_COMMIT" \
        "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
        "handoff_contents_manifest_sha256=$HANDOFF_MANIFEST_SHA256" \
        "handoff_archive_sha256=$HANDOFF_ARCHIVE_SHA256" \
        "sample_rate=$EXPECTED_SAMPLE_RATE" \
        "callback_samples=$EXPECTED_CALLBACK_SAMPLES" \
        "pdc_samples=$EXPECTED_PDC_SAMPLES" \
        "warmup_callbacks=$EXPECTED_WARMUP_CALLBACKS" \
        "measured_callbacks=$EXPECTED_CALLBACKS" \
        "$PACED_SUMMARY"
    cat "$EVIDENCE_DIR/memory-summary.txt"
} > "$EVIDENCE_DIR/qualification-summary.txt"

cat > "$EVIDENCE_DIR/NEXT_STEPS.md" <<EOF
# Cropped1024 target-Mac follow-up

Machine qualification passed, but this candidate is still unpromoted.

1. Install the exact sealed Release bundles from this source tree:

       cd "$PROJECT_ROOT"
       ./scripts/install-plugins.sh --release
       killall AudioComponentRegistrar || true
       auval -v aufx Stem Swee

2. Fully restart the DAW. At 44.1 kHz / 256 samples, confirm the loaded plug-in
   reports 512 samples of PDC, worker priority Applied, zero due-boundary
   misses, zero dry fallback, and no timing warning during representative load.

3. Listen specifically for low-frequency continuity and 172.27 Hz hop buzz,
   hop seams, reset/seek behavior, the two-callback stop tail, and fallback
   transitions. Record pass/fail notes without changing this evidence directory.

4. Compare the installed plugin against the accepted Raw L1 +250 research
   audition. Record the DAW, sample rate, buffer, hardware and listening result.
   This script records machine evidence; it does not publish a release.
EOF

printf '%s\n' "status=machine_pass_listening_pending_unpromoted" > "$EVIDENCE_DIR/STATUS"
(
    cd "$EVIDENCE_DIR"
    find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort |
        while IFS= read -r evidence_file; do
            shasum -a 256 "$evidence_file"
        done > SHA256SUMS
)
EVIDENCE_MANIFEST_SHA256="$(shasum -a 256 "$EVIDENCE_DIR/SHA256SUMS" | awk '{print $1}')"
printf '%s  %s/SHA256SUMS\n' "$EVIDENCE_MANIFEST_SHA256" "$EVIDENCE_NAME" \
    > "$EVIDENCE_SIDECAR"

QUALIFICATION_COMPLETE=1
printf '%s\n' \
    "STEMGENRT_MAC_QUALIFICATION: MACHINE PASS" \
    "Evidence: $EVIDENCE_DIR" \
    "Evidence manifest SHA-256: $EVIDENCE_MANIFEST_SHA256" \
    "Sidecar: $EVIDENCE_SIDECAR" \
    "Status: listening pending, unpromoted"
