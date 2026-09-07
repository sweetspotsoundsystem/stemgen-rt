#!/bin/bash
# A serial 2 -> 1 -> 2 thread diagnostic. Never installs or promotes a plugin.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: ./scripts/compare-paced-threads-macos.sh NEW_EVIDENCE_DIRECTORY

Builds the Release test/model target, then runs three 30,000-callback tests
with two, one, and two ORT intra-op threads. Allow about 4.5 minutes after
building. Keep the Mac on power and stop competing CPU work. Each test gets
a fresh processor/session, worker tracing and process CPU-time measurements.

All three logs and exit codes are preserved even when a timing gate fails.
A failed test makes this script exit nonzero after completing the comparison.
Crashes/interruption stop it immediately. This is diagnostic evidence; an
override passing does not qualify the production default or an installed DAW.
EOF
    exit 0
fi

fail() { printf '%s\n' "$*" >&2; exit 2; }
[[ $# -eq 1 ]] || fail "Supply one new evidence directory; see --help."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] ||
    fail "Run this diagnostic in a native Apple Silicon macOS shell."
for command_name in cmake git shasum sysctl sw_vers pmset cmp; do
    command -v "$command_name" >/dev/null || fail "Missing $command_name"
done

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
evidence_parent="$(cd "$(dirname "$1")" && pwd -P)"
evidence_dir="$evidence_parent/$(basename "$1")"
[[ ! -e "$evidence_dir" && ! -L "$evidence_dir" ]] ||
    fail "Refusing to overwrite existing evidence: $evidence_dir"
umask 077
mkdir "$evidence_dir"
cd "$source_root"

{
    git rev-parse HEAD
    git status --porcelain=v1 --untracked-files=no
} > "$evidence_dir/source-before.txt"
{
    uname -a
    sw_vers
    sysctl -n machdep.cpu.brand_string hw.model hw.logicalcpu
} > "$evidence_dir/host.txt"
pmset -g batt > "$evidence_dir/power-before.txt"
pmset -g therm > "$evidence_dir/thermal-before.txt" 2>&1 || true

cmake --preset release > "$evidence_dir/configure.log" 2>&1
cmake --build build-release --target AudioPluginTest_BundleModel --parallel 2 \
    > "$evidence_dir/build.log" 2>&1
cp build-release/CMakeCache.txt "$evidence_dir/cmake-cache.txt"
test_binary="$source_root/build-release/test/AudioPluginTest"
[[ -x "$test_binary" ]] || fail "Release test binary is unavailable."
shasum -a 256 "$test_binary" model/model.onnx \
    > "$evidence_dir/inputs-before.sha256"

overall_status=0
trial=0
for threads in 2 1 2; do
    trial=$((trial + 1))
    log="$evidence_dir/trial-${trial}-threads-${threads}.log"
    printf 'Trial %s: %s ORT thread(s), 30,000 measured callbacks...\n' \
        "$trial" "$threads"
    if env STEMGENRT_TRACE_WORKER=1 STEMGENRT_PACED_ORT_THREADS="$threads" \
        STEMGENRT_QUALIFICATION_CALLBACKS=30000 \
        "$test_binary" --gtest_also_run_disabled_tests --gtest_color=no \
        --gtest_filter=RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenAsyncRealtimePaced \
        > "$log" 2>&1; then
        stage_status=0
    else
        stage_status=$?
    fi
    printf '%s\t%s\t%s\n' "$trial" "$threads" "$stage_status" \
        >> "$evidence_dir/exit-codes.tsv"
    printf 'Trial %s exited %s; log: %s\n' "$trial" "$stage_status" "$log"
    if [[ "$stage_status" -gt 1 ]]; then
        exit "$stage_status"
    fi
    if [[ "$stage_status" -ne 0 ]]; then
        overall_status=1
    fi
done

{
    git rev-parse HEAD
    git status --porcelain=v1 --untracked-files=no
} > "$evidence_dir/source-after.txt"
shasum -a 256 "$test_binary" model/model.onnx \
    > "$evidence_dir/inputs-after.sha256"
pmset -g batt > "$evidence_dir/power-after.txt"
pmset -g therm > "$evidence_dir/thermal-after.txt" 2>&1 || true
cmp -s "$evidence_dir/source-before.txt" "$evidence_dir/source-after.txt" ||
    fail "Source identity changed during comparison; retain the logs as invalid."
cmp -s "$evidence_dir/inputs-before.sha256" "$evidence_dir/inputs-after.sha256" ||
    fail "Binary/model changed during comparison; retain the logs as invalid."
printf 'Comparison complete; per-test exit codes and logs: %s\n' "$evidence_dir"
exit "$overall_status"
