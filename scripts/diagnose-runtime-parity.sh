#!/bin/bash
# Compare three ORT sessions against this checkout's independent fixture.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'Usage: bash scripts/diagnose-runtime-parity.sh /path/to/stemgen-rt /path/to/new-evidence-directory' >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$1" && pwd -P)"
SCRIPT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
mkdir "$2"
EVIDENCE_ROOT="$(cd "$2" && pwd -P)"
MODEL="$SOURCE_ROOT/model/model.onnx"
FIXTURE="$SOURCE_ROOT/test/fixtures/cropped1024-pytorch.bin"
cmake "-DSTEMGENRT_SOURCE_ROOT=$SOURCE_ROOT" -P "$SCRIPT_ROOT/cmake/VerifyParityFixture.cmake" > "$EVIDENCE_ROOT/fixture-verification.log" 2>&1 || {
  cat "$EVIDENCE_ROOT/fixture-verification.log" >&2
  exit 2
}
case "$(uname -s)" in
  Darwin) RUNTIME="$SOURCE_ROOT/libs/onnxruntime/lib/libonnxruntime.dylib" ;;
  Linux) RUNTIME="$SOURCE_ROOT/libs/onnxruntime/lib/libonnxruntime.so" ;;
  *) echo 'This diagnostic supports macOS and Linux.' >&2; exit 2 ;;
esac
{
  date -u
  uname -a
  git -C "$SOURCE_ROOT" rev-parse HEAD
  git -C "$SOURCE_ROOT" status --short
  "${CXX:-c++}" --version
  shasum -a 256 "$MODEL" "$FIXTURE" "$RUNTIME" "$SOURCE_ROOT/test/fixtures/cropped1024-pytorch.json" "$SOURCE_ROOT/cmake/QualifiedModelContract.cmake" "$SCRIPT_ROOT/tools/RuntimeParityDiagnostic.cpp" "$SCRIPT_ROOT/scripts/diagnose-runtime-parity.sh" "$SCRIPT_ROOT/cmake/VerifyParityFixture.cmake"
  if [[ "$(uname -s)" == Darwin ]]; then
    sysctl machdep.cpu.brand_string
    sysctl hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2 || true
  fi
} > "$EVIDENCE_ROOT/identity.txt"
set +e
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror \
  -I "$SOURCE_ROOT/libs/onnxruntime/include" "$SCRIPT_ROOT/tools/RuntimeParityDiagnostic.cpp" \
  "$RUNTIME" -Wl,-rpath,"$SOURCE_ROOT/libs/onnxruntime/lib" \
  -o "$EVIDENCE_ROOT/diagnose" > "$EVIDENCE_ROOT/build.log" 2>&1
BUILD_EXIT=$?
set -e
printf '%s\n' "$BUILD_EXIT" > "$EVIDENCE_ROOT/build-exit.txt"
if (( BUILD_EXIT != 0 )); then cat "$EVIDENCE_ROOT/build.log"; exit "$BUILD_EXIT"; fi
shasum -a 256 "$EVIDENCE_ROOT/diagnose" >> "$EVIDENCE_ROOT/identity.txt"
set +e
"$EVIDENCE_ROOT/diagnose" "$MODEL" "$FIXTURE" > "$EVIDENCE_ROOT/results.jsonl" 2> "$EVIDENCE_ROOT/stderr.log"
DIAGNOSTIC_EXIT=$?
set -e
printf '%s\n' "$DIAGNOSTIC_EXIT" > "$EVIDENCE_ROOT/diagnostic-exit.txt"
LOADED_RUNTIME="$(sed -n 's/^Loaded runtime: //p' "$EVIDENCE_ROOT/stderr.log")"
if [[ -n "$LOADED_RUNTIME" && -f "$LOADED_RUNTIME" ]]; then
  shasum -a 256 "$LOADED_RUNTIME" >> "$EVIDENCE_ROOT/identity.txt"
fi
cat "$EVIDENCE_ROOT/results.jsonl"
cat "$EVIDENCE_ROOT/stderr.log"
echo "Diagnostic completion exit: $DIAGNOSTIC_EXIT; parity verdicts are in results.jsonl."
exit "$DIAGNOSTIC_EXIT"
