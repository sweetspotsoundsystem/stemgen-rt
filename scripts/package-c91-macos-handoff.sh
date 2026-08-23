#!/bin/bash
# Seal the exact current c91 source snapshot for transfer to the target Mac.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/package-c91-macos-handoff.sh NEW_OUTPUT_DIRECTORY

Creates a deterministic source archive plus SHA-256 sidecar. The output
directory must not exist and must be outside the source tree. The archive
contains every tracked or untracked non-ignored source input, the exact ONNX
model, provenance, and a complete manifest verified by the Mac qualifier.
EOF
}

fail() {
    printf '%s\n' "C91_HANDOFF_PACKAGE: FAIL: $*" >&2
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

for command_name in cp dirname git gzip mkdir mv shasum sort tar; do
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "required command is unavailable: $command_name"
done
tar --version 2>/dev/null | grep -q 'GNU tar' ||
    fail "GNU tar is required to produce the deterministic archive"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    fail "source must be a Git worktree"

OUTPUT_ARGUMENT="$1"
case "$OUTPUT_ARGUMENT" in
    ""|-*) fail "output directory must be a nonempty path not beginning with '-'" ;;
esac
OUTPUT_PARENT="$(dirname "$OUTPUT_ARGUMENT")"
OUTPUT_NAME="$(basename "$OUTPUT_ARGUMENT")"
[[ -d "$OUTPUT_PARENT" ]] || fail "output parent does not exist: $OUTPUT_PARENT"
OUTPUT_PARENT="$(cd "$OUTPUT_PARENT" && pwd -P)"
OUTPUT_DIR="$OUTPUT_PARENT/$OUTPUT_NAME"
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] ||
    fail "refusing to reuse or overwrite output: $OUTPUT_DIR"
case "$OUTPUT_DIR/" in
    "$PROJECT_ROOT/"*) fail "output directory must be outside the source tree" ;;
esac

umask 077
mkdir "$OUTPUT_DIR"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/c91-mac-handoff.XXXXXX")"
cleanup() {
    rm -rf "$STAGING_DIR"
}
trap cleanup EXIT
SOURCE_ROOT="$STAGING_DIR/c91-hardened-23ms"
mkdir "$SOURCE_ROOT"
FILE_LIST="$STAGING_DIR/source-files.txt"
git ls-files -co --exclude-standard | LC_ALL=C sort > "$FILE_LIST"
[[ -s "$FILE_LIST" ]] || fail "source file inventory is empty"

for required_path in \
    model/model.onnx \
    scripts/download-onnxruntime.sh \
    scripts/qualify-c91-macos.sh \
    scripts/package-c91-macos-handoff.sh \
    test/source/RealtimeStemSanityTest.cpp; do
    grep -Fxq "$required_path" "$FILE_LIST" ||
        fail "required source input is absent from inventory: $required_path"
done

while IFS= read -r relative_path; do
    [[ -n "$relative_path" && ( -f "$relative_path" || -L "$relative_path" ) ]] ||
        fail "source inventory entry is unavailable: $relative_path"
    destination="$SOURCE_ROOT/$relative_path"
    mkdir -p "$(dirname "$destination")"
    cp -pP "$relative_path" "$destination"
done < "$FILE_LIST"

cp "$FILE_LIST" "$SOURCE_ROOT/SOURCE_FILES.txt"
{
    printf 'snapshot_kind=tracked_and_untracked_nonignored_exact_files\n'
    printf 'base_commit='; git rev-parse HEAD
    if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
        printf 'source_worktree_clean=false\n'
    else
        printf 'source_worktree_clean=true\n'
    fi
    printf 'source_file_count='; wc -l < "$FILE_LIST" | tr -d '[:space:]'; printf '\n'
} > "$SOURCE_ROOT/SOURCE_PROVENANCE.txt"

(
    cd "$SOURCE_ROOT"
    while IFS= read -r relative_path; do
        shasum -a 256 "$relative_path"
    done < SOURCE_FILES.txt
) > "$SOURCE_ROOT/SOURCE_MANIFEST.sha256"

cat > "$SOURCE_ROOT/HANDOFF_README.md" <<'EOF'
# C91 hardened 23 ms target-Mac handoff

This is an exact source snapshot, not a promoted release.

Before building, verify it from this directory:

    shasum -a 256 -c HANDOFF_CONTENTS.sha256

Then run the fail-closed target-machine qualification into a new directory
outside this tree. Supply the archive hash from the externally verified
`SHA256SUMS` so the final machine receipt carries the transfer identity:

    STEMGENRT_HANDOFF_ARCHIVE_SHA256=<archive-sha256> \
      ./scripts/qualify-c91-macos.sh /absolute/existing-parent/c91-mac-evidence-v1

A machine pass remains unpromoted until AU/VST3 DAW checks, locked blind
votes, and explicit user approval are all recorded.
EOF

(
    cd "$SOURCE_ROOT"
    find . \( -type f -o -type l \) ! -name HANDOFF_CONTENTS.sha256 -print |
        LC_ALL=C sort |
        while IFS= read -r relative_path; do
            shasum -a 256 "$relative_path"
        done
) > "$SOURCE_ROOT/HANDOFF_CONTENTS.sha256"
(
    cd "$SOURCE_ROOT"
    shasum -a 256 -c HANDOFF_CONTENTS.sha256 >/dev/null
)

ARCHIVE_NAME="c91-hardened-23ms-source.tar.gz"
ARCHIVE_TMP="$OUTPUT_DIR/.${ARCHIVE_NAME}.tmp"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    -C "$SOURCE_ROOT" -cf - . | gzip -n > "$ARCHIVE_TMP"
mv "$ARCHIVE_TMP" "$OUTPUT_DIR/$ARCHIVE_NAME"
(
    cd "$OUTPUT_DIR"
    shasum -a 256 "$ARCHIVE_NAME" > SHA256SUMS
    shasum -a 256 -c SHA256SUMS >/dev/null
)

printf '%s\n' \
    "C91_HANDOFF_PACKAGE: PASS" \
    "Archive: $OUTPUT_DIR/$ARCHIVE_NAME" \
    "Checksums: $OUTPUT_DIR/SHA256SUMS"
