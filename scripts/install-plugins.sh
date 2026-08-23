#!/bin/bash
# Install StemgenRT plugins to the current user's macOS plugin directories.
# Usage: ./scripts/install-plugins.sh [--debug|--release] [--sign-identity "Developer ID Application: ..."]
#
# Qualified Release artifacts are installed by default.
# For local Debug development: ./scripts/install-plugins.sh --debug
# For distribution signing:
#   ./scripts/install-plugins.sh --release --sign-identity "Developer ID Application: Your Name (XXXXXXXXXX)"
#
# Ad-hoc signing (the default) seals bundle integrity but does not provide
# Developer ID trust or notarization for quarantined downloads.

set -euo pipefail

usage() {
    echo "Usage: ./scripts/install-plugins.sh [--debug|--release] [--sign-identity \"Developer ID Application: ...\"]"
    echo ""
    echo "Options:"
    echo "  --release          Install qualified Release artifacts (default)"
    echo "  --debug            Install Debug artifacts"
    echo "  --sign-identity ID Sign staged bundles with ID instead of ad-hoc signing"
    echo "  -h, --help         Show this help"
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

BUILD_DIR="build-release"
BUILD_TYPE="Release"
BUILD_PRESET="release"
SIGN_IDENTITY="-"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            BUILD_DIR="build"
            BUILD_TYPE="Debug"
            BUILD_PRESET="default"
            shift
            ;;
        --release)
            BUILD_DIR="build-release"
            BUILD_TYPE="Release"
            BUILD_PRESET="release"
            shift
            ;;
        --sign-identity)
            [[ $# -ge 2 ]] || fail "--sign-identity requires an identity"
            SIGN_IDENTITY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown option '$1'"
            ;;
    esac
done

[[ "$(uname -s)" == "Darwin" ]] || fail "this installer supports macOS only"
command -v cmake >/dev/null 2>&1 || fail "cmake is required"
command -v ditto >/dev/null 2>&1 || fail "ditto is required"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTEFACTS_DIR="$PROJECT_ROOT/$BUILD_DIR/plugin/AudioPlugin_artefacts/$BUILD_TYPE"
SEAL_SCRIPT="$PROJECT_ROOT/cmake/SignAndVerifyAppleBundle.cmake"

AU_PLUGIN="$ARTEFACTS_DIR/AU/StemgenRT.component"
VST3_PLUGIN="$ARTEFACTS_DIR/VST3/StemgenRT.vst3"

AU_DEST_DIR="$HOME/Library/Audio/Plug-Ins/Components"
VST3_DEST_DIR="$HOME/Library/Audio/Plug-Ins/VST3"
AU_INSTALLED="$AU_DEST_DIR/StemgenRT.component"
VST3_INSTALLED="$VST3_DEST_DIR/StemgenRT.vst3"

[[ -f "$SEAL_SCRIPT" ]] || fail "bundle sealing script not found at $SEAL_SCRIPT"

missing_artifact=false
if [[ ! -d "$AU_PLUGIN" ]]; then
    echo "Missing AU artifact: $AU_PLUGIN" >&2
    missing_artifact=true
fi
if [[ ! -d "$VST3_PLUGIN" ]]; then
    echo "Missing VST3 artifact: $VST3_PLUGIN" >&2
    missing_artifact=true
fi
if [[ "$missing_artifact" == true ]]; then
    echo "Build and seal the $BUILD_TYPE artifacts first:" >&2
    echo "  cmake --preset $BUILD_PRESET" >&2
    echo "  cmake --build --preset $BUILD_PRESET --target StemgenRT_VerifyMacBundles" >&2
    exit 1
fi

verify_source_bundle() {
    local bundle_path="$1"
    local format_name="$2"

    echo "Verifying sealed $format_name source artifact..."
    cmake \
        "-DSTEMGENRT_BUNDLE_PATH=$bundle_path" \
        -DSTEMGENRT_VERIFY_ONLY=ON \
        -P "$SEAL_SCRIPT"
}

install_bundle() {
    local source_bundle="$1"
    local destination_dir="$2"
    local destination_bundle="$3"
    local format_name="$4"
    local backup_root=""
    local backup_bundle=""
    local had_existing_bundle=false
    local staging_root
    local staged_bundle

    [[ "${destination_bundle%/*}" == "$destination_dir" ]] ||
        fail "refusing destination outside '$destination_dir': $destination_bundle"
    case "$destination_bundle" in
        "$AU_INSTALLED"|"$VST3_INSTALLED") ;;
        *) fail "refusing unexpected destination bundle '$destination_bundle'" ;;
    esac

    echo "Installing $format_name plugin..."
    mkdir -p "$destination_dir"
    staging_root="$(mktemp -d "$destination_dir/.stemgenrt-install.XXXXXX")"
    case "$staging_root" in
        "$destination_dir"/.stemgenrt-install.*) ;;
        *) fail "refusing unexpected staging path '$staging_root'" ;;
    esac
    staged_bundle="$staging_root/${destination_bundle##*/}"

    if ! ditto "$source_bundle" "$staged_bundle"; then
        rm -rf -- "$staging_root"
        fail "failed to stage the $format_name bundle"
    fi

    # The central sealer owns install-name normalization, nested-to-outer
    # signing order, model/ORT validation, and strict signature verification.
    if ! cmake \
        "-DSTEMGENRT_BUNDLE_PATH=$staged_bundle" \
        "-DSTEMGENRT_SIGN_IDENTITY=$SIGN_IDENTITY" \
        -P "$SEAL_SCRIPT"; then
        rm -rf -- "$staging_root"
        fail "failed to seal the staged $format_name bundle; the installed copy was left unchanged"
    fi

    # Keep replacement and rollback on the destination filesystem. The old
    # bundle remains recoverable until the verified staged rename succeeds.
    if [[ -e "$destination_bundle" || -L "$destination_bundle" ]]; then
        backup_root="$(mktemp -d "$destination_dir/.stemgenrt-backup.XXXXXX")"
        case "$backup_root" in
            "$destination_dir"/.stemgenrt-backup.*) ;;
            *) fail "refusing unexpected backup path '$backup_root'" ;;
        esac
        backup_bundle="$backup_root/${destination_bundle##*/}"
        case "$backup_bundle" in
            "$backup_root"/StemgenRT.component|"$backup_root"/StemgenRT.vst3) ;;
            *) fail "refusing unexpected backup bundle '$backup_bundle'" ;;
        esac

        if ! mv "$destination_bundle" "$backup_bundle"; then
            rm -rf -- "$staging_root"
            rmdir "$backup_root" || true
            fail "failed to preserve the installed $format_name bundle"
        fi
        had_existing_bundle=true
    fi

    if ! mv "$staged_bundle" "$destination_bundle"; then
        if [[ "$had_existing_bundle" == true ]]; then
            if ! mv "$backup_bundle" "$destination_bundle"; then
                fail "failed to place the verified $format_name bundle and could not restore the previous bundle; it remains at $backup_bundle"
            fi
            rmdir "$backup_root"
            rm -rf -- "$staging_root"
            fail "failed to place the verified $format_name bundle; the previous installation was restored"
        fi
        rm -rf -- "$staging_root"
        fail "failed to place the verified $format_name bundle; no existing installation was changed"
    fi

    if [[ "$had_existing_bundle" == true ]]; then
        rm -rf -- "$backup_bundle"
        rmdir "$backup_root"
    fi
    rmdir "$staging_root"

    echo "  Installed and verified: $destination_bundle"
}

echo "Installing StemgenRT $BUILD_TYPE plugins from $BUILD_DIR..."
echo ""

# Validate both inputs before replacing either installed plugin, preventing a
# missing or incompletely sealed format from causing a partial installation.
verify_source_bundle "$AU_PLUGIN" "AU"
verify_source_bundle "$VST3_PLUGIN" "VST3"

install_bundle "$AU_PLUGIN" "$AU_DEST_DIR" "$AU_INSTALLED" "AU"
install_bundle "$VST3_PLUGIN" "$VST3_DEST_DIR" "$VST3_INSTALLED" "VST3"

echo ""
echo "Done. Restart your DAW if it has already scanned the previous plugins."
echo "To force an Audio Unit rescan, run:"
echo "  killall AudioComponentRegistrar"

if [[ "$SIGN_IDENTITY" == "-" ]]; then
    echo ""
    echo "Note: the installed plugins are ad-hoc signed for local use."
    echo "A public release still requires Developer ID signing and notarization."
fi
