#!/bin/bash
# Download the official ONNX Runtime release (self-contained, no external dependencies)
# Usage: ./scripts/download-onnxruntime.sh

set -euo pipefail

VERSION="1.26.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$PROJECT_ROOT/libs/onnxruntime"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/stemgenrt-ort.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    PLATFORM="osx-arm64"
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "ONNX Runtime ${VERSION} does not publish an official macOS x86_64 archive."
    echo "Run this script from a native Apple Silicon shell."
    exit 1
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

FILENAME="onnxruntime-${PLATFORM}-${VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${FILENAME}"

echo "Downloading ONNX Runtime ${VERSION} for ${PLATFORM}..."
echo "URL: $URL"

# Download and validate before replacing an existing SDK.
ARCHIVE_PATH="$TEMP_DIR/$FILENAME"
EXTRACT_DIR="$TEMP_DIR/extracted"
mkdir -p "$EXTRACT_DIR"
curl --fail --location --retry 3 -o "$ARCHIVE_PATH" "$URL"
tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"

SDK_DIR="$EXTRACT_DIR/onnxruntime-${PLATFORM}-${VERSION}"
if [[ ! -d "$SDK_DIR" ]]; then
    echo "Downloaded archive did not contain the expected SDK directory: $SDK_DIR"
    exit 1
fi

INSTALLED_VERSION="$(tr -d '[:space:]' < "$SDK_DIR/VERSION_NUMBER")"
if [[ "$INSTALLED_VERSION" != "$VERSION" ]]; then
    echo "Downloaded SDK reports ONNX Runtime $INSTALLED_VERSION; expected $VERSION"
    exit 1
fi
if [[ ! -f "$SDK_DIR/include/onnxruntime_c_api.h" || ! -e "$SDK_DIR/lib/libonnxruntime.dylib" ]]; then
    echo "Downloaded SDK is incomplete (header or dylib missing)."
    exit 1
fi

rm -rf "$DEST_DIR"
mkdir -p "$(dirname "$DEST_DIR")"
mv "$SDK_DIR" "$DEST_DIR"

echo ""
echo "✓ ONNX Runtime ${VERSION} installed to: $DEST_DIR"
echo ""
echo "Contents:"
ls -la "$DEST_DIR"
echo ""
echo "Now rebuild your project:"
echo "  cmake --preset release"
echo "  cmake --build --preset release"
