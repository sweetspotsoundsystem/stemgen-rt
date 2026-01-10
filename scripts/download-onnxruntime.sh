#!/bin/bash
# Download the official ONNX Runtime release (self-contained, no external dependencies)
# Usage: ./scripts/download-onnxruntime.sh

set -e

VERSION="1.22.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$PROJECT_ROOT/libs/onnxruntime"

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    PLATFORM="osx-arm64"
elif [[ "$ARCH" == "x86_64" ]]; then
    PLATFORM="osx-x64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

FILENAME="onnxruntime-${PLATFORM}-${VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${FILENAME}"

echo "Downloading ONNX Runtime ${VERSION} for ${PLATFORM}..."
echo "URL: $URL"

# Create destination directory
mkdir -p "$DEST_DIR"

# Download and extract
cd "$DEST_DIR"
curl -L -o "$FILENAME" "$URL"
tar -xzf "$FILENAME" --strip-components=1
rm "$FILENAME"

echo ""
echo "✓ ONNX Runtime ${VERSION} installed to: $DEST_DIR"
echo ""
echo "Contents:"
ls -la "$DEST_DIR"
echo ""
echo "Now rebuild your project:"
echo "  rm -rf build && cmake --preset release && cmake --build build-release"

