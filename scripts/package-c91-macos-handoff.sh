#!/bin/bash
# Compatibility entry point; the model contract now selects cropped1024.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/package-macos-handoff.sh" "$@"
