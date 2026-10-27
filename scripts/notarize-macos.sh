#!/usr/bin/env bash
# Run manually on a Mac with Developer ID certificates and a notarytool profile.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: $0 BUILD_DIR 'Developer ID Application: Name (TEAM)' KEYCHAIN_PROFILE NEW_OUTPUT.dmg" >&2
  exit 2
fi
build_dir="$1"
identity="$2"
profile="$3"
output="$4"
[[ $(uname -s) == Darwin && "$identity" == 'Developer ID Application:'* ]]
[[ "$output" == *.dmg && ! -e "$output" && ! -e "$output.notary.json" ]]
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/stemgenrt-notary.XXXXXX")"
trap 'rm -rf "$staging_dir"' EXIT
mkdir "$staging_dir/payload"
for pair in 'VST3/StemgenRT.vst3' 'AU/StemgenRT.component'; do
  bundle="$build_dir/plugin/AudioPlugin_artefacts/Release/$pair"
  ditto "$bundle" "$staging_dir/payload/${pair##*/}"
  cmake "-DSTEMGENRT_BUNDLE_PATH=$staging_dir/payload/${pair##*/}" \
    "-DSTEMGENRT_SIGN_IDENTITY=$identity" \
    -P "$repo_dir/cmake/SignAndVerifyAppleBundle.cmake"
done
cp "$repo_dir/docs/mac-installation.md" "$staging_dir/payload/INSTALL.txt"
hdiutil create -volname StemgenRT -srcfolder "$staging_dir/payload" -format UDZO "$output"
codesign --force --sign "$identity" --timestamp "$output"
xcrun notarytool submit "$output" --keychain-profile "$profile" --wait --output-format json > "$output.notary.json"
python3 - "$output.notary.json" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
if receipt.get('status') != 'Accepted':
    raise SystemExit('Notarization not accepted; preserve receipt and inspect notarytool log')
PY
xcrun stapler staple "$output"
xcrun stapler validate "$output"
spctl --assess --type open --context context:primary-signature --verbose=2 "$output"
shasum -a 256 "$output" > "$output.sha256"
echo "Notarized disk image: $output. Test installation from this exact image on another Mac."
