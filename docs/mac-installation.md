# macOS installation and signing status

Apple Silicon, macOS 14 or later, stereo 44.1 kHz. Current CI candidates are
ad-hoc signed test builds. They are **not Developer ID signed or notarized**;
macOS may refuse to load an internet-downloaded candidate. Do not advertise these
as a frictionless public installer. Do not disable Gatekeeper globally.

Close Logic/Ableton before installation. Replace the complete `StemgenRT.component`
in `~/Library/Audio/Plug-Ins/Components` and/or complete `StemgenRT.vst3` in
`~/Library/Audio/Plug-Ins/VST3`, retaining the old bundles outside the scan folders
for rollback. Avoid simultaneously installed system-wide and user copies.
For source builds use `scripts/install-plugins.sh --release`, which validates
and seals complete bundles. Restart the host and rescan if necessary.

For a downloaded ad-hoc candidate that is blocked, verify the supplied checksum
and origin first. Follow Apple's per-application security approval guidance where
available; if the host offers no approval path for the plugin, use a local source
build for testing. Signing and notarization are still required for the intended
broad launch. See [Apple's Developer ID guidance](https://developer.apple.com/developer-id/).

After membership and certificates are available, configure Release with
`-DSTEMGENRT_MACOS_SIGN_IDENTITY='Developer ID Application: Name (TEAM)'`.
Store notarization credentials in a local keychain profile using `xcrun notarytool
store-credentials`; never commit credentials. The bundle-sealing helper signs
embedded libraries before the outer bundle and enables timestamps and hardened
runtime for a real identity. Then run:

```sh
bash scripts/notarize-macos.sh build 'Developer ID Application: Name (TEAM)' \
  profile-name /path/to/new/StemgenRT-macOS.dmg
```

The script creates a disk image, requires an accepted notarization result,
staples and validates its ticket, and records the final image hash. Preserve the
JSON receipt and ship the matching CI source archive alongside the image. The
script has not been exercised with a Developer ID identity yet; test the exact
downloaded final image on a separate Mac before announcing notarized availability.
