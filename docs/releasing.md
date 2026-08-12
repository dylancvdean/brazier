# Releasing Brazier

Every desktop release is created by tagging the same version recorded in the
root and desktop manifests, for example `v0.2.2`. The GitHub Actions release
workflow builds three updateable artifacts:

- an Apple Silicon macOS DMG and ZIP, signed with the project's Developer ID
  certificate and notarized by Apple;
- a Linux x86_64 AppImage, always named `Brazier.AppImage`, with
  `latest-linux.yml` update metadata;
- a Windows x86_64 NSIS installer, always named `Brazier-Setup.exe`, signed
  with the project's Authenticode certificate and accompanied by `latest.yml`.

The AppImage and macOS updater use the GitHub Releases feed configured in the
desktop build. Electron's updater validates the SHA-512 recorded in the
platform metadata; macOS also validates the downloaded application's code
signature. A network or update failure never blocks launching Brazier.

Arch users should install the `PKGBUILD`/AUR package instead. Pacman/AUR owns
updates for that install and the application updater is not involved.

## AppImage update behavior

The AppImage is deliberately versionless. Download it once to a durable
location such as `~/Applications/Brazier.AppImage` and create any desktop-file
entry against that path. When the app offers an update and the user chooses
**Restart and update**, electron-updater replaces the running `APPIMAGE` path
in place. The desktop launcher therefore continues to point at the same file;
releases do not leave old versioned AppImages behind.

The Linux updater is disabled unless the process was launched as an AppImage.
It is also disabled for `BRAZIER_INSTALLED=1`, which is how the PKGBUILD marks
the pacman-managed installation.

## Release credentials

The `Release` workflow intentionally fails rather than produce an unsigned
macOS application. Configure these repository Action secrets before the first
tag:

- `MACOS_CERTIFICATE_P12`: base64 or file URL accepted by electron-builder's
  `CSC_LINK`, containing the Developer ID Application certificate;
- `MACOS_CERTIFICATE_PASSWORD`;
- `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, and `APPLE_TEAM_ID` for notarization.
- `WINDOWS_CERTIFICATE_P12` and `WINDOWS_CERTIFICATE_PASSWORD` for the Windows
  Authenticode signature.

GitHub's OIDC token signs each executable distribution, the macOS ZIP, the SPDX
SBOM, and `SHA512SUMS` with Sigstore; no Linux private signing key is stored in
GitHub. Updater metadata is covered by `SHA512SUMS` but does not receive a
separate Sigstore bundle. Verify the fixed-name AppImage (and its bundle) with:

```sh
cosign verify-blob \
  --bundle Brazier.AppImage.sigstore.json \
  --certificate-identity-regexp 'https://github.com/dylancvdean/brazier/.github/workflows/release.yml@refs/tags/v.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  Brazier.AppImage
```

The same command works with the matching bundle for the macOS DMG/ZIP, Windows
installer, SPDX SBOM, and `SHA512SUMS`. The certificate identity pins the
signature to this repository's tagged release workflow.

## Optional voice qualification

Voice qualification is an optional hardware-regression workflow. It does not
block ordinary releases. To assess a particular commit, check it out on both
hosts listed in `qualification/beta-manifest.json`, run the in-app **Qualify
voice** protocol, and submit the two compact saved JSON objects:

```sh
gh workflow run beta-voice-qualification.yml \
  -f commit="$(git rev-parse HEAD)" \
  -f macos_result="$(jq -c . macos-apple-silicon.json)" \
  -f linux_result="$(jq -c . linux-nvidia-x64.json)"
```

The evidence workflow validates and retains a commit-named artifact for bug
investigation. It is intentionally independent of release publication.

## Beta channels

Use normal semver prerelease tags, such as `v0.3.0-beta.1`. The workflow marks
them as full GitHub releases so the README's `/releases/latest/download/` links
continue to resolve. Do not retag or replace a published version: issue a
higher version instead, so updater metadata and provenance stay immutable.
