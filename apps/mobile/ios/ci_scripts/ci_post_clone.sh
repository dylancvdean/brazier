#!/bin/bash
set -euo pipefail

export CI=1
export EXPO_NO_TELEMETRY=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1

cd "${CI_PRIMARY_REPOSITORY_PATH:?Xcode Cloud repository path is required}"

tools="${CI_DERIVED_DATA_PATH:?Xcode Cloud derived data path is required}/brazier-tools"

# Node 24 is required by this workspace. Xcode Cloud supplies Homebrew, but its
# bottle host is sometimes unreachable from the build machine, so retry and then
# fall back to the official build, checked against the published checksums.
install_node() {
  for attempt in 1 2 3; do
    if brew install node@24; then
      export PATH="$(brew --prefix node@24)/bin:$PATH"
      return
    fi
    sleep $((attempt * 10))
  done
  local dist=https://nodejs.org/dist/latest-v24.x arch=arm64 archive sums
  [ "$(uname -m)" = arm64 ] || arch=x64
  mkdir -p "$tools/node"
  sums="$(curl --fail --silent --show-error --location --retry 3 "$dist/SHASUMS256.txt")"
  archive="$(printf '%s\n' "$sums" | awk -v suffix="-darwin-$arch.tar.gz" 'index($2, suffix) { print $2 }')"
  [ -n "$archive" ] || { echo "No Node 24 build for darwin-$arch" >&2; return 1; }
  curl --fail --silent --show-error --location --retry 3 --output "$tools/$archive" "$dist/$archive"
  (cd "$tools" && printf '%s\n' "$sums" | grep " $archive\$" | shasum -a 256 -c -)
  tar -xzf "$tools/$archive" -C "$tools/node" --strip-components 1
  export PATH="$tools/node/bin:$PATH"
}
install_node
npm install --prefix "$tools" --no-package-lock --ignore-scripts pnpm@11.17.0
export PATH="$tools/node_modules/.bin:$PATH"

# Only mobile dependencies are needed. Skip the desktop Electron postinstall.
pnpm install --frozen-lockfile --ignore-scripts
pnpm --filter @brazier/mobile --fail-if-no-match check
pnpm --filter @brazier/mobile --fail-if-no-match test

cd apps/mobile/ios
# Xcode build phases run in a separate shell, so persist the Node location.
printf 'export NODE_BINARY="%s"\n' "$(command -v node)" > .xcode.env.local
if ! command -v pod >/dev/null 2>&1; then
  brew install cocoapods
fi
pod install --deployment
