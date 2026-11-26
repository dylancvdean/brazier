#!/bin/bash
set -euo pipefail

export CI=1
export EXPO_NO_TELEMETRY=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1

cd "${CI_PRIMARY_REPOSITORY_PATH:?Xcode Cloud repository path is required}"

# Xcode Cloud supplies Homebrew; Node 24 is required by this workspace.
brew install node@24
export PATH="$(brew --prefix node@24)/bin:$PATH"
tools="${CI_DERIVED_DATA_PATH:?Xcode Cloud derived data path is required}/brazier-tools"
npm install --prefix "$tools" --no-package-lock --ignore-scripts pnpm@11.17.0
export PATH="$tools/node_modules/.bin:$PATH"

# Only mobile dependencies are needed. Skip the desktop Electron postinstall.
pnpm install --frozen-lockfile --ignore-scripts
pnpm --filter @brazier/mobile --fail-if-no-match check
pnpm --filter @brazier/mobile --fail-if-no-match test
# Installs skip package scripts; llama.rn's native libraries are fetched and checksum-verified here.
pnpm --filter @brazier/mobile --fail-if-no-match llama:artifacts

cd apps/mobile/ios
# Xcode build phases run in a separate shell, so persist the Node location.
printf 'export NODE_BINARY="%s"\n' "$(command -v node)" > .xcode.env.local
if ! command -v pod >/dev/null 2>&1; then
  brew install cocoapods
fi
pod install --deployment
