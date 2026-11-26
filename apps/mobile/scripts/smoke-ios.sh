#!/bin/bash
set -euo pipefail

device="${1:?Pass a booted simulator UDID}"
app="${2:?Pass the Release simulator Brazier.app path}"
bundle_id=ai.brazier.mobile

# Release must include its own bundle, as TestFlight does; a Debug build can
# appear healthy while depending on a developer's running Metro server.
test -s "$app/main.jsbundle"
xcrun simctl bootstatus "$device" -b
xcrun simctl install "$device" "$app"

for launch in 1 2; do
  output="$(xcrun simctl launch --terminate-running-process "$device" "$bundle_id")"
  pid="${output##*: }"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "Could not determine app process: $output" >&2
    exit 1
  fi
  # simctl launch succeeds even if UIKit immediately aborts the app. Check the
  # actual process for long enough to catch native and JS initialization errors.
  for second in {1..15}; do
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
      xcrun simctl spawn "$device" log show --last 1m --style compact --predicate 'process == "Brazier"'
      echo "Release launch $launch crashed." >&2
      exit 1
    fi
  done
  echo "Release launch $launch stayed alive without Metro."
done
