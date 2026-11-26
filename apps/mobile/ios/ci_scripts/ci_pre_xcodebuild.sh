#!/bin/bash
set -euo pipefail

cd "${CI_PRIMARY_REPOSITORY_PATH:?}/apps/mobile/ios"
# Each archive gets Xcode Cloud's increasing build number for TestFlight.
xcrun agvtool new-version -all "${CI_BUILD_NUMBER:?Xcode Cloud build number is required}"
