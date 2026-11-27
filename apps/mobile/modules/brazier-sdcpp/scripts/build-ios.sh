#!/bin/bash
# Builds stable-diffusion.cpp (pinned in SDCPP_COMMIT) with Metal as a static
# xcframework for the BrazierSdcpp pod. The Podfile runs this before installing
# pods; it does nothing when the xcframework already matches the pinned commit.
#
# SDCPP_IOS_SLICES chooses what to build: "device simulator" (default),
# "device" or "simulator".
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
commit="$(tr -d '[:space:]' < "$root/SDCPP_COMMIT")"
slices="${SDCPP_IOS_SLICES:-device simulator}"
build="$root/build/ios"
framework="$build/sdcpp.xcframework"
stamp="$commit $slices"

if [ -d "$framework" ] && [ "$(cat "$build/stamp" 2>/dev/null)" = "$stamp" ]; then
  exit 0
fi
command -v cmake >/dev/null || { echo "cmake is required to build stable-diffusion.cpp (brew install cmake)." >&2; exit 1; }

bash "$root/scripts/fetch.sh"
source="$root/vendor/stable-diffusion.cpp"
rm -rf "$framework" "$build/stamp"

arguments=()
for slice in $slices; do
  case "$slice" in
    device) sdk=iphoneos ;;
    simulator) sdk=iphonesimulator ;;
    *) echo "Unknown slice: $slice" >&2; exit 1 ;;
  esac
  out="$build/$slice"
  cmake -S "$source" -B "$out" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$sdk" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 \
    -DSD_METAL=ON \
    -DSD_BUILD_EXAMPLES=OFF \
    -DSD_WEBP=OFF \
    -DSD_WEBM=OFF \
    -DGGML_OPENMP=OFF
  cmake --build "$out" --target stable-diffusion --parallel "$(sysctl -n hw.ncpu)"
  # One archive per slice: the library and the ggml libraries it was built with.
  find "$out" -name '*.a' ! -name libsdcpp.a -print0 | xargs -0 xcrun libtool -static -o "$out/libsdcpp.a"
  arguments+=(-library "$out/libsdcpp.a" -headers "$source/include")
done

xcodebuild -create-xcframework "${arguments[@]}" -output "$framework"
printf '%s' "$stamp" > "$build/stamp"
