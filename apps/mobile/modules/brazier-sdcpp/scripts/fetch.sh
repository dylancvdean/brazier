#!/bin/bash
# Checks out stable-diffusion.cpp at the pinned commit into vendor/, with the
# ggml submodule only. Does nothing when that commit is already there.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
commit="$(tr -d '[:space:]' < "$root/SDCPP_COMMIT")"
source="$root/vendor/stable-diffusion.cpp"

if [ "$(git -C "$source" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]; then
  rm -rf "$source"
  mkdir -p "$source"
  git -C "$source" init --quiet
  git -C "$source" remote add origin https://github.com/leejet/stable-diffusion.cpp.git
  git -C "$source" fetch --quiet --depth 1 origin "$commit"
  git -C "$source" checkout --quiet FETCH_HEAD
fi
git -C "$source" submodule update --quiet --init --depth 1 ggml
[ "$(git -C "$source" rev-parse HEAD)" = "$commit" ]
