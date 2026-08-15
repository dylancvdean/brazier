# Maintainer: Dylan C. V. Dean <dylan@dylancvdean.com>

pkgname=brazier
# Pacman reserves `-` as the separator before pkgrel, so encode upstream
# prerelease separators with `_` in pkgver.
pkgver=0.2.13_beta.74
pkgrel=1
pkgdesc='Desktop client and local API for open-weight AI models'
arch=('x86_64' 'aarch64')
url='https://github.com/dylancvdean/brazier'
license=('MIT')
# The daemon is built as a stripped Cargo release binary; makepkg's debug
# package/index hook has no DWARF data to process.
options=(!debug !lto)
depends=('electron')
makedepends=('cargo' 'git' 'nodejs' 'pnpm' 'rust')
# Host packages Brazier uses when building or preparing engine runtimes from
# the app. None are required just to launch the UI against a prebuilt runtime.
optdepends=(
  'git: clone engine sources for llama.cpp, whisper.cpp, stable-diffusion.cpp, vLLM, and PersonaPlex builds'
  'cmake: configure CMake-based engine builds (llama.cpp, whisper.cpp, stable-diffusion.cpp)'
  'base-devel: C/C++ toolchain (gcc, make, pkgconf, …) for compiling engine sources'
  'cuda: NVIDIA CUDA toolkit for CUDA builds of llama.cpp, whisper.cpp, stable-diffusion.cpp, and vLLM'
  'rocm-hip-sdk: AMD ROCm/HIP SDK for ROCm builds of llama.cpp and related engines'
  'rocm-opencl-sdk: OpenCL SDK used alongside ROCm engine builds'
  'hipsparselt: hipSPARSELt library required for vLLM on AMD GPUs (ROCm)'
  'vulkan-headers: Vulkan headers for Vulkan builds of llama.cpp and related engines'
  'vulkan-icd-loader: Vulkan loader for Vulkan engine builds'
  'spirv-headers: SPIR-V headers for Vulkan engine builds'
  'glslang: SPIR-V shader tooling for Vulkan engine builds'
  'uv: create Python environments for vLLM, streaming ASR, and PersonaPlex'
  'ffmpeg: sample video frames for vision models and convert audio for transcription'
  'polkit: authorize installation of the optional Wayland emergency-key fallback'
)
# source=("${pkgname}::git+${url}.git#branch=master")
# Package the checkout that contains this PKGBUILD. `makepkg` clones it into
# $srcdir, so commit the changes you want included before building.
source=("${pkgname}::git+file://${startdir}")
b2sums=('SKIP')

prepare() {
  cd "${srcdir}/${pkgname}"
  # electron-vite only needs Electron's Node typings to build. Do not fetch or
  # install the upstream Electron binary: the launcher uses Arch's `electron`.
  export ELECTRON_SKIP_BINARY_DOWNLOAD=1
  pnpm install --frozen-lockfile --ignore-scripts
}

build() {
  cd "${srcdir}/${pkgname}"
  # sqlx builds bundled SQLite for its procedural macro. Arch's C-level LTO
  # leaves SQLite symbols out of that loadable macro (`sqlite3_db_config` is
  # then unresolved), so keep this mixed Rust/C build non-LTO.
  export CFLAGS="${CFLAGS//-flto=auto/}"
  export CXXFLAGS="${CXXFLAGS//-flto=auto/}"
  cargo clean
  pnpm --filter @brazier/desktop run build:agent
  pnpm --filter @brazier/desktop exec electron-vite build
  cargo build --release --locked -p brazierd -p brazier-safety -p brazier-input-guard
}

package() {
  cd "${srcdir}/${pkgname}"

  install -dm755 "${pkgdir}/usr/lib/${pkgname}"
  cp -a apps/desktop/out "${pkgdir}/usr/lib/${pkgname}/"
  # The agent worker keeps Pi packages external. Copy the full runtime closure,
  # not just top-level symlinks — otherwise transitive deps like `openai` are
  # missing and the utilityProcess worker exits on import (code 1).
  node apps/desktop/scripts/stage-packaged-node-modules.mjs \
    "${pkgdir}/usr/lib/${pkgname}/node_modules"
  install -Dm644 apps/desktop/package.json "${pkgdir}/usr/lib/${pkgname}/package.json"
  install -Dm755 target/release/brazierd "${pkgdir}/usr/lib/${pkgname}/brazierd"
  install -Dm755 target/release/brazier-safety "${pkgdir}/usr/lib/${pkgname}/brazier-safety"
  # Ship an ordinary source copy. Settings installs the explicitly authorized,
  # root-owned setgid copy at /usr/lib/brazier-input-guard only on request.
  install -Dm755 target/release/brazier-input-guard "${pkgdir}/usr/lib/${pkgname}/brazier-input-guard"
  install -Dm644 apps/desktop/build/icon.png "${pkgdir}/usr/share/pixmaps/${pkgname}.png"
  install -Dm644 apps/desktop/build/icon.png "${pkgdir}/usr/share/icons/hicolor/1024x1024/apps/${pkgname}.png"
  install -Dm644 LICENSE "${pkgdir}/usr/share/licenses/${pkgname}/LICENSE"
  install -Dm644 THIRD_PARTY_NOTICES.md "${pkgdir}/usr/share/licenses/${pkgname}/THIRD_PARTY_NOTICES.md"

  install -dm755 "${pkgdir}/usr/bin"
  cat > "${pkgdir}/usr/bin/${pkgname}" <<'EOF'
#!/bin/sh
export BRAZIER_INSTALLED=1
# Prefer native Wayland. X11 software compositing on rootless XWayland
# fails to paint. Override with ELECTRON_OZONE_PLATFORM_HINT=x11.
if [ -z "${ELECTRON_OZONE_PLATFORM_HINT:-}" ]; then
  if [ "${XDG_SESSION_TYPE:-}" = "wayland" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    export ELECTRON_OZONE_PLATFORM_HINT=wayland
  else
    export ELECTRON_OZONE_PLATFORM_HINT=x11
  fi
fi
if [ "${ELECTRON_OZONE_PLATFORM_HINT}" = "x11" ]; then
  unset WAYLAND_DISPLAY
fi
exec /usr/bin/electron --class=brazier /usr/lib/brazier "$@"
EOF
  chmod 755 "${pkgdir}/usr/bin/${pkgname}"

  install -Dm644 /dev/stdin "${pkgdir}/usr/share/applications/${pkgname}.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Brazier
Comment=${pkgdesc}
Exec=${pkgname} %U
Icon=${pkgname}
Terminal=false
Categories=Utility;Development;
StartupWMClass=brazier
StartupNotify=true
EOF
}
