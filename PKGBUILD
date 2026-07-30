# Maintainer: Dylan C. V. Dean <dylan@dylancvdean.com>

pkgname=brazier
# Pacman reserves `-` as the separator before pkgrel, so encode upstream
# prerelease separators with `_` in pkgver.
pkgver=0.2.9_beta.29
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
  cargo build --release --locked -p brazierd
}

package() {
  cd "${srcdir}/${pkgname}"

  install -dm755 "${pkgdir}/usr/lib/${pkgname}"
  cp -a apps/desktop/out "${pkgdir}/usr/lib/${pkgname}/"
  # The agent worker intentionally keeps its JavaScript dependencies external.
  # Dereference pnpm's workspace links so they resolve inside /usr/lib/brazier.
  cp -aL apps/desktop/node_modules "${pkgdir}/usr/lib/${pkgname}/"
  install -Dm644 apps/desktop/package.json "${pkgdir}/usr/lib/${pkgname}/package.json"
  install -Dm755 target/release/brazierd "${pkgdir}/usr/lib/${pkgname}/brazierd"
  install -Dm644 apps/desktop/build/icon.png "${pkgdir}/usr/share/pixmaps/${pkgname}.png"
  install -Dm644 apps/desktop/build/icon.png "${pkgdir}/usr/share/icons/hicolor/1024x1024/apps/${pkgname}.png"
  install -Dm644 LICENSE "${pkgdir}/usr/share/licenses/${pkgname}/LICENSE"
  install -Dm644 THIRD_PARTY_NOTICES.md "${pkgdir}/usr/share/licenses/${pkgname}/THIRD_PARTY_NOTICES.md"

  install -dm755 "${pkgdir}/usr/bin"
  cat > "${pkgdir}/usr/bin/${pkgname}" <<'EOF'
#!/bin/sh
export BRAZIER_INSTALLED=1
# Brazier currently uses XWayland on Linux for a reliable Chromium render path.
# Setting this before Electron starts makes `--class` apply to the actual
# window, allowing Plasma to associate it with brazier.desktop.
export ELECTRON_OZONE_PLATFORM_HINT=x11
unset WAYLAND_DISPLAY
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
