/**
 * Pin Ozone before Chromium reads the environment.
 *
 * Electron JS can only append switches before app.ready. Ozone/Wayland is
 * chosen earlier from WAYLAND_DISPLAY, so a Plasma/AppImage launch without a
 * wrapper can paint on the wrong platform. The Arch launcher already exports
 * these; wrap the unpacked Linux binary so AppImage and unpacked dirs match.
 */
import { existsSync, openSync, readSync, closeSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

function startsWithShebang(path) {
  const fd = openSync(path, 'r')
  try {
    const buffer = Buffer.alloc(2)
    const bytes = readSync(fd, buffer, 0, 2, 0)
    return bytes >= 2 && buffer[0] === 0x23 && buffer[1] === 0x21
  } finally {
    closeSync(fd)
  }
}

export function linuxLauncherScript(executableName) {
  return `#!/bin/sh
if [ -z "\${ELECTRON_OZONE_PLATFORM_HINT:-}" ]; then
  export ELECTRON_OZONE_PLATFORM_HINT=x11
fi
if [ "\${ELECTRON_OZONE_PLATFORM_HINT}" = "x11" ]; then
  unset WAYLAND_DISPLAY
fi
here=$(dirname -- "$0")
exec "$here/${executableName}.bin" --class=${executableName} "$@"
`
}

export function wrapLinuxExecutable(appOutDir, executableName) {
  const executable = join(appOutDir, executableName)
  const real = `${executable}.bin`
  if (!existsSync(executable)) {
    throw new Error(`Linux launcher wrap: missing ${executable}`)
  }
  if (existsSync(real)) return
  if (startsWithShebang(executable)) return
  renameSync(executable, real)
  writeFileSync(executable, linuxLauncherScript(executableName), { mode: 0o755 })
}

export async function afterPack(context) {
  if (context.electronPlatformName !== 'linux') return
  const executableName = context.packager.executableName
  wrapLinuxExecutable(context.appOutDir, executableName)
}

export default afterPack
