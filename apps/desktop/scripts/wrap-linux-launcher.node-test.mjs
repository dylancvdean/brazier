import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it } from 'node:test'
import assert from 'node:assert/strict'

import { linuxLauncherScript, wrapLinuxExecutable } from './wrap-linux-launcher.mjs'

describe('wrapLinuxExecutable', () => {
  it('renames the ELF binary and writes a wrapper that pins Ozone before exec', () => {
    const directory = mkdtempSync(join(tmpdir(), 'brazier-linux-wrap-'))
    const binary = join(directory, 'brazier')
    writeFileSync(binary, '\x7fELFfakelinuxbinary')
    chmodSync(binary, 0o755)

    wrapLinuxExecutable(directory, 'brazier')

    assert.equal(existsSync(join(directory, 'brazier.bin')), true)
    const script = readFileSync(binary, 'utf8')
    assert.equal(script, linuxLauncherScript('brazier'))
    assert.match(script, /ELECTRON_OZONE_PLATFORM_HINT=x11/)
    assert.match(script, /unset WAYLAND_DISPLAY/)
    assert.match(script, /brazier\.bin/)
    assert.match(script, /--class=brazier/)
  })

  it('is idempotent when the binary is already wrapped', () => {
    const directory = mkdtempSync(join(tmpdir(), 'brazier-linux-wrap-'))
    writeFileSync(join(directory, 'brazier.bin'), '\x7fELFfakelinuxbinary')
    writeFileSync(join(directory, 'brazier'), linuxLauncherScript('brazier'))

    wrapLinuxExecutable(directory, 'brazier')

    assert.equal(readFileSync(join(directory, 'brazier'), 'utf8'), linuxLauncherScript('brazier'))
    assert.equal(readFileSync(join(directory, 'brazier.bin'), 'utf8'), '\x7fELFfakelinuxbinary')
  })
})
