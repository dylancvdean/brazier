#!/usr/bin/env node
/**
 * Ensure the Electron binary was fully extracted. Partial extracts leave
 * only files like snapshot_blob.bin and cause electron-vite's
 * "Electron uninstall" error.
 *
 * Electron's own install.js uses extract-zip, which can hang or partially
 * extract on some Linux setups. We download with @electron/get and extract
 * with unzip on macOS (preserves framework symlinks) or Python's zipfile on
 * Linux (reliable fallback, but breaks macOS .framework symlinks).
 */
import { createRequire } from 'node:module'
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync
} from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const scriptRequire = createRequire(import.meta.url)

function electronPackageRoot() {
  try {
    return dirname(scriptRequire.resolve('electron/package.json'))
  } catch {
    try {
      const desktopRequire = createRequire(
        join(fileURLToPath(new URL('..', import.meta.url)), 'apps/desktop/package.json')
      )
      return dirname(desktopRequire.resolve('electron/package.json'))
    } catch {
      return null
    }
  }
}

function platformBinaryName() {
  if (process.platform === 'win32') return 'electron.exe'
  if (process.platform === 'darwin') return 'Electron.app/Contents/MacOS/Electron'
  return 'electron'
}

function extractWithUnzip(zipPath, destDir) {
  const result = spawnSync('unzip', ['-o', '-q', zipPath, '-d', destDir], {
    encoding: 'utf8'
  })
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || 'unzip failed')
  }
}

function extractWithPython(zipPath, destDir) {
  const result = spawnSync(
    'python3',
    [
      '-c',
      `
import zipfile, sys
from pathlib import Path
zip_path, dest = sys.argv[1], Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(dest)
`,
      zipPath,
      destDir
    ],
    { encoding: 'utf8' }
  )
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || 'python extract failed')
  }
}

function extractZip(zipPath, destDir) {
  if (process.platform === 'darwin') {
    extractWithUnzip(zipPath, destDir)
    return
  }
  extractWithPython(zipPath, destDir)
}

async function downloadAndExtract(root, version) {
  const requireFromElectron = createRequire(join(root, 'package.json'))
  const { downloadArtifact } = requireFromElectron('@electron/get')
  const checksums = requireFromElectron('./checksums.json')
  const zipPath = await downloadArtifact({
    version,
    artifactName: 'electron',
    force: process.env.force_no_cache === 'true',
    checksums,
    platform: process.env.npm_config_platform || process.platform,
    arch: process.env.npm_config_arch || process.arch
  })

  const dist = join(root, 'dist')
  rmSync(dist, { recursive: true, force: true })
  mkdirSync(dist, { recursive: true })
  extractZip(zipPath, dist)

  writeFileSync(join(root, 'path.txt'), platformBinaryName())
  const binary = join(dist, platformBinaryName())
  if (existsSync(binary) && process.platform !== 'win32') {
    try {
      chmodSync(binary, 0o755)
    } catch {
      // best-effort
    }
  }
}

function isDarwinFrameworkHealthy(root) {
  const frameworkRoot = join(
    root,
    'dist/Electron.app/Contents/Frameworks/Electron Framework.framework'
  )
  const current = join(frameworkRoot, 'Versions/Current')
  const frameworkBinary = join(frameworkRoot, 'Electron Framework')
  try {
    return (
      lstatSync(current).isSymbolicLink() &&
      (lstatSync(frameworkBinary).isSymbolicLink() ||
        spawnSync('file', ['-b', frameworkBinary], { encoding: 'utf8' }).stdout.includes(
          'Mach-O'
        ))
    )
  } catch {
    return false
  }
}

function isHealthy(root, version) {
  const platformPath = platformBinaryName()
  const binary = join(root, 'dist', platformPath)
  const versionFile = join(root, 'dist', 'version')
  const pathTxt = join(root, 'path.txt')
  try {
    const baseHealthy =
      existsSync(binary) &&
      existsSync(pathTxt) &&
      readFileSync(versionFile, 'utf8').replace(/^v/, '').trim() === version &&
      readFileSync(pathTxt, 'utf8').trim() === platformPath
    if (!baseHealthy) {
      return false
    }
    if (process.platform === 'darwin') {
      return isDarwinFrameworkHealthy(root)
    }
    return true
  } catch {
    return false
  }
}

async function main() {
  const electronRoot = electronPackageRoot()
  if (!electronRoot) {
    console.warn('[ensure-electron] electron package not installed yet; skip')
    return
  }

  const { version } = JSON.parse(readFileSync(join(electronRoot, 'package.json'), 'utf8'))
  if (isHealthy(electronRoot, version)) {
    return
  }

  console.log(`[ensure-electron] repairing incomplete Electron ${version} install…`)
  await downloadAndExtract(electronRoot, version)

  const binary = join(electronRoot, 'dist', platformBinaryName())
  if (!existsSync(binary)) {
    throw new Error(`still missing ${binary}`)
  }
  console.log(`[ensure-electron] ready: ${binary}`)
}

main().catch((error) => {
  console.error('[ensure-electron] repair failed:', error)
  process.exitCode = 1
})
