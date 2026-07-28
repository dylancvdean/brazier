import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const version = process.argv.slice(2).find((argument) => argument !== '--')
const semver = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/

if (!version || !semver.test(version)) {
  console.error('Usage: pnpm version:bump -- <semver>')
  process.exit(1)
}

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))

function replace(file, pattern, replacement) {
  const path = resolve(root, file)
  const source = readFileSync(path, 'utf8')
  if (!pattern.test(source)) {
    throw new Error(`Could not find the version field in ${file}`)
  }
  writeFileSync(path, source.replace(pattern, replacement))
}

replace('Cargo.toml', /^version = ".+"$/m, `version = "${version}"`)
replace('package.json', /"version": "[^"]+"/, `"version": "${version}"`)
replace('apps/desktop/package.json', /"version": "[^"]+"/, `"version": "${version}"`)
replace('crates/brazier-runtime/python/streaming_asr_pkg/pyproject.toml', /^version = ".+"$/m, `version = "${version}"`)
replace('crates/brazier-runtime/python/streaming_asr_pkg/brazier_streaming_asr/__init__.py', /^__version__ = ".+"$/m, `__version__ = "${version}"`)
replace('PKGBUILD', /^pkgver=.+$/m, `pkgver=${version}`)

// Workspace package versions are recorded in Cargo.lock. Regenerate it while
// keeping dependency resolution offline so a version bump cannot upgrade deps.
execFileSync('cargo', ['generate-lockfile', '--offline'], { cwd: root, stdio: 'inherit' })

console.log(`Bumped Brazier to ${version}. Review and commit the changed files.`)
