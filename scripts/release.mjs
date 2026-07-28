import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
// pnpm forwards its argument separator, so `pnpm release -- --patch` reaches
// this script as both `--` and `--patch`.
const args = process.argv.slice(2).filter((argument) => argument !== '--')
const bump = args[0] ?? 'beta'

if (!['beta', '--major', '--minor', '--patch'].includes(bump) || args.length > 1) {
  console.error('Usage: pnpm release -- [--major|--minor|--patch]')
  process.exit(1)
}

function git(args, options = {}) {
  const output = execFileSync('git', args, { cwd: root, encoding: 'utf8', ...options })
  return typeof output === 'string' ? output.trim() : ''
}

if (git(['status', '--porcelain']).length > 0) {
  console.error('Refusing to package a release from an unclean checkout. Commit or stash every change first.')
  process.exit(1)
}

const version = JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8')).version
const match = /^(?<major>0|[1-9]\d*)\.(?<minor>0|[1-9]\d*)\.(?<patch>0|[1-9]\d*)(?:-beta\.(?<beta>0|[1-9]\d*))?$/.exec(version)
if (!match?.groups) {
  console.error(`Expected a stable version or -beta.N prerelease, received ${version}.`)
  process.exit(1)
}

let major = Number(match.groups.major)
let minor = Number(match.groups.minor)
let patch = Number(match.groups.patch)
const beta = match.groups.beta == null ? null : Number(match.groups.beta)

switch (bump) {
  case '--major':
    major += 1
    minor = 0
    patch = 0
    break
  case '--minor':
    minor += 1
    patch = 0
    break
  case '--patch':
    patch += 1
    break
  default:
    // A beta-only release advances its prerelease counter. Once stable, the
    // natural no-flag action is a stable patch release; it never revives beta.
    if (beta == null) patch += 1
}

const nextBeta = beta == null ? null : beta + 1
const next = `${major}.${minor}.${patch}${nextBeta == null ? '' : `-beta.${nextBeta}`}`
const tag = `v${next}`

try {
  git(['rev-parse', '--verify', '--quiet', `refs/tags/${tag}`], { stdio: 'ignore' })
  console.error(`Refusing to overwrite existing local tag ${tag}.`)
  process.exit(1)
} catch {
  // `rev-parse` exits nonzero when the tag is available.
}

console.log(`Preparing Brazier ${next} from ${version}.`)
execFileSync('pnpm', ['version:bump', '--', next], { cwd: root, stdio: 'inherit' })

// The release tag must point at the commit whose manifests carry that exact
// version. Stage only tracked bump output: the checkout was verified clean
// before this command, so this cannot accidentally absorb unrelated files.
if (git(['status', '--porcelain']).length === 0) {
  throw new Error('Version bump did not change any tracked files.')
}
git(['add', '-u'], { stdio: 'inherit' })
git(['commit', '-m', `release: ${tag}`], { stdio: 'inherit' })
git(['tag', '-a', tag, '-m', `Brazier ${tag}`], { stdio: 'inherit' })
git(['push', 'origin', 'HEAD', tag], { stdio: 'inherit' })
console.log(`Released ${tag}.`)
