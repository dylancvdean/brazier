import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { nextBetaVersion, parseReleaseVersion } from './release-version.mjs'

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
// pnpm forwards its argument separator, so filter it before parsing modes.
const args = process.argv.slice(2).filter((argument) => argument !== '--')
const mode = args[0]
const bump = args[1] ?? 'beta'

if (
  !['prepare', 'publish'].includes(mode) ||
  (mode === 'prepare' && (!['beta', '--major', '--minor', '--patch'].includes(bump) || args.length > 2)) ||
  (mode === 'publish' && args.length > 1)
) {
  console.error(
    'Usage: pnpm release:prepare -- [--major|--minor|--patch]\n' +
    '       pnpm release:publish'
  )
  process.exit(1)
}

function git(args, options = {}) {
  const output = execFileSync('git', args, { cwd: root, encoding: 'utf8', ...options })
  return typeof output === 'string' ? output.trim() : ''
}

function requireCleanCheckout() {
  if (git(['status', '--porcelain']).length === 0) return
  console.error('Refusing to package a release from an unclean checkout. Commit or stash every change first.')
  process.exit(1)
}

function currentVersion() {
  return JSON.parse(readFileSync(resolve(root, 'package.json'), 'utf8')).version
}

function parsedVersion(version) {
  try {
    return parseReleaseVersion(version)
  } catch (cause) {
    console.error(cause instanceof Error ? cause.message : String(cause))
    process.exit(1)
  }
}

function requireAvailableTag(tag) {
  try {
    git(['rev-parse', '--verify', '--quiet', `refs/tags/${tag}`], { stdio: 'ignore' })
    console.error(`Refusing to overwrite existing local tag ${tag}.`)
    process.exit(1)
  } catch {
    // `rev-parse` exits nonzero when the tag is available.
  }
}

function prepare() {
  requireCleanCheckout()
  const version = currentVersion()
  parsedVersion(version)
  const next = nextBetaVersion(version, bump)
  const tag = `v${next}`
  requireAvailableTag(tag)

  console.log(`Preparing Brazier candidate ${next} from ${version}.`)
  execFileSync('pnpm', ['version:bump', '--', next], { cwd: root, stdio: 'inherit' })
  if (git(['status', '--porcelain']).length === 0) {
    throw new Error('Version bump did not change any tracked files.')
  }
  // The checkout was clean above, so stage only the tracked version outputs.
  git(['add', '-u'], { stdio: 'inherit' })
  git(['commit', '-m', `release: prepare ${tag}`], { stdio: 'inherit' })
  git(['push', 'origin', 'HEAD'], { stdio: 'inherit' })
  const commit = git(['rev-parse', 'HEAD'])
  console.log(
    `Prepared and pushed ${tag} candidate ${commit}.\n` +
    'Run both exact-commit voice qualifications and upload their evidence, then run `pnpm release:publish`.'
  )
}

function publish() {
  requireCleanCheckout()
  const version = currentVersion()
  parsedVersion(version)
  const tag = `v${version}`
  requireAvailableTag(tag)
  const head = git(['rev-parse', 'HEAD'])
  let upstream
  try {
    upstream = git(['rev-parse', '@{upstream}'])
  } catch {
    console.error('The candidate branch has no upstream. Push the exact candidate commit before publishing.')
    process.exit(1)
  }
  if (head !== upstream) {
    console.error(`Candidate ${head} is not the exact pushed upstream commit ${upstream}.`)
    process.exit(1)
  }
  git(['tag', '-a', tag, '-m', `Brazier ${tag}`], { stdio: 'inherit' })
  git(['push', 'origin', tag], { stdio: 'inherit' })
  console.log(`Published ${tag}; the gated release workflow now owns packaging and publication.`)
}

if (mode === 'prepare') prepare()
else publish()
