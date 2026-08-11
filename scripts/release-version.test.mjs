import assert from 'node:assert/strict'
import test from 'node:test'

import { nextBetaVersion, parseReleaseVersion } from './release-version.mjs'

test('advances an existing beta without changing its core version', () => {
  assert.equal(nextBetaVersion('0.2.13-beta.65'), '0.2.13-beta.66')
})

test('starts a patch beta after a stable release', () => {
  assert.equal(nextBetaVersion('1.4.2'), '1.4.3-beta.1')
})

test('core version bumps reset the beta counter', () => {
  assert.equal(nextBetaVersion('1.4.2-beta.9', '--major'), '2.0.0-beta.1')
  assert.equal(nextBetaVersion('1.4.2-beta.9', '--minor'), '1.5.0-beta.1')
  assert.equal(nextBetaVersion('1.4.2-beta.9', '--patch'), '1.4.3-beta.1')
})

test('rejects versions outside the supported release grammar', () => {
  assert.throws(() => parseReleaseVersion('1.2.3-rc.1'), /Expected a stable version/)
})
