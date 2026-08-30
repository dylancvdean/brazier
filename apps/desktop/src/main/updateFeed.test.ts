import { describe, expect, it } from 'vitest'

import { pickReleaseManifestUrl, versionLooksLikePrerelease } from './updateFeed'

const mac = (url: string) => ({
  name: 'latest-mac.yml',
  browser_download_url: url
})

describe('pickReleaseManifestUrl', () => {
  it('skips drafts and, for stable apps, prereleases', () => {
    const url = pickReleaseManifestUrl(
      [
        { draft: true, prerelease: false, assets: [mac('https://example.invalid/draft/latest-mac.yml')] },
        { draft: false, prerelease: true, assets: [mac('https://example.invalid/beta/latest-mac.yml')] },
        { draft: false, prerelease: false, assets: [mac('https://example.invalid/stable/latest-mac.yml')] }
      ],
      'latest-mac.yml',
      false
    )
    expect(url).toBe('https://example.invalid/stable/latest-mac.yml')
  })

  it('includes prereleases when the running app is itself a prerelease', () => {
    const url = pickReleaseManifestUrl(
      [
        { draft: false, prerelease: true, assets: [mac('https://example.invalid/beta/latest-mac.yml')] },
        { draft: false, prerelease: false, assets: [mac('https://example.invalid/stable/latest-mac.yml')] }
      ],
      'latest-mac.yml',
      true
    )
    expect(url).toBe('https://example.invalid/beta/latest-mac.yml')
  })
})

describe('versionLooksLikePrerelease', () => {
  it('detects semver prerelease components', () => {
    expect(versionLooksLikePrerelease('0.2.13-beta.75')).toBe(true)
    expect(versionLooksLikePrerelease('0.3.0')).toBe(false)
  })
})
