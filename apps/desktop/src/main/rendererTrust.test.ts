import { describe, expect, it } from 'vitest'

import {
  isRendererDevelopmentOrigin,
  isSafeExternalUrl,
  isTrustedRendererUrl,
  shouldCancelRendererNetworkRequest
} from './rendererTrust'

const packagedIndexPath = '/Applications/Brazier.app/Contents/Resources/app.asar/out/renderer/index.html'

describe('renderer trust boundary', () => {
  it('accepts only the exact packaged renderer file, with an optional hash', () => {
    const options = { packagedIndexPath }
    expect(
      isTrustedRendererUrl(
        'file:///Applications/Brazier.app/Contents/Resources/app.asar/out/renderer/index.html',
        options
      )
    ).toBe(true)
    expect(
      isTrustedRendererUrl(
        'file:///Applications/Brazier.app/Contents/Resources/app.asar/out/renderer/index.html#voice',
        options
      )
    ).toBe(true)
    expect(isTrustedRendererUrl('file:///tmp/attacker.html', options)).toBe(false)
  })

  it('rejects development prefix lookalikes and alternate paths', () => {
    const options = { developmentUrl: 'http://127.0.0.1:5173/', packagedIndexPath }
    expect(isTrustedRendererUrl('http://127.0.0.1:5173/', options)).toBe(true)
    expect(isTrustedRendererUrl('http://127.0.0.1:5173.evil.test/', options)).toBe(false)
    expect(isTrustedRendererUrl('http://127.0.0.1:5173/attacker.html', options)).toBe(false)
    expect(isTrustedRendererUrl('http://127.0.0.1:9999/', options)).toBe(false)
  })

  it('treats loopback aliases as the same Vite document', () => {
    const options = { developmentUrl: 'http://127.0.0.1:5173/', packagedIndexPath }
    expect(isTrustedRendererUrl('http://localhost:5173/', options)).toBe(true)
    expect(isTrustedRendererUrl('http://[::1]:5173/', options)).toBe(true)
    expect(isTrustedRendererUrl('http://localhost:5173/', { developmentUrl: 'http://localhost:5173', packagedIndexPath })).toBe(true)
    expect(isTrustedRendererUrl('http://127.0.0.1:5173/attacker.html', options)).toBe(false)
    expect(isTrustedRendererUrl('http://localhost:9999/', options)).toBe(false)
  })

  it('allows Vite and HMR traffic across loopback aliases of the development origin', () => {
    expect(isRendererDevelopmentOrigin('http://localhost:5173/src/main.tsx', 'http://127.0.0.1:5173')).toBe(true)
    expect(isRendererDevelopmentOrigin('ws://[::1]:5173/', 'http://127.0.0.1:5173')).toBe(true)
    expect(isRendererDevelopmentOrigin('http://127.0.0.1:5173/', 'http://localhost:5173')).toBe(true)
    expect(isRendererDevelopmentOrigin('http://localhost:9999/', 'http://127.0.0.1:5173')).toBe(false)
    expect(isRendererDevelopmentOrigin('https://127.0.0.1:5173/', 'http://127.0.0.1:5173')).toBe(false)
    expect(isRendererDevelopmentOrigin('http://user:secret@127.0.0.1:5173/', 'http://127.0.0.1:5173')).toBe(false)
  })

  it('does not cancel the Vite shell document even when the profile guard would', () => {
    const allows = (): boolean => false
    expect(
      shouldCancelRendererNetworkRequest(
        { url: 'http://127.0.0.1:5173/', webContentsId: 1, resourceType: 'mainFrame' },
        'http://127.0.0.1:5173',
        allows
      )
    ).toBe(false)
    expect(
      shouldCancelRendererNetworkRequest(
        { url: 'http://127.0.0.1:5173/src/main.tsx', webContentsId: 1, resourceType: 'script' },
        'http://127.0.0.1:5173',
        allows
      )
    ).toBe(false)
    expect(
      shouldCancelRendererNetworkRequest(
        { url: 'http://127.0.0.1:9/private', webContentsId: 1, resourceType: 'xhr' },
        'http://127.0.0.1:5173',
        allows
      )
    ).toBe(true)
    expect(
      shouldCancelRendererNetworkRequest(
        { url: 'http://127.0.0.1:9/private', resourceType: 'xhr' },
        'http://127.0.0.1:5173',
        allows
      )
    ).toBe(false)
  })

  it('opens only credential-free HTTPS links outside the app', () => {
    expect(isSafeExternalUrl('https://example.com/consent')).toBe(true)
    expect(isSafeExternalUrl('https://user:secret@example.com/')).toBe(false)
    expect(isSafeExternalUrl('file:///tmp/secret')).toBe(false)
    expect(isSafeExternalUrl('javascript:alert(1)')).toBe(false)
    expect(isSafeExternalUrl('httpsx://example.com')).toBe(false)
  })
})
