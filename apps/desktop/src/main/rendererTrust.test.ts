import { describe, expect, it } from 'vitest'

import { isSafeExternalUrl, isTrustedRendererUrl } from './rendererTrust'

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
    expect(isTrustedRendererUrl('http://localhost:5173/', options)).toBe(false)
  })

  it('opens only credential-free HTTPS links outside the app', () => {
    expect(isSafeExternalUrl('https://example.com/consent')).toBe(true)
    expect(isSafeExternalUrl('https://user:secret@example.com/')).toBe(false)
    expect(isSafeExternalUrl('file:///tmp/secret')).toBe(false)
    expect(isSafeExternalUrl('javascript:alert(1)')).toBe(false)
    expect(isSafeExternalUrl('httpsx://example.com')).toBe(false)
  })
})
