import { describe, expect, it } from 'vitest'

import { serverSettingsFailureIsRecoverable } from './serverSettingsFile'

describe('serverSettingsFailureIsRecoverable', () => {
  it('keeps a file whose API keys could not be decrypted', () => {
    expect(
      serverSettingsFailureIsRecoverable(
        new Error('Secure storage is unavailable for server API keys.')
      )
    ).toBe(true)
    expect(
      serverSettingsFailureIsRecoverable(Object.assign(new Error('denied'), { code: 'EACCES' }))
    ).toBe(true)
  })

  it('lets a missing or unparseable file be replaced', () => {
    expect(
      serverSettingsFailureIsRecoverable(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    ).toBe(false)
    let syntax: unknown
    try {
      JSON.parse('{')
    } catch (cause) {
      syntax = cause
    }
    expect(serverSettingsFailureIsRecoverable(syntax)).toBe(false)
  })
})
