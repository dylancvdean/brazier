import { describe, expect, it } from 'vitest'

import {
  configYamlWithModelRoles,
  configYamlWithSettings,
  ompSettingsYaml,
  sanitizeOmpSettings
} from './ompSettings'

describe('ompSettingsYaml', () => {
  it('emits only explicitly-set keys, grouping nested paths under one key', () => {
    expect(
      ompSettingsYaml({
        'retry.enabled': false,
        'retry.maxRetries': 5,
        temperature: 0.7,
        'memory.backend': 'local'
      })
    ).toBe([
      'temperature: 0.7',
      'retry:',
      '  enabled: false',
      '  maxRetries: 5',
      'memory:',
      '  backend: local'
    ].join('\n'))
  })

  it('emits an empty string when nothing is set', () => {
    expect(ompSettingsYaml({})).toBe('')
  })

  it('formats booleans and numbers in YAML scalar form', () => {
    expect(ompSettingsYaml({ 'advisor.enabled': true, 'tools.maxTimeout': 0 })).toBe(
      'tools:\n  maxTimeout: 0\nadvisor:\n  enabled: true'
    )
  })
})

describe('configYamlWithSettings', () => {
  it('strips owned top-level keys from freeform YAML and appends the settings block', () => {
    const merged = configYamlWithSettings(
      'theme:\n  dark: titanium\nretry:\n  enabled: true\n  maxRetries: 99',
      { 'retry.enabled': false, 'lsp.enabled': true }
    )
    expect(merged).toContain('theme:\n  dark: titanium')
    expect(merged).toContain('retry:\n  enabled: false')
    expect(merged).not.toContain('maxRetries: 99')
    expect(merged).toContain('lsp:\n  enabled: true')
  })

  it('leaves freeform YAML untouched when no settings are set', () => {
    expect(configYamlWithSettings('theme:\n  dark: titanium', {})).toBe('theme:\n  dark: titanium')
  })
})

describe('configYamlWithModelRoles', () => {
  it('replaces a prior modelRoles block with the profile assignments', () => {
    const merged = configYamlWithModelRoles('modelRoles:\n  smol: spark/x\nlsp:\n  enabled: true', {
      smol: 'gguf:smol/model.gguf'
    })
    expect(merged).toContain('modelRoles:\n  smol: brazier/gguf:smol/model.gguf')
    expect(merged).toContain('lsp:\n  enabled: true')
    expect(merged).not.toContain('spark/x')
  })
})

describe('sanitizeOmpSettings', () => {
  it('keeps only known paths and coerces values to their declared types', () => {
    expect(
      sanitizeOmpSettings({
        'retry.enabled': true,
        'retry.maxRetries': 5,
        temperature: 0.7,
        'edit.mode': 'hashline',
        unknown: 'dropped',
        'edit.fuzzyMatch': 'not-a-boolean',
        'memory.backend': 'nope'
      })
    ).toEqual({
      'retry.enabled': true,
      'retry.maxRetries': 5,
      temperature: 0.7,
      'edit.mode': 'hashline'
    })
  })

  it('tolerates missing or malformed input', () => {
    expect(sanitizeOmpSettings(undefined)).toEqual({})
    expect(sanitizeOmpSettings('nope')).toEqual({})
    expect(sanitizeOmpSettings(null)).toEqual({})
  })
})
