import { describe, expect, it } from 'vitest'

import { fallbackRuntimeForEngine, runtimeNoticeForModel } from './model-utils'
import type { LocalModel, RuntimeEntry } from './api'

function model(id: string, engine: string): LocalModel {
  return {
    id,
    object: 'model',
    owned_by: `brazier:${engine}`,
    engine
  }
}

function runtime(partial: Partial<RuntimeEntry> & Pick<RuntimeEntry, 'id' | 'engine'>): RuntimeEntry {
  return {
    kind: 'managed',
    label: partial.id,
    target: 'cpu',
    version: 'b1',
    path: `/tmp/${partial.id}`,
    active: false,
    deletable: true,
    ...partial
  }
}

describe('runtimeNoticeForModel', () => {
  const models = [model('gguf:acme/a.gguf', 'llama.cpp')]

  it('explains a missing pairing when another runtime can take over', () => {
    const runtimes = [runtime({ id: 'managed', engine: 'llama.cpp', active: true, label: 'llama.cpp · CPU' })]
    expect(
      runtimeNoticeForModel('gguf:acme/a.gguf', models, runtimes, {
        'gguf:acme/a.gguf': 'source-gone'
      })
    ).toBe('The paired runtime was removed. This model will use llama.cpp · CPU.')
  })

  it('asks the user to choose when nothing remains for that engine', () => {
    expect(
      runtimeNoticeForModel('gguf:acme/a.gguf', models, [], {
        'gguf:acme/a.gguf': 'source-gone'
      })
    ).toBe('The paired runtime is missing. Choose another runtime below.')
  })
})

describe('fallbackRuntimeForEngine', () => {
  it('prefers the active runtime, then a managed install', () => {
    const runtimes = [
      runtime({ id: 'source-old', engine: 'llama.cpp', kind: 'source', active: false }),
      runtime({ id: 'managed', engine: 'llama.cpp', kind: 'managed', active: true })
    ]
    expect(fallbackRuntimeForEngine(runtimes, 'llama.cpp')?.id).toBe('managed')
  })
})
