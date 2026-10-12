import { describe, expect, it } from 'vitest'
import type { LocalModel } from '../../api'
import {
  libraryDiskBytes,
  libraryModelMatches,
  modelFolderKey,
  sortLibraryModels
} from './library'

function model(partial: Partial<LocalModel> & Pick<LocalModel, 'id'>): LocalModel {
  return {
    object: 'model',
    owned_by: 'brazier:llama.cpp',
    ...partial
  }
}

describe('sortLibraryModels', () => {
  const models = [
    model({ id: 'gguf:acme/small/model-q4.gguf', size_bytes: 1_000 }),
    model({
      id: 'gguf:acme/vision/model-q4.gguf',
      size_bytes: 4_000,
      projectors: [{ name: 'mmproj-f16.gguf', size_bytes: 2_000 }]
    }),
    model({ id: 'gguf:acme/huge/model-f16.gguf', size_bytes: 9_000 })
  ]

  it('sorts by weight plus projector, largest first', () => {
    expect(sortLibraryModels(models, 'largest').map((entry) => entry.id)).toEqual([
      'gguf:acme/huge/model-f16.gguf',
      'gguf:acme/vision/model-q4.gguf',
      'gguf:acme/small/model-q4.gguf'
    ])
  })

  it('sorts by name when sizes are not the point', () => {
    expect(sortLibraryModels(models, 'name').map((entry) => entry.id)).toEqual([
      'gguf:acme/huge/model-f16.gguf',
      'gguf:acme/small/model-q4.gguf',
      'gguf:acme/vision/model-q4.gguf'
    ])
  })
})

describe('libraryDiskBytes', () => {
  it('counts a projector shared by two quants once', () => {
    const shared = [{ name: 'mmproj-f16.gguf', size_bytes: 500 }]
    const models = [
      model({ id: 'gguf:acme/vision/model-q4.gguf', size_bytes: 1_000, projectors: shared }),
      model({ id: 'gguf:acme/vision/model-q8.gguf', size_bytes: 2_000, projectors: shared }),
      model({ id: 'mlx:acme/text', size_bytes: 100, engine: 'mlx-lm' })
    ]
    expect(modelFolderKey(models[0])).toBe(modelFolderKey(models[1]))
    expect(libraryDiskBytes(models)).toBe(3_600)
  })

  it('counts projectors in the library root for external files', () => {
    const models = [
      model({
        id: 'gguf-ext:0:a.gguf',
        size_bytes: 10,
        projectors: [{ name: 'mmproj.gguf', size_bytes: 4 }]
      }),
      model({
        id: 'gguf-ext:0:b.gguf',
        size_bytes: 20,
        projectors: [{ name: 'mmproj.gguf', size_bytes: 4 }]
      })
    ]
    expect(libraryDiskBytes(models)).toBe(34)
  })
})

describe('libraryModelMatches', () => {
  it('matches a projector name and ignores blank queries', () => {
    const entry = model({
      id: 'gguf:acme/vision/model-q4.gguf',
      projectors: [{ name: 'mmproj-f16.gguf', size_bytes: 1 }]
    })
    expect(libraryModelMatches(entry, '  ')).toBe(true)
    expect(libraryModelMatches(entry, 'MMPROJ')).toBe(true)
    expect(libraryModelMatches(entry, 'missing')).toBe(false)
  })
})
