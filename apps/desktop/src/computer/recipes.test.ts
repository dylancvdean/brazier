import { describe, expect, it } from 'vitest'

import { MANAGED_FARA_BUNDLES, modelIdForManagedFara, recipeForModel } from './recipes'

describe('computer-use recipes', () => {
  it('ships complete managed model and projector pairs for every Fara1.5 size', () => {
    expect(MANAGED_FARA_BUNDLES.map((bundle) => bundle.label)).toEqual([
      'Fara1.5 4B',
      'Fara1.5 9B',
      'Fara1.5 27B'
    ])
    for (const bundle of MANAGED_FARA_BUNDLES) {
      expect(bundle.modelFile).toMatch(/Q4_K_M\.gguf$/)
      expect(bundle.projectorFile).toMatch(/^mmproj-.*-f16\.gguf$/)
      expect(modelIdForManagedFara(bundle)).toBe(
        `gguf:${bundle.quantRepo}/${bundle.modelFile}`
      )
      expect(recipeForModel(modelIdForManagedFara(bundle))?.actionDialect).toBe('fara-xml')
    }
  })
})
