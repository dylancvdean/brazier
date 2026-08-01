export type ComputerUseRecipe = {
  id: string
  label: string
  actionDialect: 'fara-xml' | 'generic-tools'
  recommendedEngine: string
  defaultTarget: 'browser' | 'desktop'
  modelMatchers: string[]
}

export type ManagedFaraBundle = {
  id: string
  label: string
  summary: string
  sourceRepo: string
  quantRepo: string
  modelFile: string
  projectorFile: string
  downloadBytes: number
  recommended?: boolean
}

/**
 * Turnkey llama.cpp bundles: one balanced Q4_K_M checkpoint plus the matching
 * F16 vision projector. The base models are Microsoft's MIT-licensed Fara1.5
 * releases; these GGUF conversions are published by bartowski.
 */
export const MANAGED_FARA_BUNDLES: ManagedFaraBundle[] = [
  {
    id: 'fara1.5-4b-q4',
    label: 'Fara1.5 4B',
    summary: 'Fastest and smallest. A good first install for browser tasks.',
    sourceRepo: 'microsoft/Fara1.5-4B',
    quantRepo: 'bartowski/Fara1.5-4B-GGUF',
    modelFile: 'Fara1.5-4B-Q4_K_M.gguf',
    projectorFile: 'mmproj-Fara1.5-4B-f16.gguf',
    downloadBytes: 3_557_274_368,
    recommended: true
  },
  {
    id: 'fara1.5-9b-q4',
    label: 'Fara1.5 9B',
    summary: 'Balanced quality and speed for more reliable multi-step work.',
    sourceRepo: 'microsoft/Fara1.5-9B',
    quantRepo: 'bartowski/Fara1.5-9B-GGUF',
    modelFile: 'Fara1.5-9B-Q4_K_M.gguf',
    projectorFile: 'mmproj-Fara1.5-9B-f16.gguf',
    downloadBytes: 6_828_949_152
  },
  {
    id: 'fara1.5-27b-q4',
    label: 'Fara1.5 27B',
    summary: 'Highest quality, with much heavier memory and compute requirements.',
    sourceRepo: 'microsoft/Fara1.5-27B',
    quantRepo: 'bartowski/Fara1.5-27B-GGUF',
    modelFile: 'Fara1.5-27B-Q4_K_M.gguf',
    projectorFile: 'mmproj-Fara1.5-27B-f16.gguf',
    downloadBytes: 18_461_159_904
  }
]

export function modelIdForManagedFara(bundle: ManagedFaraBundle): string {
  return `gguf:${bundle.quantRepo}/${bundle.modelFile}`
}

/**
 * Known computer-use model recipes. Matchers are lowercase substrings checked
 * against model ids / library labels.
 */
export const COMPUTER_USE_RECIPES: ComputerUseRecipe[] = [
  {
    id: 'fara1.5-4b',
    label: 'Fara1.5 4B',
    actionDialect: 'fara-xml',
    recommendedEngine: 'vllm',
    defaultTarget: 'browser',
    modelMatchers: ['fara1.5-4b', 'fara1.5_4b', 'microsoft/fara1.5-4b']
  },
  {
    id: 'fara1.5-9b',
    label: 'Fara1.5 9B',
    actionDialect: 'fara-xml',
    recommendedEngine: 'vllm',
    defaultTarget: 'browser',
    modelMatchers: ['fara1.5-9b', 'fara1.5_9b', 'microsoft/fara1.5-9b']
  },
  {
    id: 'fara1.5-27b',
    label: 'Fara1.5 27B',
    actionDialect: 'fara-xml',
    recommendedEngine: 'vllm',
    defaultTarget: 'browser',
    modelMatchers: ['fara1.5-27b', 'fara1.5_27b', 'microsoft/fara1.5-27b']
  }
]

export function recipeForModel(modelId: string): ComputerUseRecipe | null {
  const lower = modelId.toLowerCase()
  return (
    COMPUTER_USE_RECIPES.find((recipe) =>
      recipe.modelMatchers.some((matcher) => lower.includes(matcher))
    ) ?? null
  )
}
