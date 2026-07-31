export type ComputerUseRecipe = {
  id: string
  label: string
  actionDialect: 'fara-xml' | 'generic-tools'
  recommendedEngine: string
  defaultTarget: 'browser' | 'desktop'
  modelMatchers: string[]
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
