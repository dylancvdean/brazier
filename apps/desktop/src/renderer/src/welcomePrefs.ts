import { fetchWelcomePreference, saveWelcomePreference } from './api'

export async function hasCompletedWelcome(): Promise<boolean> {
  return (await fetchWelcomePreference()).completed
}

export async function markWelcomeCompleted(): Promise<void> {
  await saveWelcomePreference(true)
}
