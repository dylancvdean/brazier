import { fetchWelcomePreference, saveWelcomePreference } from './api'

const WELCOME_KEY = 'brazier.welcomeCompleted.v1'

function legacyWelcomeCompleted(): boolean {
  try {
    return localStorage.getItem(WELCOME_KEY) === '1'
  } catch {
    return false
  }
}

function storeLegacyWelcomeCompleted(): void {
  try {
    localStorage.setItem(WELCOME_KEY, '1')
  } catch {
    // Ignore quota errors — preference is best-effort.
  }
}

function clearLegacyWelcomeCompleted(): void {
  try {
    localStorage.removeItem(WELCOME_KEY)
  } catch {
    // Ignore.
  }
}

/**
 * Read the origin-independent daemon preference. A completed legacy
 * localStorage flag is promoted once so upgrades do not replay onboarding.
 */
export async function hasCompletedWelcome(): Promise<boolean> {
  const legacyCompleted = legacyWelcomeCompleted()
  try {
    const stored = await fetchWelcomePreference()
    if (stored.completed) {
      clearLegacyWelcomeCompleted()
      return true
    }
    if (legacyCompleted) {
      await saveWelcomePreference(true)
      clearLegacyWelcomeCompleted()
      return true
    }
    return false
  } catch {
    return legacyCompleted
  }
}

/**
 * Write locally first so an immediate quit still has a migration fallback,
 * then make the daemon database authoritative.
 */
export async function markWelcomeCompleted(): Promise<void> {
  storeLegacyWelcomeCompleted()
  try {
    await saveWelcomePreference(true)
    clearLegacyWelcomeCompleted()
  } catch {
    // Keep the local flag for the next launch to promote.
  }
}
