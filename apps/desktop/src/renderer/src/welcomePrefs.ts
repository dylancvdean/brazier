const WELCOME_KEY = 'brazier.welcomeCompleted.v1'

export function hasCompletedWelcome(): boolean {
  try {
    return localStorage.getItem(WELCOME_KEY) === '1'
  } catch {
    return false
  }
}

export function markWelcomeCompleted(): void {
  try {
    localStorage.setItem(WELCOME_KEY, '1')
  } catch {
    // Ignore quota errors — preference is best-effort.
  }
}

export function clearWelcomeCompleted(): void {
  try {
    localStorage.removeItem(WELCOME_KEY)
  } catch {
    // Ignore.
  }
}
