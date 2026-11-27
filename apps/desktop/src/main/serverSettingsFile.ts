/**
 * Whether a failure to open saved server settings leaves a file worth keeping.
 *
 * A missing or syntactically broken file holds nothing to recover. Any other
 * failure (a locked keyring, an undecryptable key, a read error) can be
 * transient, so the file must not be replaced with defaults.
 */
export function serverSettingsFailureIsRecoverable(cause: unknown): boolean {
  if ((cause as NodeJS.ErrnoException | null)?.code === 'ENOENT') return false
  return !(cause instanceof SyntaxError)
}
