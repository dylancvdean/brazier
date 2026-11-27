export type AppView = 'chat' | 'settings' | 'pairing' | 'generate' | 'models';

export type BackStep = { view: AppView } | { agent: 'hidden' };

/**
 * Where the system back action goes from the current screen, or null to leave the app.
 * It mirrors each screen's own back button, so both always land in the same place.
 */
export function backStep({ view, agentShown, onDevice, hasHost }: {
  view: AppView; agentShown: boolean; onDevice: boolean; hasHost: boolean;
}): BackStep | null {
  // With nothing to return to, pairing is the first screen.
  if (view === 'pairing' || (!hasHost && !onDevice)) return hasHost || onDevice ? { view: 'settings' } : null;
  if (view === 'models') return { view: onDevice ? 'chat' : 'settings' };
  if (view !== 'chat') return { view: 'chat' };
  return agentShown && !onDevice ? { agent: 'hidden' } : null;
}

const scopeNames: Record<string, string> = {
  inference: 'Chat and generation', agent: 'Agent tasks', management: 'Host management',
};

/** The access an invite grants, in the words the app uses for those features. */
export function describeScopes(scopes: readonly string[]): string {
  return scopes.map((scope) => scopeNames[scope] ?? scope).join(', ') || 'None';
}

/** Bottom padding that lifts a screen above the keyboard, given the keyboard's top edge. */
export function keyboardInset(windowHeight: number, keyboardTop: number | null, resting: number): number {
  if (keyboardTop === null) return resting;
  return Math.max(resting, Math.round(windowHeight - keyboardTop));
}
