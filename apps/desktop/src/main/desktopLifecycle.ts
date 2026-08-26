export type QuitDecision = 'start-cleanup' | 'wait-for-cleanup' | 'allow-exit'
export type ActivationDecision = 'show-window' | 'relaunch-after-cleanup'

/**
 * Small application-level state machine for the awkward interval where every
 * window is gone but owned child processes are still shutting down.
 */
export class DesktopLifecycle {
  private state: 'running' | 'quitting' | 'exiting' = 'running'
  private relaunchRequested = false

  get canShowWindow(): boolean {
    return this.state === 'running'
  }

  requestQuit(): QuitDecision {
    if (this.state === 'exiting') return 'allow-exit'
    if (this.state === 'quitting') return 'wait-for-cleanup'
    this.state = 'quitting'
    return 'start-cleanup'
  }

  requestActivation(): ActivationDecision {
    if (this.state === 'running') return 'show-window'
    // The secondary process already surrendered the single-instance lock. A
    // fresh launch after cleanup is the only way not to lose the user's click.
    this.relaunchRequested = true
    return 'relaunch-after-cleanup'
  }

  cleanupFinished(): { relaunch: boolean } {
    this.state = 'exiting'
    return { relaunch: this.relaunchRequested }
  }
}
