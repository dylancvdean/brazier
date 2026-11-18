/**
 * Repetitive tool-call loop guard for the native agent harness.
 *
 * Tracks `(tool, canonicalArgsJson, isError, outputDigest)` across a run,
 * warns at 3 identical outcomes, short-circuits broker execution at 5 identical
 * calls (or 4 consecutive identical errors), and halts the turn loop cleanly if
 * the model still repeats a blocked call or exceeds `maxTurnsPerRun`.
 */

export const DEFAULT_MAX_TURNS_PER_RUN = 40

export type LoopBeforeDecision =
  | { action: 'allow' }
  | { action: 'short-circuit'; message: string }
  | { action: 'halt-run'; message: string }

function sortCanonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortCanonical)
  if (value && typeof value === 'object') {
    const sorted: Record<string, unknown> = {}
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      const entry = (value as Record<string, unknown>)[key]
      if (entry !== undefined) sorted[key] = sortCanonical(entry)
    }
    return sorted
  }
  return value
}

export function canonicalizeArgs(args: Record<string, unknown>): string {
  return JSON.stringify(sortCanonical(args))
}

/** Deterministic FNV-1a digest for comparing tool outputs without retaining megabytes. */
export function digestOutput(output: string): string {
  let h1 = 0x811c9dc5
  let h2 = 0x01000193
  for (let i = 0; i < output.length; i += 1) {
    const code = output.charCodeAt(i)
    h1 ^= code
    h1 = Math.imul(h1, 0x01000193)
    h2 ^= code ^ (i & 0xff)
    h2 = Math.imul(h2, 0x85ebca6b)
  }
  return `${output.length}:${(h1 >>> 0).toString(16)}:${(h2 >>> 0).toString(16)}`
}

export class LoopGuard {
  readonly maxTurnsPerRun: number
  private turns = 0
  private readonly outcomeCounts = new Map<string, number>()
  private readonly latestOutcomeCountByCallKey = new Map<string, number>()
  private consecutiveErrorStreak: {
    callKey: string
    outcomeKey: string
    count: number
  } | null = null
  private readonly shortCircuitedKeys = new Set<string>()

  constructor(options?: { maxTurnsPerRun?: number }) {
    this.maxTurnsPerRun = options?.maxTurnsPerRun ?? DEFAULT_MAX_TURNS_PER_RUN
  }

  get turnCount(): number {
    return this.turns
  }

  startTurn(): { allowed: boolean; turn: number; message?: string } {
    this.turns += 1
    if (this.turns > this.maxTurnsPerRun) {
      return {
        allowed: false,
        turn: this.turns,
        message: `[harness notice: Stopped after ${this.maxTurnsPerRun} turns in a single run to prevent an unbounded loop.]`
      }
    }
    return { allowed: true, turn: this.turns }
  }

  beforeCall(tool: string, args: Record<string, unknown>): LoopBeforeDecision {
    const callKey = `${tool}:${canonicalizeArgs(args)}`

    if (this.shortCircuitedKeys.has(callKey)) {
      return {
        action: 'halt-run',
        message: `[harness notice: Stopping turn loop because \`${tool}\` was repeated again with identical arguments after being blocked.]`
      }
    }

    if (
      this.consecutiveErrorStreak?.callKey === callKey &&
      this.consecutiveErrorStreak.count >= 3
    ) {
      this.shortCircuitedKeys.add(callKey)
      return {
        action: 'short-circuit',
        message: `[harness notice: Blocked 4th consecutive identical failing call to \`${tool}\`. Change strategy, fix the arguments, or explain the blocker to the user instead of repeating this call.]`
      }
    }

    const priorIdentical = this.latestOutcomeCountByCallKey.get(callKey) ?? 0
    if (priorIdentical >= 4) {
      this.shortCircuitedKeys.add(callKey)
      return {
        action: 'short-circuit',
        message: `[harness notice: Blocked 5th identical call to \`${tool}\` with unchanged arguments and output. Change strategy or explain the blocker to the user instead of repeating this call.]`
      }
    }

    return { action: 'allow' }
  }

  afterCall(
    tool: string,
    args: Record<string, unknown>,
    isError: boolean,
    rawOutput: string
  ): string {
    const callKey = `${tool}:${canonicalizeArgs(args)}`
    const digest = digestOutput(rawOutput)
    const outcomeKey = `${callKey}::${isError ? 'err' : 'ok'}::${digest}`
    const count = (this.outcomeCounts.get(outcomeKey) ?? 0) + 1
    this.outcomeCounts.set(outcomeKey, count)
    this.latestOutcomeCountByCallKey.set(callKey, count)

    if (isError) {
      if (this.consecutiveErrorStreak?.outcomeKey === outcomeKey) {
        this.consecutiveErrorStreak.count += 1
      } else {
        this.consecutiveErrorStreak = { callKey, outcomeKey, count: 1 }
      }
    } else {
      this.consecutiveErrorStreak = null
    }

    if (count >= 3) {
      return `${rawOutput}\n\n[harness notice: \`${tool}\` has returned this identical ${
        isError ? 'error' : 'result'
      } ${count} times for the same arguments. Do not repeat this exact call.]`
    }

    return rawOutput
  }
}
