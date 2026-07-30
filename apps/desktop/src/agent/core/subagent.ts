/**
 * Helpers for sandboxed `spawn_subagent` children.
 *
 * Depth is capped at 1 (children cannot spawn). Concurrency comes from the
 * parent model's TextProfile (`max_subagents`, default 2).
 */

export const SPAWN_SUBAGENT_TOOL = 'spawn_subagent'

/** Default when the parent TextProfile leaves `max_subagents` unset. */
export const DEFAULT_MAX_SUBAGENTS = 2

export type SubagentProfileDefaults = {
  subagent_model?: string | null
  context_size?: number | null
  max_subagents?: number | null
}

export type SubagentRuntimeMetadata = {
  kind: 'subagent'
  parent_session_id: string
  worktree?: {
    source_path: string
    path: string
    branch: string
  }
}

export function resolveMaxSubagents(profile: SubagentProfileDefaults | null | undefined): number {
  const raw = profile?.max_subagents
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    return Math.min(8, Math.max(1, Math.trunc(raw)))
  }
  return DEFAULT_MAX_SUBAGENTS
}

/** Collect task prompts from a spawn_subagent argument object. */
export function collectSpawnPrompts(args: Record<string, unknown>): string[] {
  if (Array.isArray(args.prompts)) {
    const many = args.prompts
      .filter((entry): entry is string => typeof entry === 'string')
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0)
    if (many.length > 0) return many
  }
  if (typeof args.prompt === 'string') {
    const one = args.prompt.trim()
    if (one.length > 0) return [one]
  }
  return []
}

/**
 * Tool arg → profile `subagent_model` → parent model.
 */
export function resolveSubagentModel(
  toolModel: unknown,
  profile: SubagentProfileDefaults | null | undefined,
  parentModel: string
): string {
  if (typeof toolModel === 'string' && toolModel.trim().length > 0) {
    return toolModel.trim()
  }
  const fromProfile = profile?.subagent_model
  if (typeof fromProfile === 'string' && fromProfile.trim().length > 0) {
    return fromProfile.trim()
  }
  return parentModel
}

/** Children inherit the parent's tools except further spawning. */
export function childEnabledTools(parentTools: string[]): string[] {
  return parentTools.filter((name) => name !== SPAWN_SUBAGENT_TOOL)
}

export function isSubagentSession(
  metadata: Record<string, unknown> | null | undefined
): boolean {
  return metadata?.kind === 'subagent'
}

export function parentSessionIdFromMetadata(
  metadata: Record<string, unknown> | null | undefined
): string | null {
  const value = metadata?.parent_session_id
  return typeof value === 'string' && value.length > 0 ? value : null
}

export function buildSubagentMetadata(
  parentSessionId: string,
  parentMetadata: Record<string, unknown> | null | undefined
): SubagentRuntimeMetadata {
  const worktree = parentMetadata?.worktree
  const meta: SubagentRuntimeMetadata = {
    kind: 'subagent',
    parent_session_id: parentSessionId
  }
  if (
    worktree &&
    typeof worktree === 'object' &&
    worktree !== null &&
    typeof (worktree as { source_path?: unknown }).source_path === 'string' &&
    typeof (worktree as { path?: unknown }).path === 'string' &&
    typeof (worktree as { branch?: unknown }).branch === 'string'
  ) {
    meta.worktree = {
      source_path: (worktree as { source_path: string }).source_path,
      path: (worktree as { path: string }).path,
      branch: (worktree as { branch: string }).branch
    }
  }
  return meta
}

/** Final assistant text from a child transcript, for the parent's tool result. */
export function summarizeSubagentResult(
  messages: Array<{ role: string; text?: string; isError?: boolean; output?: string }>,
  options?: { failed?: boolean; error?: string }
): string {
  if (options?.failed) {
    return `Subagent failed: ${options.error ?? 'unknown error'}`
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.role === 'assistant' && typeof message.text === 'string') {
      const text = message.text.trim()
      if (text && text !== 'null' && text !== 'undefined') return text
    }
  }
  return 'Subagent finished without a text reply.'
}
