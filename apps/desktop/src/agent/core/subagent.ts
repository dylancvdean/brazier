/**
 * Helpers for sandboxed `spawn_subagent` children.
 *
 * Depth is capped at 1 (children cannot spawn). Concurrency comes from the
 * parent model's TextProfile (`max_subagents`, default 2) and the backend's
 * slot model (`independent_backend`, `parallel_llama_slots`, or
 * `serialized_shared_slot`).
 */

export const SPAWN_SUBAGENT_TOOL = 'spawn_subagent'

/** Default when the parent TextProfile leaves `max_subagents` unset. */
export const DEFAULT_MAX_SUBAGENTS = 2

export type SubagentProfileDefaults = {
  subagent_model?: string | null
  context_size?: number | null
  max_subagents?: number | null
  parallel_subagents?: boolean | null
}

export type SubagentExecutionMode =
  | {
      kind: 'independent_backend'
      concurrency: number
      slot: 0
    }
  | {
      kind: 'parallel_llama_slots'
      parallelSlots: number
      childSlotCount: number
      concurrency: number
    }
  | {
      kind: 'serialized_shared_slot'
      concurrency: 1
      slot: 0
    }

/** llama-server `--parallel` slot count implied by a text profile. */
export function llamaParallelSlots(
  profile: SubagentProfileDefaults | null | undefined
): number {
  if (!profile?.parallel_subagents) return 1
  return 1 + resolveMaxSubagents(profile)
}

/**
 * Determine how child subagent sessions share or allocate inference slots:
 * - `independent_backend`: `remote:` / `vllm:` or distinct child model -> up to `maxSubagents` concurrently on slot 0.
 * - `parallel_llama_slots`: same-model `gguf:`/`gguf-ext:` with `parallelSlots > 1` -> batched over child slots `1..parallelSlots-1`.
 * - `serialized_shared_slot`: single-slot local model (`gguf:` or `mlx:`) -> sequential one-at-a-time on idle parent slot 0.
 */
export function resolveSubagentExecutionMode(options: {
  parentModelId: string
  targetModelId: string
  parallelSlots: number
  maxSubagents?: number
}): SubagentExecutionMode {
  const parent = options.parentModelId.trim()
  const target = options.targetModelId.trim()
  const targetLower = target.toLowerCase()
  const maxSubagents = Math.max(1, Math.trunc(options.maxSubagents ?? DEFAULT_MAX_SUBAGENTS))

  if (
    targetLower.startsWith('remote:') ||
    targetLower.startsWith('vllm:') ||
    target !== parent
  ) {
    return {
      kind: 'independent_backend',
      concurrency: maxSubagents,
      slot: 0
    }
  }

  const isLlama = targetLower.startsWith('gguf:') || targetLower.startsWith('gguf-ext:')
  if (isLlama && options.parallelSlots > 1) {
    const childSlotCount = options.parallelSlots - 1
    return {
      kind: 'parallel_llama_slots',
      parallelSlots: options.parallelSlots,
      childSlotCount,
      concurrency: Math.max(1, Math.min(maxSubagents, childSlotCount))
    }
  }

  return {
    kind: 'serialized_shared_slot',
    concurrency: 1,
    slot: 0
  }
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
