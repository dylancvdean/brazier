/**
 * Runtime registry.
 *
 * Agent modes are selected the way model providers are: by id. `simple` exposes
 * the standard broker-sandboxed tool set; `powerful` adds the operator-enabled
 * power-tool surface. Both run the Pi adapter today — powerful grows its own
 * adapter as its tools land.
 */

import type { BrokerClient } from './core/brokerClient'
import type { AgentRuntime, AgentRuntimeDescriptor } from './core/types'
import { PiAgentRuntime } from './pi/piRuntime'

export type AgentRuntimeFactory = (broker: BrokerClient, id: string) => AgentRuntime

const FACTORIES = new Map<string, AgentRuntimeFactory>([
  ['simple', (broker, id) => new PiAgentRuntime(broker, id)],
  ['powerful', (broker, id) => new PiAgentRuntime(broker, id)]
])

export const DEFAULT_RUNTIME_ID = 'simple'

/** Retired aliases from before Simple/Powerful modes existed. */
const ALIASES: Record<string, string> = {
  pi: 'simple',
  balanced: 'simple'
}

/**
 * Map a stored session runtime id onto a live adapter.
 *
 * Unknown or deleted modes become Simple so restoring a task cannot fail
 * every tool call because a catalog entry went away.
 */
export function normalizeRuntimeId(id: string | null | undefined): string {
  const trimmed = (id ?? '').trim()
  if (!trimmed) return DEFAULT_RUNTIME_ID
  const aliased = ALIASES[trimmed] ?? trimmed
  if (FACTORIES.has(aliased)) return aliased
  return DEFAULT_RUNTIME_ID
}

export function registerRuntime(id: string, factory: AgentRuntimeFactory): void {
  FACTORIES.set(id, factory)
}

export function createRuntime(id: string, broker: BrokerClient): AgentRuntime {
  const normalized = normalizeRuntimeId(id)
  const factory = FACTORIES.get(normalized)
  if (!factory) {
    return new PiAgentRuntime(broker, DEFAULT_RUNTIME_ID)
  }
  return factory(broker, normalized)
}

export function availableRuntimes(broker: BrokerClient): AgentRuntimeDescriptor[] {
  return [...FACTORIES.entries()].map(([id, factory]) => {
    const runtime = factory(broker, id)
    try {
      return runtime.descriptor
    } finally {
      void runtime.dispose().catch(() => undefined)
    }
  })
}
