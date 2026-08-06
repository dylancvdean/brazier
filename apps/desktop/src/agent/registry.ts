/**
 * Runtime registry.
 *
 * Agent modes are selected the way model providers are: by id. `simple` exposes
 * the standard broker-sandboxed tool set; `powerful` adds the operator-enabled
 * power-tool surface. Both run the Pi adapter today — powerful grows its own
 * adapter as its tools land. `pi` is a legacy alias so sessions created before
 * modes existed still open.
 */

import type { BrokerClient } from './core/brokerClient'
import type { AgentRuntime, AgentRuntimeDescriptor } from './core/types'
import { PiAgentRuntime } from './pi/piRuntime'

export type AgentRuntimeFactory = (broker: BrokerClient) => AgentRuntime

const FACTORIES = new Map<string, AgentRuntimeFactory>([
  ['simple', (broker) => new PiAgentRuntime(broker)],
  ['powerful', (broker) => new PiAgentRuntime(broker)],
  ['pi', (broker) => new PiAgentRuntime(broker)]
])

export const DEFAULT_RUNTIME_ID = 'simple'

export function registerRuntime(id: string, factory: AgentRuntimeFactory): void {
  FACTORIES.set(id, factory)
}

export function createRuntime(id: string, broker: BrokerClient): AgentRuntime {
  const factory = FACTORIES.get(id)
  if (!factory) {
    throw new Error(`Unknown agent runtime \`${id}\`. Available: ${[...FACTORIES.keys()].join(', ')}`)
  }
  return factory(broker)
}

let cachedDescriptors: AgentRuntimeDescriptor[] | undefined

export function availableRuntimes(broker: BrokerClient): AgentRuntimeDescriptor[] {
  return [...FACTORIES.values()].map((factory) => {
    const runtime = factory(broker)
    try {
      return runtime.descriptor
    } finally {
      void runtime.dispose().catch(() => undefined)
    }
  })
}
