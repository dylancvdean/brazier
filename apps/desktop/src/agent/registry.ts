/**
 * Runtime registry.
 *
 * Agent runtimes are selected the way model providers are: by id. V1 ships one,
 * Pi. Adding another means adding a factory here and an adapter beside
 * `pi/piRuntime.ts` — no change to the UI, tools, policy, or persistence.
 */

import type { BrokerClient } from './core/brokerClient'
import type { AgentRuntime, AgentRuntimeDescriptor } from './core/types'
import { PiAgentRuntime } from './pi/piRuntime'

export type AgentRuntimeFactory = (broker: BrokerClient) => AgentRuntime

const FACTORIES = new Map<string, AgentRuntimeFactory>([
  ['pi', (broker) => new PiAgentRuntime(broker)]
])

export const DEFAULT_RUNTIME_ID = 'pi'

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

export function availableRuntimes(broker: BrokerClient): AgentRuntimeDescriptor[] {
  return [...FACTORIES.values()].map((factory) => factory(broker).descriptor)
}
