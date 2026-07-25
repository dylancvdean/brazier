/**
 * Command handling for the agent worker, independent of Electron so it can be
 * exercised directly in tests.
 *
 * The worker holds sessions, forwards their events, and does nothing else. It
 * owns no policy: every tool call goes to the daemon, which decides.
 */

import { BrokerClient } from './core/brokerClient'
import { inferModelCapabilities } from './core/modelCompat'
import type { WorkerCommand, WorkerMessage } from './core/protocol'
import type { AgentRuntime, AgentSession, AgentToolDefinition } from './core/types'
import { createRuntime, DEFAULT_RUNTIME_ID } from './registry'

export type PostMessage = (message: WorkerMessage) => void

export class AgentWorkerCore {
  private readonly post: PostMessage
  private readonly runtimeFactory: (broker: BrokerClient, id: string) => AgentRuntime
  private broker?: BrokerClient
  private runtime?: AgentRuntime
  private tools?: AgentToolDefinition[]
  private readonly sessions = new Map<string, AgentSession>()
  /** Runs in flight, so a cancel can reach the right session. */
  private readonly running = new Set<string>()

  constructor(
    post: PostMessage,
    runtimeFactory: (broker: BrokerClient, id: string) => AgentRuntime = (broker, id) =>
      createRuntime(id, broker)
  ) {
    this.post = post
    this.runtimeFactory = runtimeFactory
  }

  async handle(command: WorkerCommand): Promise<void> {
    try {
      const data = await this.dispatch(command)
      this.post({ type: 'result', requestId: command.requestId, ok: true, data })
    } catch (cause) {
      const error = cause instanceof Error ? cause.message : String(cause)
      this.post({ type: 'result', requestId: command.requestId, ok: false, error })
    }
  }

  private async dispatch(command: WorkerCommand): Promise<unknown> {
    switch (command.type) {
      case 'init': {
        this.broker = new BrokerClient({
          address: command.connection.address,
          apiKey: command.connection.apiKey
        })
        this.runtime = this.runtimeFactory(this.broker, DEFAULT_RUNTIME_ID)
        this.tools = await this.broker.tools()
        this.post({
          type: 'ready',
          runtimeId: this.runtime.descriptor.id,
          runtimeVersion: this.runtime.descriptor.version
        })
        return { runtime: this.runtime.descriptor, tools: this.tools }
      }
      case 'open-session': {
        const session = await this.openSession(command.sessionId)
        this.post({ type: 'session-state', sessionId: session.id, state: session.getState() })
        return session.getState()
      }
      case 'run': {
        const session = await this.openSession(command.sessionId)
        if (this.running.has(session.id)) {
          throw new Error('That session is already running. Cancel it first.')
        }
        this.running.add(session.id)
        try {
          for await (const event of session.run(command.input)) {
            this.post({ type: 'event', sessionId: session.id, event })
          }
        } finally {
          this.running.delete(session.id)
        }
        this.post({ type: 'session-state', sessionId: session.id, state: session.getState() })
        return session.getState()
      }
      case 'cancel': {
        const session = this.sessions.get(command.sessionId)
        if (!session) {
          // Nothing is loaded, but processes and approvals may still be live.
          await this.requireBroker().cancel(command.sessionId)
          return { cancelled: true }
        }
        await session.cancel()
        return { cancelled: true }
      }
      case 'compact': {
        const session = await this.openSession(command.sessionId)
        const state = await session.compact()
        this.post({ type: 'session-state', sessionId: session.id, state: session.getState() })
        return state
      }
      case 'set-model': {
        const session = await this.openSession(command.sessionId)
        await session.setModel(command.model)
        return session.getState()
      }
      case 'set-tools': {
        const session = await this.openSession(command.sessionId)
        await session.setEnabledTools(command.tools)
        return session.getState()
      }
      case 'close-session': {
        const session = this.sessions.get(command.sessionId)
        if (session) {
          await session.dispose()
          this.sessions.delete(command.sessionId)
        }
        return { closed: true }
      }
      case 'shutdown': {
        for (const session of this.sessions.values()) {
          await session.dispose()
        }
        this.sessions.clear()
        await this.runtime?.dispose()
        return { shutdown: true }
      }
      default: {
        const exhaustive: never = command
        throw new Error(`Unknown command ${JSON.stringify(exhaustive)}`)
      }
    }
  }

  private requireBroker(): BrokerClient {
    if (!this.broker) throw new Error('The agent worker has not been initialized.')
    return this.broker
  }

  /**
   * Load a session, building its runtime state from what the daemon stored. The
   * system prompt and tool catalog come from the daemon so the model always
   * sees the live sandbox and permission state.
   */
  private async openSession(sessionId: string): Promise<AgentSession> {
    const existing = this.sessions.get(sessionId)
    if (existing) return existing
    const broker = this.requireBroker()
    if (!this.runtime) throw new Error('The agent worker has no runtime.')
    const remote = await broker.session(sessionId)
    const prompt = await broker.systemPrompt(sessionId)
    this.tools ??= await broker.tools()
    const enabled = remote.session.enabled_tools ?? undefined
    const tools = enabled
      ? this.tools.filter((tool) => enabled.includes(tool.name))
      : this.tools
    const session = await this.runtime.createSession({
      sessionId,
      model: { id: remote.session.model, name: remote.session.model },
      systemPrompt: prompt.system_prompt,
      tools,
      messages: remote.messages.map((record) => record.payload),
      capabilities: inferModelCapabilities(remote.session.model)
    })
    this.sessions.set(sessionId, session)
    return session
  }
}
