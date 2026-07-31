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
        // Explicit open always rehydrates from the daemon so a mode switch or
        // sidebar reselection prefills model context with stored history.
        const session = await this.openSession(command.sessionId, { rehydrate: true })
        this.post({ type: 'session-state', sessionId: session.id, state: session.getState() })
        return session.getState()
      }
      case 'run': {
        const session = await this.openSession(command.sessionId, { rehydrate: false })
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
        const session = await this.openSession(command.sessionId, { rehydrate: false })
        const state = await session.compact()
        this.post({ type: 'session-state', sessionId: session.id, state: session.getState() })
        return state
      }
      case 'set-model': {
        const session = await this.openSession(command.sessionId, { rehydrate: false })
        await session.setModel(command.model)
        return session.getState()
      }
      case 'set-tools': {
        const session = await this.openSession(command.sessionId, { rehydrate: false })
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
   *
   * When `rehydrate` is true (explicit open/resume), refresh transcript + prompt
   * from the daemon even if the session is cached — otherwise switching modes
   * can leave the UI showing history while the model has an empty or stale
   * context. Run/compact paths keep the live in-memory transcript so a prompt
   * that has not finished persisting is not wiped.
   */
  private async openSession(
    sessionId: string,
    options: { rehydrate: boolean }
  ): Promise<AgentSession> {
    let existing = this.sessions.get(sessionId)
    if (existing?.isDisposed()) {
      this.sessions.delete(sessionId)
      existing = undefined
    }
    if (existing && !options.rehydrate) return existing
    // Rehydrating mid-run replaces Pi's live transcript and can crash the worker.
    if (existing && options.rehydrate && this.running.has(sessionId)) {
      return existing
    }

    const broker = this.requireBroker()
    if (!this.runtime) throw new Error('The agent worker has no runtime.')
    const remote = await broker.session(sessionId)
    const prompt = await broker.systemPrompt(sessionId)
    // MCP configuration can change while the utility process remains alive.
    // Refresh when opening a session so newly enabled server tools do not
    // require restarting the desktop application.
    this.tools = await broker.tools()
    const enabled = remote.session.enabled_tools ?? undefined
    const tools = enabled
      ? this.tools.filter((tool) => enabled.includes(tool.name))
      : this.tools
    const [profile, defaults] = await Promise.all([
      broker.textProfile(remote.session.model),
      broker.runtimeInferenceSettings()
    ])
    const messages = remote.messages.map((record) => record.payload)
    if (existing) {
      existing.rehydrate(messages, prompt.system_prompt)
      await existing.refreshInferencePrefs()
      return existing
    }
    const session = await this.runtime.createSession({
      sessionId,
      model: {
        id: remote.session.model,
        name: remote.session.model,
        contextWindow: profile?.context_size ?? defaults.context_size,
        maxTokens: profile?.max_tokens ?? defaults.max_tokens ?? undefined
      },
      systemPrompt: prompt.system_prompt,
      tools,
      messages,
      capabilities: inferModelCapabilities(remote.session.model)
    })
    await session.refreshInferencePrefs()
    this.sessions.set(sessionId, session)
    return session
  }
}
