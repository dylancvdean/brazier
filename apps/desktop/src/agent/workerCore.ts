/**
 * Command handling for the agent worker, independent of Electron so it can be
 * exercised directly in tests.
 *
 * The worker holds sessions and forwards their events. Every tool call goes to
 * the daemon, which decides; the worker selects the runtime adapter by
 * `session.runtime_id`.
 */

import { BrokerClient } from './core/brokerClient'
import { inferModelCapabilities } from './core/modelCompat'
import type { WorkerCommand, WorkerMessage } from './core/protocol'
import type {
  AgentEvent,
  AgentRuntime,
  AgentSession,
  AgentSessionState,
  AgentToolDefinition
} from './core/types'
import { createRuntime, DEFAULT_RUNTIME_ID, normalizeRuntimeId } from './registry'

export type PostMessage = (message: WorkerMessage) => void

/** Queue key for commands that are not about one session. */
const GLOBAL_QUEUE = '*'

export class AgentWorkerCore {
  private readonly post: PostMessage
  private readonly runtimeFactory: (broker: BrokerClient, id: string) => AgentRuntime
  private broker?: BrokerClient
  private readonly runtimes = new Map<string, AgentRuntime>()
  private tools?: AgentToolDefinition[]
  private readonly sessions = new Map<string, AgentSession>()
  /** Runs in flight, so a cancel can reach the right session. */
  private readonly running = new Set<string>()
  /** Runs accepted but still opening their session. */
  private readonly starting = new Set<string>()
  /** Cancels that arrived while their run was still starting. */
  private readonly cancelledBeforeStart = new Set<string>()
  /** Tail of each session's command queue. */
  private readonly queues = new Map<string, Promise<unknown>>()

  constructor(
    post: PostMessage,
    runtimeFactory: (broker: BrokerClient, id: string) => AgentRuntime = (broker, id) =>
      createRuntime(id, broker)
  ) {
    this.post = post
    this.runtimeFactory = runtimeFactory
  }

  /**
   * Schedule one command and post its result.
   *
   * Commands for the same session run one at a time, so an open/rehydrate
   * cannot race a run starting. A run only holds its session's queue while it
   * starts, not while the model works: otherwise Stop would wait behind the
   * very run it is meant to stop, and one long run would block every other
   * session. Cancel and shutdown never queue.
   */
  async handle(command: WorkerCommand): Promise<void> {
    if (command.type === 'cancel' || command.type === 'shutdown') {
      await this.respond(command.requestId, () => this.dispatch(command))
      return
    }
    const key = 'sessionId' in command ? command.sessionId : GLOBAL_QUEUE
    if (command.type === 'run') {
      this.starting.add(command.sessionId)
      const started = this.enqueue(key, () => this.startRun(command.sessionId, command.input))
      await this.respond(command.requestId, async () => {
        const { session, events } = await started
        return this.pumpRun(session, events)
      })
      return
    }
    await this.respond(command.requestId, () => this.enqueue(key, () => this.dispatch(command)))
  }

  private async respond(requestId: string, task: () => Promise<unknown>): Promise<void> {
    try {
      const data = await task()
      this.post({ type: 'result', requestId, ok: true, data })
    } catch (cause) {
      const error = cause instanceof Error ? cause.message : String(cause)
      this.post({ type: 'result', requestId, ok: false, error })
    }
  }

  private enqueue<T>(key: string, task: () => Promise<T>): Promise<T> {
    const previous = this.queues.get(key) ?? Promise.resolve()
    const next = previous.then(task, task)
    const tail = next.then(
      () => undefined,
      () => undefined
    )
    this.queues.set(key, tail)
    void tail.then(() => {
      if (this.queues.get(key) === tail) this.queues.delete(key)
    })
    return next
  }

  private async startRun(
    sessionId: string,
    input: Extract<WorkerCommand, { type: 'run' }>['input']
  ): Promise<{ session: AgentSession; events: AsyncIterable<AgentEvent> | null }> {
    try {
      const session = await this.openSession(sessionId, { rehydrate: false })
      if (this.running.has(session.id)) {
        throw new Error('That session is already running. Cancel it first.')
      }
      if (this.cancelledBeforeStart.delete(sessionId)) {
        return { session, events: null }
      }
      const events = session.run(input)
      this.running.add(session.id)
      return { session, events }
    } finally {
      this.starting.delete(sessionId)
      this.cancelledBeforeStart.delete(sessionId)
    }
  }

  private async pumpRun(
    session: AgentSession,
    events: AsyncIterable<AgentEvent> | null
  ): Promise<AgentSessionState> {
    if (!events) {
      // Report the stop so the supervisor and UI see this run end.
      this.post({
        type: 'event',
        sessionId: session.id,
        event: {
          type: 'run-cancelled',
          sessionId: session.id,
          runId: `run-cancelled-before-start-${Date.now().toString(36)}`,
          timestamp: new Date().toISOString(),
          sequence: 0
        }
      })
    } else {
      try {
        for await (const event of events) {
          this.post({ type: 'event', sessionId: session.id, event })
        }
      } finally {
        this.running.delete(session.id)
      }
    }
    this.post({ type: 'session-state', sessionId: session.id, state: session.getState() })
    return session.getState()
  }

  private async dispatch(command: WorkerCommand): Promise<unknown> {
    switch (command.type) {
      case 'init': {
        this.broker = new BrokerClient({
          address: command.connection.address,
          apiKey: command.connection.apiKey
        })
        // Eagerly warm the default runtime so ready reports something useful;
        // sessions may still open a different adapter via runtime_id.
        const defaultRuntime = this.runtimeFor(DEFAULT_RUNTIME_ID)
        this.tools = await this.broker.tools()
        this.post({
          type: 'ready',
          runtimeId: defaultRuntime.descriptor.id,
          runtimeVersion: defaultRuntime.descriptor.version
        })
        return { runtime: defaultRuntime.descriptor, tools: this.tools }
      }
      case 'open-session': {
        // Explicit open always rehydrates from the daemon so a mode switch or
        // sidebar reselection prefills model context with stored history.
        const session = await this.openSession(command.sessionId, { rehydrate: true })
        this.post({ type: 'session-state', sessionId: session.id, state: session.getState() })
        return session.getState()
      }
      case 'run': {
        const { session, events } = await this.startRun(command.sessionId, command.input)
        return this.pumpRun(session, events)
      }
      case 'cancel': {
        if (this.starting.has(command.sessionId)) {
          this.cancelledBeforeStart.add(command.sessionId)
        }
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
      case 'set-permission-mode': {
        const session = await this.openSession(command.sessionId, { rehydrate: false })
        await session.setPermissionMode(command.mode)
        this.post({ type: 'session-state', sessionId: session.id, state: session.getState() })
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
        for (const runtime of this.runtimes.values()) {
          await runtime.dispose()
        }
        this.runtimes.clear()
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

  private runtimeFor(runtimeId: string): AgentRuntime {
    const broker = this.requireBroker()
    const id = normalizeRuntimeId(runtimeId)
    let runtime = this.runtimes.get(id)
    if (!runtime) {
      runtime = this.runtimeFactory(broker, id)
      this.runtimes.set(id, runtime)
    }
    return runtime
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
    // Rehydrating mid-run replaces the live transcript and can crash the worker.
    if (existing && options.rehydrate && this.running.has(sessionId)) {
      return existing
    }

    const broker = this.requireBroker()
    const remote = await broker.session(sessionId)
    const runtime = this.runtimeFor(remote.session.runtime_id)
    const prompt = await broker.systemPrompt(sessionId)
    // MCP configuration can change while the utility process remains alive.
    // Refresh when opening a session so newly enabled server tools do not
    // require restarting the desktop application.
    this.tools = await broker.tools()
    const enabled = remote.session.enabled_tools ?? undefined
    const tools = enabled
      ? this.tools.filter((tool) => enabled.includes(tool.name))
      : this.tools.filter(
          (tool) => tool.powerTool !== true || remote.session.runtime_id === 'powerful'
        )
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
    const session = await runtime.createSession({
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
      capabilities: inferModelCapabilities(remote.session.model),
      preloaded: {
        session: {
          id: remote.session.id,
          title: remote.session.title,
          workspace_path: remote.session.workspace_path,
          permission_mode: remote.session.permission_mode,
          permission_settings: remote.session.permission_settings,
          enabled_tools: remote.session.enabled_tools,
          created_at: remote.session.created_at,
          updated_at: remote.session.updated_at,
          runtime_id: remote.session.runtime_id,
          runtime_metadata: remote.session.runtime_metadata as Record<string, unknown> | null
        },
        tool_executions: remote.tool_executions,
        sandbox: {
          backend: remote.sandbox.backend,
          profile: 'workspace',
          isolated: remote.sandbox.isolated,
          network: false,
          detail: remote.sandbox.detail
        }
      },
      preloadedInference: defaults
    })
    await session.refreshInferencePrefs()
    this.sessions.set(sessionId, session)
    return session
  }
}
