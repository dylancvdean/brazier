/**
 * Everything a voice session needs decided before it starts.
 *
 * Voice mode owns its whole screen, so the pieces it depends on cannot be
 * reached from anywhere else while it is open: the runtime that speaks, the
 * transcription that hears, and — depending on where speech is pointed — the
 * chat model and tools, or the agent's workspace and permission mode. What is
 * shown follows the destination, because a setting that does not apply to the
 * session about to start is only noise.
 */

import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronRight,
  FolderOpen,
  LoaderCircle,
  ShieldAlert,
  ShieldCheck,
  Wrench
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import {
  activateRuntime,
  fetchModelSettings,
  listAdapters,
  listRuntimes,
  listTools,
  saveModelProfile,
  saveRuntimeSettings,
  type Adapter,
  type BundledTool,
  type LocalModel,
  type ModelProfile,
  type RuntimeEntry,
  type RuntimeSettings
} from '../api'
import { modelEngine, modelKindFor } from '../model-utils'
import { ModelSettingsFields, emptyProfile } from './ModelSettingsFields'
import { daemonPathLabel, useConnectionProfile } from '../connectionProfile'
import {
  createAgentSession,
  fetchAgentCapabilities,
  fetchAgentSession,
  fetchAgentTools,
  sandboxBadge,
  updateAgentSession,
  type AgentSandboxCapabilities,
  type AgentSessionSummary,
  type AgentToolCatalogEntry
} from '../agentApi'
import type { AgentPermissionMode } from '../../../agent/core/types'
import { resolveAsrEngine, type AsrPreference, type VoiceSessionTarget } from '../session/config'
import { modelDisplayName } from '../model-utils'

type Props = {
  target: VoiceSessionTarget
  models: LocalModel[]
  /** PersonaPlex model chosen in the top bar. */
  voiceModelId: string
  /** Chat model, shared with the rest of the app. */
  chatModelId: string
  onChatModelChange: (modelId: string) => void
  tools: BundledTool[]
  enabledTools: string[]
  onEnabledToolsChange: (names: string[]) => void
  settings: RuntimeSettings | null
  onSettingsSaved: (settings: RuntimeSettings) => void
  /** Which ASR interfaces the daemon reports as usable. */
  asrAvailable: { batch: boolean; streaming: boolean }
  asrPreference: AsrPreference
  onAsrPreferenceChange: (preference: AsrPreference) => void
  /** Re-read host capabilities after activating a runtime. */
  onRuntimeActivated?: () => void
  /** Agent session bound to this conversation, when there is one. */
  agentSessionId: string | null
  onAgentSessionBound: (agentSessionId: string) => void
  onError: (message: string | null) => void
}

const PERMISSION_LABELS: Record<AgentPermissionMode, { title: string; detail: string }> = {
  ask: {
    title: 'Ask first',
    detail: 'Approve writes, commands, network use, and anything outside the workspace.'
  },
  'sandbox-only': {
    title: 'Sandbox only',
    detail: 'Sandboxed work runs without prompts. Host access is refused outright.'
  },
  'skip-permissions': {
    title: 'Skip permissions',
    detail: 'No prompts for sandboxed work. Host actions still need the separate opt-in.'
  }
}

const ASR_LABELS: Record<AsrPreference, string> = {
  auto: 'Automatic',
  'whisper.cpp': 'Whisper',
  'streaming-asr': 'Nemotron streaming'
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

function shortPath(path: string | null | undefined): string {
  if (!path) return 'No workspace chosen'
  const parts = path.split('/').filter(Boolean)
  return parts.length <= 2 ? path : `…/${parts.slice(-2).join('/')}`
}

type ChatToolGroup = { key: string; label: string; tools: BundledTool[] }

function groupChatTools(tools: BundledTool[]): ChatToolGroup[] {
  const bundled: BundledTool[] = []
  const mcp = new Map<string, ChatToolGroup>()
  for (const tool of tools) {
    if (tool.source !== 'mcp' && !tool.name.startsWith('mcp/')) {
      bundled.push(tool)
      continue
    }
    const serverId = tool.name.split('/')[1] ?? 'mcp'
    const key = `mcp:${serverId}`
    const group = mcp.get(key) ?? {
      key,
      label: tool.server_name ?? serverId,
      tools: []
    }
    group.tools.push(tool)
    mcp.set(key, group)
  }
  return [
    ...(bundled.length > 0 ? [{ key: 'bundled', label: 'Built-in', tools: bundled }] : []),
    ...mcp.values()
  ]
}

function chatToolLabel(tool: BundledTool): string {
  return tool.source === 'mcp' || tool.name.startsWith('mcp/')
    ? tool.name.replace(/^mcp\//, '')
    : tool.title
}

/** The pending agent setup a session will be created from, when none is bound. */
export type PendingAgentSetup = {
  workspacePath: string | null
  permissionMode: AgentPermissionMode
}

/**
 * Advanced settings for one of the models a voice session uses.
 *
 * A voice session is not one model but three — the one that speaks, the one
 * that transcribes, and the one that answers — so each is configured on its own
 * rather than through a single menu that would have to guess which was meant.
 */
function AdvancedModelPanel(props: {
  role: string
  model: LocalModel
  settings: RuntimeSettings | null
  profile: ModelProfile | undefined
  adapters: Adapter[]
  onSaved: (models: Record<string, ModelProfile>) => void
  onAdapterAdded: () => void
  onError: (message: string | null) => void
}): React.JSX.Element {
  const kind = modelKindFor(props.model.id)
  const stored = props.profile ?? emptyProfile(kind)
  const [draft, setDraft] = useState<ModelProfile | null>(null)
  const [saving, setSaving] = useState(false)
  const effective = draft ?? stored
  const dirty = draft != null && JSON.stringify(draft) !== JSON.stringify(stored)

  async function save(): Promise<void> {
    if (!draft) return
    setSaving(true)
    props.onError(null)
    try {
      props.onSaved(await saveModelProfile(props.model.id, draft))
      setDraft(null)
    } catch (cause) {
      props.onError(errorText(cause))
    } finally {
      setSaving(false)
    }
  }

  return (
    <details className="voice-advanced-model">
      <summary>
        <span>{props.role}</span>
        <small>{modelDisplayName(props.model.id, props.model).title}</small>
      </summary>
      <ModelSettingsFields
        modelId={props.model.id}
        kind={kind}
        engine={modelEngine(props.model)}
        profile={effective}
        adapters={props.adapters}
        inherited={{
          contextSize: props.settings?.context_size,
          batchSize: props.settings?.batch_size,
          temperature: props.settings?.temperature,
          topP: props.settings?.top_p,
          flashAttention: props.settings?.flash_attention,
          kvCacheTypeK: props.settings?.kv_cache_type_k,
          kvCacheTypeV: props.settings?.kv_cache_type_v,
          maxTokens: props.settings?.max_tokens ?? null
        }}
        onChange={setDraft}
        onAdapterAdded={props.onAdapterAdded}
        onError={props.onError}
      />
      <button
        type="button"
        className="popover-apply"
        disabled={!dirty || saving}
        onClick={() => void save()}
      >
        {saving ? <LoaderCircle className="spin" size={14} /> : null}
        {dirty ? 'Save' : 'Saved'}
      </button>
    </details>
  )
}

export function VoiceSessionConfig(props: Props): React.JSX.Element {
  const connectionProfile = useConnectionProfile()
  const { onError } = props
  const [runtimes, setRuntimes] = useState<RuntimeEntry[]>([])
  const [sandbox, setSandbox] = useState<AgentSandboxCapabilities | null>(null)
  const [agentSession, setAgentSession] = useState<AgentSessionSummary | null>(null)
  const [agentTools, setAgentTools] = useState<AgentToolCatalogEntry[]>([])
  const [pendingAgentTools, setPendingAgentTools] = useState<string[] | null>(null)
  const [chatTools, setChatTools] = useState<BundledTool[]>(props.tools)
  const [pending, setPending] = useState<PendingAgentSetup>({
    workspacePath: null,
    permissionMode: 'ask'
  })
  const [busy, setBusy] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [profiles, setProfiles] = useState<Record<string, ModelProfile>>({})
  const [adapters, setAdapters] = useState<Adapter[]>([])

  const needsTranscripts = props.target !== 'neither'
  const voiceModels = props.models.filter((model) => model.id.startsWith('personaplex:'))
  const chatModels = props.models.filter(
    (model) =>
      !model.id.startsWith('personaplex:') &&
      !model.id.startsWith('whisper:') &&
      !model.id.startsWith('streaming-asr:') &&
      !model.id.startsWith('sdcpp-')
  )
  const whisperModels = props.models.filter((model) => model.id.startsWith('whisper:'))
  const streamingModels = props.models.filter((model) => model.id.startsWith('streaming-asr:'))

  const refreshRuntimes = useCallback(async () => {
    try {
      const response = await listRuntimes()
      setRuntimes(response.data.filter((entry) => entry.engine.startsWith('personaplex')))
    } catch (cause) {
      onError(errorText(cause))
    }
  }, [onError])

  useEffect(() => {
    void refreshRuntimes()
  }, [refreshRuntimes])

  // The app-level catalog is a useful initial value, but MCP tools can be
  // added while this window is open. Refresh on entering the Chat setup so the
  // list describes the tools a spoken turn can actually call.
  useEffect(() => {
    if (props.target !== 'chat') return
    setChatTools(props.tools)
    void listTools().then(setChatTools).catch(() => undefined)
  }, [props.target, props.tools])

  const refreshAdapters = useCallback(() => {
    void listAdapters()
      .then(setAdapters)
      .catch(() => {
        // Non-fatal: the adapter pickers just show an empty library.
      })
  }, [])

  // Loaded when the section is opened rather than on mount: most sessions start
  // without ever needing it.
  useEffect(() => {
    if (!advancedOpen) return
    refreshAdapters()
    void fetchModelSettings()
      .then((response) => setProfiles(response.models))
      .catch((cause: unknown) => onError(errorText(cause)))
  }, [advancedOpen, refreshAdapters, onError])

  useEffect(() => {
    if (props.target !== 'agent') return
    void Promise.all([fetchAgentCapabilities(), fetchAgentTools()])
      .then(([capabilities, tools]) => {
        setSandbox(capabilities.sandbox)
        setAgentTools(tools)
      })
      .catch(() => {
        setSandbox(null)
        setAgentTools([])
      })
  }, [props.target])

  const agentSessionId = props.agentSessionId
  useEffect(() => {
    if (!agentSessionId) {
      setAgentSession(null)
      return
    }
    void fetchAgentSession(agentSessionId)
      .then((detail) => setAgentSession(detail.session))
      .catch(() => setAgentSession(null))
  }, [agentSessionId])

  async function guard(action: () => Promise<void>): Promise<void> {
    setBusy(true)
    onError(null)
    try {
      await action()
    } catch (cause) {
      onError(errorText(cause))
    } finally {
      setBusy(false)
    }
  }

  async function saveSetting(patch: Partial<RuntimeSettings>): Promise<void> {
    if (!props.settings) return
    const saved = await saveRuntimeSettings({ ...props.settings, ...patch })
    props.onSettingsSaved(saved)
  }

  /**
   * Point the agent at a folder. With no task bound yet this creates one, so
   * choosing a folder is the whole setup: `agent` never opens a conversation
   * that refuses its first sentence for want of a session.
   */
  async function chooseWorkspace(): Promise<void> {
    const selected = connectionProfile.kind === 'remote'
      ? window.prompt(
          `Workspace path on ${connectionProfile.name} (${connectionProfile.hostLabel})`,
          workspace ?? ''
        )?.trim() ?? null
      : await window.brazier.selectWorkspace()
    if (!selected) return
    if (agentSession) {
      const updated = await updateAgentSession(agentSession.id, { workspace_path: selected })
      setAgentSession(updated)
      return
    }
    if (!props.chatModelId) {
      throw new Error('Choose a model for the agent before picking a workspace.')
    }
    const created = await createAgentSession({
      title: 'Voice session',
      workspace_path: selected,
      model: props.chatModelId,
      permission_mode: pending.permissionMode,
      ...(pendingAgentTools ? { enabled_tools: pendingAgentTools } : {})
    })
    setAgentSession(created)
    setPending((current) => ({ ...current, workspacePath: selected }))
    props.onAgentSessionBound(created.id)
  }

  async function choosePermissionMode(mode: AgentPermissionMode): Promise<void> {
    if (agentSession) {
      const updated = await updateAgentSession(agentSession.id, { permission_mode: mode })
      setAgentSession(updated)
      return
    }
    setPending((current) => ({ ...current, permissionMode: mode }))
  }

  const availableAgentToolNames = agentTools.map((tool) => tool.name)
  const enabledAgentTools = (
    agentSession?.enabled_tools ?? pendingAgentTools ?? availableAgentToolNames
  ).filter((name) => availableAgentToolNames.includes(name))

  async function setAgentTool(name: string, enabled: boolean): Promise<void> {
    const next = enabled
      ? Array.from(new Set([...enabledAgentTools, name]))
      : enabledAgentTools.filter((entry) => entry !== name)
    if (!agentSession) {
      setPendingAgentTools(next)
      return
    }
    const updated = await updateAgentSession(agentSession.id, { enabled_tools: next })
    setAgentSession(updated)
    // The utility process caches a session's tool definitions. Reopen it now
    // so a selection made here is what the next spoken turn actually receives.
    await window.brazier.agent.closeSession(updated.id)
    await window.brazier.agent.openSession(updated.id)
  }

  // Resolved by the same function the session uses, so what is shown is what
  // will happen rather than a second opinion about it.
  const resolved = resolveAsrEngine(props.asrPreference, props.asrAvailable)

  // The models this session will actually put to work, in the order the session
  // uses them. A model that is not part of this target is left out.
  const transcriptionModelId =
    resolved === 'streaming-asr'
      ? (props.settings?.streaming_asr_model ?? streamingModels[0]?.id)
      : (props.settings?.whisper_model ?? whisperModels[0]?.id)
  const advancedModels = (
    [
      ['Voice', props.voiceModelId],
      ...(needsTranscripts ? [['Transcription', transcriptionModelId] as const] : []),
      ...(props.target === 'chat' ? [['Chat', props.chatModelId] as const] : [])
    ] as Array<readonly [string, string | undefined]>
  ).flatMap(([role, modelId]) => {
    const model = modelId ? props.models.find((entry) => entry.id === modelId) : undefined
    return model ? [{ role, model }] : []
  })
  const workspace = agentSession?.workspace_path ?? pending.workspacePath
  const permissionMode: AgentPermissionMode =
    agentSession?.permission_mode ?? pending.permissionMode

  return (
    <div className="voice-config">
      <section className="voice-config-group">
        <span className="section-label">PersonaPlex runtime</span>
        {runtimes.length === 0 ? (
          <p className="voice-notice">
            None built yet. On Apple Silicon build PersonaPlex MLX under Manage → Runtimes; on Linux
            CUDA build PersonaPlex / Moshi.
          </p>
        ) : (
          <div className="voice-choice-list">
            {runtimes.map((entry) => (
              <button
                key={entry.id}
                type="button"
                role="radio"
                aria-checked={entry.active}
                className={entry.active ? 'active' : ''}
                disabled={busy}
                onClick={() =>
                  void guard(async () => {
                    await activateRuntime(entry.id)
                    await refreshRuntimes()
                    props.onRuntimeActivated?.()
                  })
                }
              >
                {entry.active ? <Check size={13} /> : <span className="voice-choice-dot" />}
                <span>{entry.label}</span>
              </button>
            ))}
          </div>
        )}
        {voiceModels.length === 0 ? (
          <p className="voice-notice">
            No <code>personaplex:</code> model downloaded yet — get one from Manage → Discover.
          </p>
        ) : null}
      </section>

      {needsTranscripts ? (
        <section className="voice-config-group">
          <span className="section-label">Transcription</span>
          <p className="voice-notice">
            PersonaPlex reports only its own speech, so what you say is transcribed separately.
          </p>
          <div className="voice-target-choices" role="radiogroup" aria-label="Transcription engine">
            {(['auto', 'whisper.cpp', 'streaming-asr'] as AsrPreference[]).map((option) => {
              const usable =
                option === 'auto'
                  ? props.asrAvailable.batch || props.asrAvailable.streaming
                  : option === 'whisper.cpp'
                    ? props.asrAvailable.batch
                    : props.asrAvailable.streaming
              return (
                <button
                  key={option}
                  type="button"
                  role="radio"
                  aria-checked={props.asrPreference === option}
                  className={props.asrPreference === option ? 'active' : ''}
                  disabled={!usable}
                  title={usable ? undefined : 'Not installed'}
                  onClick={() => props.onAsrPreferenceChange(option)}
                >
                  {ASR_LABELS[option]}
                </button>
              )
            })}
          </div>
          {props.asrPreference !== 'streaming-asr' && whisperModels.length > 0 ? (
            <label className="voice-field">
              <span>Whisper model</span>
              <select
                value={props.settings?.whisper_model ?? ''}
                onChange={(event) =>
                  void guard(() => saveSetting({ whisper_model: event.target.value || null }))
                }
              >
                <option value="">Automatic</option>
                {whisperModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {modelDisplayName(model.id, model).title}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {props.asrPreference !== 'whisper.cpp' && streamingModels.length > 0 ? (
            <label className="voice-field">
              <span>Streaming ASR model</span>
              <select
                value={props.settings?.streaming_asr_model ?? ''}
                onChange={(event) =>
                  void guard(() => saveSetting({ streaming_asr_model: event.target.value || null }))
                }
              >
                <option value="">Automatic</option>
                {streamingModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {modelDisplayName(model.id, model).title}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <p className="voice-notice">
            {resolved === undefined && !props.asrAvailable.batch
              ? 'Nothing chosen would transcribe: this would go to Whisper, which is not installed.'
              : `A spoken turn will be transcribed by ${
                  resolved === 'streaming-asr' ? 'Nemotron streaming' : 'Whisper'
                }.`}
          </p>
          {!props.asrAvailable.batch && !props.asrAvailable.streaming ? (
            <p className="voice-notice">
              <AlertTriangle size={13} /> Nothing installed to transcribe with. Build WhisperKit
              under Manage → Runtimes, or download a Whisper or Nemotron ASR model from Discover.
            </p>
          ) : null}
        </section>
      ) : null}

      {props.target === 'chat' ? (
        <>
          <section className="voice-config-group">
            <span className="section-label">Chat model</span>
            <label className="voice-field">
              <select
                value={props.chatModelId}
                onChange={(event) => props.onChatModelChange(event.target.value)}
              >
                <option value="">Choose a model…</option>
                {chatModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {modelDisplayName(model.id, model).title}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section className="voice-config-group">
            <span className="section-label">
              <Wrench size={12} /> Tools
            </span>
            {chatTools.length === 0 ? (
              <p className="voice-notice">No tools available.</p>
            ) : (
              <div className="voice-tool-list">
                {groupChatTools(chatTools).map((group) => (
                  <div key={group.key} className="voice-tool-group">
                    <span className="section-label">{group.label}</span>
                    {group.tools.map((tool) => {
                      const on = props.enabledTools.includes(tool.name)
                      return (
                        <label key={tool.name} title={tool.description}>
                          <input
                            type="checkbox"
                            checked={on}
                            onChange={() =>
                              props.onEnabledToolsChange(
                                on
                                  ? props.enabledTools.filter((name) => name !== tool.name)
                                  : [...props.enabledTools, tool.name]
                              )
                            }
                          />
                          {chatToolLabel(tool)}
                        </label>
                      )
                    })}
                  </div>
                ))}
              </div>
            )}
          </section>
        </>
      ) : null}

      {props.target === 'agent' ? (
        <section className="voice-config-group">
          <span className="section-label">Agent task</span>
          {agentSession ? (
            <p className="voice-notice">
              Speaking to “{agentSession.title}”. Spoken turns join this task.
            </p>
          ) : (
            <p className="voice-notice">
              No task is bound to this conversation yet — one is created from these settings when
              the conversation starts.
            </p>
          )}

          <button
            type="button"
            className="voice-workspace"
            title={connectionProfile.kind === 'remote' ? `Enter ${daemonPathLabel(connectionProfile).toLowerCase()}` : 'Choose a workspace folder'}
            onClick={() => void guard(chooseWorkspace)}
          >
            <FolderOpen size={15} />
            <span>
              <strong>{shortPath(workspace)}</strong>
              <small>{daemonPathLabel(connectionProfile)} — the agent reads, edits, and runs commands here</small>
            </span>
          </button>

          <div className="voice-field">
            <span>Permissions</span>
            <div className="voice-choice-list">
              {(Object.keys(PERMISSION_LABELS) as AgentPermissionMode[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  role="radio"
                  aria-checked={permissionMode === mode}
                  className={permissionMode === mode ? 'active' : ''}
                  disabled={busy}
                  title={PERMISSION_LABELS[mode].detail}
                  onClick={() => void guard(() => choosePermissionMode(mode))}
                >
                  {permissionMode === mode ? <Check size={13} /> : <span className="voice-choice-dot" />}
                  <span>{PERMISSION_LABELS[mode].title}</span>
                </button>
              ))}
            </div>
            <p className="voice-notice">{PERMISSION_LABELS[permissionMode].detail}</p>
          </div>

          <div className="voice-field">
            <span className="section-label">
              <Wrench size={12} /> Agent tools
            </span>
            {agentTools.length === 0 ? (
              <p className="voice-notice">No agent tools available.</p>
            ) : (
              <div className="voice-tool-list">
                {agentTools.map((tool) => {
                  const on = enabledAgentTools.includes(tool.name)
                  return (
                    <label key={tool.name} title={tool.description}>
                      <input
                        type="checkbox"
                        checked={on}
                        disabled={busy}
                        onChange={() => void guard(() => setAgentTool(tool.name, !on))}
                      />
                      {tool.label}
                    </label>
                  )
                })}
              </div>
            )}
            <p className="voice-notice">
              Includes user-added MCP tools. Changes apply to the next spoken turn.
            </p>
          </div>

          {sandbox ? (
            <p className={`voice-notice ${sandbox.isolated ? '' : 'warn'}`}>
              {sandbox.isolated ? <ShieldCheck size={13} /> : <ShieldAlert size={13} />}
              {sandboxBadge(sandbox).label} — {sandbox.detail}
            </p>
          ) : null}
        </section>
      ) : null}

      {/* Advanced settings live at the foot of the setup screen rather than in
          the top bar's inference menu: a session runs several models at once,
          and that menu can only speak for one. Once a session starts this
          screen is replaced anyway, so there is room to be thorough here. */}
      <section className="voice-config-group voice-advanced">
        <button
          type="button"
          className="voice-advanced-toggle"
          aria-expanded={advancedOpen}
          onClick={() => setAdvancedOpen((open) => !open)}
        >
          {advancedOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          Show advanced options
        </button>
        {advancedOpen ? (
          <>
            <p className="voice-notice">
              Defaults for each model this session uses. They apply wherever the model is used,
              not only here.
            </p>
            {advancedModels.length === 0 ? (
              <p className="voice-notice">
                Nothing to configure until a model is chosen above.
              </p>
            ) : (
              advancedModels.map(({ role, model }) => (
                <AdvancedModelPanel
                  key={model.id}
                  role={role}
                  model={model}
                  settings={props.settings}
                  profile={profiles[model.id]}
                  adapters={adapters}
                  onSaved={setProfiles}
                  onAdapterAdded={refreshAdapters}
                  onError={onError}
                />
              ))
            )}
          </>
        ) : null}
      </section>
    </div>
  )
}
