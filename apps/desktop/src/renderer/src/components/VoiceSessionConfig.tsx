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

import { AlertTriangle, Check, FolderOpen, ShieldAlert, ShieldCheck, Wrench } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import {
  activateRuntime,
  listRuntimes,
  saveRuntimeSettings,
  type BundledTool,
  type LocalModel,
  type RuntimeEntry,
  type RuntimeSettings
} from '../api'
import {
  createAgentSession,
  fetchAgentCapabilities,
  fetchAgentSession,
  sandboxBadge,
  updateAgentSession,
  type AgentSandboxCapabilities,
  type AgentSessionSummary
} from '../agentApi'
import type { AgentPermissionMode } from '../../../agent/core/types'
import type { AsrPreference, VoiceSessionTarget } from '../session/config'
import { modelDisplayName } from '../model-utils'

type Props = {
  target: VoiceSessionTarget
  models: LocalModel[]
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

/** The pending agent setup a session will be created from, when none is bound. */
export type PendingAgentSetup = {
  workspacePath: string | null
  permissionMode: AgentPermissionMode
}

export function VoiceSessionConfig(props: Props): React.JSX.Element {
  const { onError } = props
  const [runtimes, setRuntimes] = useState<RuntimeEntry[]>([])
  const [sandbox, setSandbox] = useState<AgentSandboxCapabilities | null>(null)
  const [agentSession, setAgentSession] = useState<AgentSessionSummary | null>(null)
  const [pending, setPending] = useState<PendingAgentSetup>({
    workspacePath: null,
    permissionMode: 'ask'
  })
  const [busy, setBusy] = useState(false)

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

  useEffect(() => {
    if (props.target !== 'agent') return
    void fetchAgentCapabilities()
      .then((capabilities) => setSandbox(capabilities.sandbox))
      .catch(() => setSandbox(null))
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
    const selected = await window.brazier.selectWorkspace()
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
      permission_mode: pending.permissionMode
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
            {props.tools.length === 0 ? (
              <p className="voice-notice">No tools available.</p>
            ) : (
              <div className="voice-tool-list">
                {props.tools.map((tool) => {
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
                      {tool.title}
                    </label>
                  )
                })}
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

          <button type="button" className="voice-workspace" onClick={() => void guard(chooseWorkspace)}>
            <FolderOpen size={15} />
            <span>
              <strong>{shortPath(workspace)}</strong>
              <small>Workspace — the agent reads, edits, and runs commands here</small>
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

          {sandbox ? (
            <p className={`voice-notice ${sandbox.isolated ? '' : 'warn'}`}>
              {sandbox.isolated ? <ShieldCheck size={13} /> : <ShieldAlert size={13} />}
              {sandboxBadge(sandbox).label} — {sandbox.detail}
            </p>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}
