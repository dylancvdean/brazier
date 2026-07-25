/**
 * React binding for the session coordinator.
 *
 * The coordinator itself is plain TypeScript and outlives any component render;
 * this hook wires it to the current conversation and re-renders on snapshots.
 * The agent run lives in the worker process, so unmounting the voice UI stops
 * voice, never the task.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { updateConversation } from '../api'
import type { Message } from '../types'
import { WorkerAgentAdapter } from './agentAdapter'
import { DaemonChatAdapter, toConversationMessage } from './chatAdapter'
import { readIntegrationConfig, writeIntegrationConfig, type IntegrationConfig } from './config'
import { SessionCoordinator, type CoordinatorSnapshot } from './coordinator'
import { PersonaPlexVoiceAdapter } from './voiceAdapter'
import type { ChatResponder } from './adapters'

/** Renewal thresholds are checked on this cadence. */
const TICK_INTERVAL_MS = 15_000

export type UseSessionCoordinatorOptions = {
  conversationId: string | null
  /** Messages already on screen, so voice sees the conversation so far. */
  messages: Message[]
  summary?: string | null
  /** Model driving the agent and the chat responder. */
  chatModelId: string
  /** PersonaPlex model for the voice session. */
  voiceModelId: string
  /**
   * ASR interface for spoken turns: `streaming-asr` for the Nemotron worker,
   * undefined for the daemon's whisper default.
   */
  asrEngine?: string
  persona: string
  /** Produces ordinary chat answers when no agent session is bound. */
  responder?: ChatResponder
  /** Called for every message the coordinator stores, to update the chat view. */
  onMessage: (message: Message) => void
  onStatus: (status: string | null) => void
  /** Parent for the next stored message; the chat UI owns branch selection. */
  parentId: () => string | null
}

export type SessionCoordinatorHandle = {
  snapshot: CoordinatorSnapshot
  config: IntegrationConfig
  setConfig: (config: IntegrationConfig) => void
  /** True when spoken delivery is possible on this host. */
  canSpeak: boolean
  inputLevel: number
  outputLevel: number
  startVoice: () => Promise<void>
  endVoice: () => Promise<void>
  setMuted: (muted: boolean) => void
  submitText: (text: string) => Promise<string | null>
  stopSpeaking: () => Promise<void>
  cancelResponse: () => Promise<void>
  cancelAgentTask: () => Promise<void>
  /** Bind an agent session so voice and text turns share it. */
  bindAgentSession: (agentSessionId: string | null) => Promise<void>
}

export function useSessionCoordinator(
  options: UseSessionCoordinatorOptions
): SessionCoordinatorHandle {
  const [config, setConfigState] = useState<IntegrationConfig>(() => readIntegrationConfig())
  const [inputLevel, setInputLevel] = useState(0)
  const [outputLevel, setOutputLevel] = useState(0)

  // Refs so the adapters read current values without being rebuilt.
  const latest = useRef(options)
  latest.current = options

  const { adapters, coordinator } = useMemo(() => {
    const agent = new WorkerAgentAdapter(() => latest.current.chatModelId || undefined)
    const voice = new PersonaPlexVoiceAdapter({
      modelId: () => latest.current.voiceModelId,
      asrEngine: () => latest.current.asrEngine,
      onInputLevel: setInputLevel,
      onOutputLevel: setOutputLevel
    })
    const chat = new DaemonChatAdapter('unbound', {})
    const instance = new SessionCoordinator({
      chat,
      agent,
      voice,
      persona: latest.current.persona,
      config: readIntegrationConfig(),
      // Indirected through the ref so a rebuilt responder — it closes over the
      // selected model and the visible branch — reaches the coordinator without
      // tearing down a live voice session.
      responder: {
        respond: (request) => {
          const responder = latest.current.responder
          if (!responder) throw new Error('No chat model is available to answer that.')
          return responder.respond(request)
        },
        cancel: (correlationId) => latest.current.responder?.cancel(correlationId)
      },
      persistSummary: (conversationId, summary) => {
        void updateConversation(conversationId, { summary }).catch(() => undefined)
      }
    })
    return { adapters: { agent, voice }, coordinator: instance }
    // Built once for the lifetime of the host component: the adapters own
    // sockets and a process, so they must not be rebuilt on every render.
  }, [])

  const [snapshot, setSnapshot] = useState<CoordinatorSnapshot>(() => coordinator.snapshot())

  useEffect(() => {
    const unsubscribe = coordinator.subscribe(setSnapshot)
    return () => {
      unsubscribe()
    }
  }, [coordinator])

  useEffect(
    () => () => {
      // Voice is a renderer resource and goes with the component. The agent run
      // does not: it lives in the worker process.
      void coordinator.endVoiceSession()
      coordinator.dispose()
      adapters.agent.dispose()
    },
    [coordinator, adapters]
  )

  // Rebind whenever the conversation changes, adopting its agent session.
  const conversationId = options.conversationId
  useEffect(() => {
    if (!conversationId) return
    const chat = new DaemonChatAdapter(conversationId, {
      onMessage: (message) => latest.current.onMessage(message),
      onStatus: (status) => latest.current.onStatus(status),
      parentId: () => latest.current.parentId(),
      model: () => latest.current.chatModelId || undefined
    })
    // The adapter is per-conversation, so the coordinator is retargeted rather
    // than rebuilt; a live voice session survives switching conversations.
    coordinator.setChatAdapter(chat)
    void coordinator.attach(conversationId, {
      messages: latest.current.messages.map(toConversationMessage),
      summary: latest.current.summary ?? ''
    })
  }, [conversationId, coordinator])

  useEffect(() => {
    coordinator.setPersona(options.persona)
  }, [coordinator, options.persona])

  useEffect(() => {
    coordinator.setConfig(config)
    writeIntegrationConfig(config)
  }, [coordinator, config])

  // Renewal thresholds. Driven from here rather than inside the coordinator so
  // the coordinator stays free of timers and testable on a fake clock.
  useEffect(() => {
    if (!snapshot.voiceSessionId) return
    const timer = window.setInterval(() => void coordinator.tick(), TICK_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [coordinator, snapshot.voiceSessionId])

  const setConfig = useCallback((next: IntegrationConfig) => setConfigState(next), [])

  return {
    snapshot,
    config,
    setConfig,
    canSpeak: adapters.voice.canSpeak(),
    inputLevel,
    outputLevel,
    startVoice: useCallback(async () => {
      setConfigState((current) => ({ ...current, voiceEnabled: true }))
      await coordinator.startVoiceSession()
    }, [coordinator]),
    endVoice: useCallback(async () => {
      await coordinator.endVoiceSession()
      setConfigState((current) => ({ ...current, voiceEnabled: false }))
    }, [coordinator]),
    setMuted: useCallback((muted: boolean) => adapters.voice.setMuted(muted), [adapters]),
    submitText: useCallback((text: string) => coordinator.submitText(text), [coordinator]),
    stopSpeaking: useCallback(() => coordinator.cancelVoiceOutput(), [coordinator]),
    cancelResponse: useCallback(async () => {
      await coordinator.cancelCurrentResponse()
    }, [coordinator]),
    cancelAgentTask: useCallback(async () => {
      await coordinator.cancelAgentTask()
    }, [coordinator]),
    bindAgentSession: useCallback(
      async (agentSessionId: string | null) => {
        await adapters.agent.bindSession(agentSessionId)
        if (conversationId) {
          await coordinator.attach(conversationId, {
            messages: latest.current.messages.map(toConversationMessage),
            summary: latest.current.summary ?? ''
          })
        }
      },
      [adapters, coordinator, conversationId]
    )
  }
}
