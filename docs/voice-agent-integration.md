# PersonaPlex · Chat · Agent integration

Brazier had three independent conversation surfaces: text chat (daemon-persisted
message graph), Agent mode (daemon-persisted agent sessions driven by a worker
process), and Voice mode (a PersonaPlex process the renderer talks to directly).
This document maps what already existed, records where the integration plan met
the real architecture, and tracks the staged work.

## Existing components

| Concern | Owner | Notes |
| --- | --- | --- |
| Conversation store | `crates/brazierd/src/db.rs` | SQLite message DAG, `conversations` / `messages`. The only conversation store; the integration extends it rather than adding a second. |
| Conversation API | `crates/brazierd/src/api.rs` | `/api/v1/conversations`, `…/messages`, `…/runs`, export/import. |
| Chat UI + send loop | `apps/desktop/src/renderer/src/App.tsx` | `submit()` writes the user message, streams `/v1/chat/completions`, writes assistant + tool messages. |
| Agent session store | `agent_store.rs`, `agent_types.rs` | Transcript, tool ledger, approvals, grants, artifacts. Authoritative for tools and task state. |
| Agent runtime | `apps/desktop/src/agent/` | Pi runtime in an Electron `utilityProcess`; reached from the renderer through `window.brazier.agent` (`protocol.ts` commands, `AgentEvent` stream). Survives renderer unmount. |
| Agent UI | `components/AgentMode.tsx` | Own composer and transcript, separate from chat. |
| Voice runtime | `crates/brazierd/src/voice.rs` | Spawns `moshi.server` (Linux CUDA) or `personaplex_mlx.local_web` (Apple Silicon) on an ephemeral loopback port. Single active session. |
| Voice API | `/api/v1/voice/sessions` | Create / read / end. Returns `ws_url` pointing **directly at the PersonaPlex process** — the daemon does not proxy the socket. |
| Voice transport | `renderer/src/audio/voiceStream.ts` | WebCodecs Opus + two AudioWorklets over the Moshi binary WebSocket. Tags: `0x00` handshake, `0x01` audio, `0x02` text. |
| Voice UI | `components/VoiceMode.tsx` | Persona box, start/mute/end, level meters, transcript pane. No conversation binding. |
| ASR | `whisper.rs`, `whisperkit.rs`, `streaming_asr.rs` | `POST /v1/audio/transcriptions` (batch, and SSE with `stream=true`). |

## Where the plan met the architecture

Four mismatches shaped the implementation.

1. **PersonaPlex emits no user transcript.** The `0x02` text frames on the Moshi
   socket are the *model's* inner monologue — what PersonaPlex is saying — not
   what the user said. The plan's `userTranscriptFinal` event has no source in
   the existing stack. Resolved by tapping the capture worklet inside
   `VoiceStream`, segmenting utterances with bundled Silero VAD plus a
   recoverable energy fallback, and transcribing each finished utterance through
   the ASR endpoint the repo already exposes. Partials come from transcribing at a pause before the
   utterance has closed, so they are per-pause rather than word-incremental; the
   coordinator refuses to invoke the agent from partials, and the one taken at
   the pause that turns out to end the turn is promoted to the final transcript
   because it covers exactly the same audio.

2. **PersonaPlex has per-connection prompts, not live prompt mutation.** The
   binary socket accepts client audio only, but both supported servers read
   `text_prompt` from the WebSocket URL when a connection starts. Reconnecting
   therefore keeps the loaded model process while resetting streaming state
   with a different prompt. It cannot mutate the active generation state.

   Platform TTS was removed: PersonaPlex is the only audible voice and the
   background answer remains authoritative in chat. Voice setup exposes five
   experiments: leave the stream continuous; reconnect with a direct result and
   replay the triggering utterance; reconnect with a service-role `Information:`
   prompt and replay; perform the same service-role experiment after a full
   process restart; or reconnect with service information but no replay as a
   control. Replay uses the exact utterance associated with the background turn,
   is paced in realtime, and bypasses capture callbacks so it cannot submit a
   duplicate turn.

   Transcription and background routing are independently selectable. Short
   speech boost lowers the accepted Silero-confirmed floor from 200 ms to
   100 ms, places silence before and after the clip, and retries an empty short
   result on the other ASR interface when both are installed. The background
   selector compares the original Always behavior with an entirely local Auto
   heuristic and an Explicit-only mode. Auto leaves short social and
   conversational turns with PersonaPlex, but still routes files, tools,
   workspace work, current facts, and active-task follow-ups.

   A second timing selector controls the old PersonaPlex stream while that
   background path runs. **Let PersonaPlex respond** is the natural control.
   **Mute when background work is detected** uses speculative pause transcripts
   as an early, reversible audio-only decision and confirms it on the final
   transcript. **Mute while every turn is routed** closes the output gate at
   sustained speech, before transcription, then reopens it for local / empty
   turns or only after the old stream has stopped and its replacement is ready.

3. **Renewal and reconnection have different costs.** A daemon voice session is
   a spawned Python server, and `SessionManager` allows exactly one. Ordinary
   experiment reconnects replace only the browser stream and keep that server
   loaded. Duration/context renewal and the explicit full-restart experiment use
   end-session → create-session and pay model load time. Neither path touches
   agent state.

4. **Agent sessions are not conversations.** `agent_sessions` has a workspace,
   permission mode, and its own transcript; it is deliberately separate from the
   chat graph. Rather than merging them, a conversation now records which agent
   session it is bound to, and the coordinator mirrors authoritative agent
   output into the conversation as `assistant_agent` messages.

Two smaller notes: the coordinator lives in the renderer because that is the
only process holding all three edges (audio devices, the agent IPC bridge, and
the chat API), and the long-running agent run itself lives in the worker
process, so a renderer reload loses UI state but not the task.

## Added and modified

Added:

- `crates/brazierd/src/db.rs` — migration 8: message `source` / `correlation_id`
  / `status` / `metadata_json`, conversation `agent_session_id` / `summary`.
- `apps/desktop/src/renderer/src/session/types.ts` — shared conversation,
  message, event, response-ownership, and voice-context types.
- `session/eventLog.ts` — event bus with idempotency.
- `session/coordinator.ts` — the session coordinator.
- `session/interruption.ts` — utterance classification.
- `session/voiceContext.ts` — bounded `VoiceContext` builder and summarizer.
- `session/config.ts` — integration configuration and defaults.
- `session/adapters.ts` — the three adapter interfaces.
- `session/agentAdapter.ts`, `session/chatAdapter.ts`, `session/voiceAdapter.ts`
  — implementations over the existing APIs.
- `session/useSessionCoordinator.ts` — React binding.
- `session/*.test.ts` — coordinator, interruption, voice-context, and
  integration-flow tests.

Modified:

- `crates/brazierd/src/types.rs`, `api.rs` — carry the new message and
  conversation fields; `PATCH /api/v1/conversations/{id}` and
  `PATCH /api/v1/conversations/{id}/messages/{messageId}`.
- `renderer/src/api.ts`, `types.ts` — the new fields and endpoints.
- `renderer/src/audio/voiceStream.ts` — capture tap for the transcript source.
- `components/VoiceMode.tsx`, `App.tsx` — shared-conversation voice mode.

## Staged checklist

- [x] **Phase 1 — shared conversation plumbing.** Normalized message model,
      correlation IDs, source attribution, one submission path for voice and
      text, conversation ↔ agent-session association.
- [x] **Phase 2 — agent-to-voice delivery.** Authoritative responses are stored
      once in chat and can drive a selected PersonaPlex reconnect/replay
      experiment without creating a duplicate assistant message.
- [x] **Phase 3 — interruptions and cancellation.** Separate
      `cancel_voice_output` / `cancel_current_response` / `cancel_agent_task`,
      utterance classification, queued follow-ups, superseded responses,
      correlation-scoped cancellation.
- [x] **Phase 4 — bounded context and renewal.** `VoiceContext` builder,
      compact summaries, session thresholds, renewal that leaves the agent
      session untouched.
- [x] **Phase 5 — hardening.** Event deduplication, error-state handling,
      structured diagnostics, configuration controls, integration tests.
- [x] **Phase 6 — PersonaPlex experiment harness and neural VAD.** PersonaPlex
      is the only audible voice; selectable continuous, reconnect, replay, and
      full-restart strategies compare prompt/data handoffs. Bundled Silero VAD
      v5 runs on the existing capture stream with an energy fallback.
- [x] **Phase 7 — local fast path and short-speech recovery.** Auto / Always /
      Explicit background routing can keep lightweight turns entirely inside
      PersonaPlex without adding another classifier model. Short speech boost
      accepts Silero-confirmed 100 ms turns, conditions their ASR audio, and
      retries an empty result on an alternate installed recognizer.
- [x] **Phase 8 — pre-handoff output timing.** Selectable natural,
      route-detected mute, and immediate mute modes compare response leakage
      against local-turn latency. Only reconnect/restart experiments engage the
      gate, and the checked-result replacement reopens it after the old stream
      has stopped.
