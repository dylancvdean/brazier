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
   `VoiceStream`, segmenting utterances with an energy/silence detector, and
   transcribing each finished utterance through the ASR endpoint the repo
   already exposes. Partial transcripts are therefore per-utterance-final
   rather than word-incremental; the coordinator already refuses to invoke the
   agent from partials, so this costs nothing in V1.

2. **PersonaPlex cannot be told what to say.** The persona is a process launch
   flag (`--text-prompt`); the wire protocol accepts audio only. Constrained
   rendering mode as described is not available at runtime, and there is no TTS
   engine in the repo either. The `speak()` side of the voice adapter is
   therefore a pluggable renderer: the platform speech synthesizer when the
   host provides one, and otherwise a text-only delivery that still reports
   `VOICE_RESPONSE_*` so the coordinator, UI, and tests behave identically.
   Nothing claims speech happened when it did not.

3. **Renewal restarts a process, not a stream.** A PersonaPlex "session" is a
   spawned Python server, and `SessionManager` allows exactly one. Renewal is
   end-session → create-session, which costs model load time. The coordinator
   keeps renewal at safe conversational boundaries and never ties it to agent
   state, which is what the plan actually requires.

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
- [x] **Phase 2 — agent-to-voice delivery.** Authoritative-response handoff,
      speech requests correlated to the authoritative message, completion and
      failure events, no duplicate assistant message.
- [x] **Phase 3 — interruptions and cancellation.** Separate
      `cancel_voice_output` / `cancel_current_response` / `cancel_agent_task`,
      utterance classification, queued follow-ups, superseded responses,
      correlation-scoped cancellation.
- [x] **Phase 4 — bounded context and renewal.** `VoiceContext` builder,
      compact summaries, session thresholds, renewal that leaves the agent
      session untouched.
- [x] **Phase 5 — hardening.** Event deduplication, error-state handling,
      structured diagnostics, configuration controls, integration tests.
