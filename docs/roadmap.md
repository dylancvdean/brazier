# Roadmap

## Implemented foundation

- Rust daemon with authenticated loopback startup and graceful shutdown.
- SQLite conversation and message-branch persistence with versioned schema migrations.
- OpenAI-compatible Chat Completions and Responses, including SSE.
- Engine capability contracts and deterministic development engine.
- Hugging Face discovery with engine filtering and Unsloth preference.
- Source-fork build-plan validation using application-owned recipes.
- Electron shell with sandboxed renderer, history, branching, attachments,
  streaming cancellation, and production builds.
- Separated usage and management surfaces: model selection and inference
  settings in the topbar; library, downloads, runtimes, and engine
  configuration in a dedicated management panel.
- Source-build execution for llama.cpp into isolated prefixes with streamed
  logs, plus runtime inventory (managed releases, source builds, system
  binaries) with activation and deletion, in-flight build cancellation, and
  structured failure diagnostics with preserved logs.
- Apple-Silicon mlx-lm and mlx-vlm virtual environments (uv) and OpenAI-compatible
  server adapters, including MLX snapshot downloads and runtime activation.
- Bundled safe tools (current time, calculator, bounded web fetch with a
  private-network guard, QuickJS JavaScript sandbox) executed server-side in a
  multi-round tool loop for streamed and non-streamed chat.
- Content-addressed attachment storage with bounded media imports; messages
  reference blobs by digest instead of inline base64.
- Hugging Face token storage for gated model downloads (environment override
  or persisted token).
- Immutable run snapshots per assistant turn (model, settings, tools, response)
  with sidebar run history in the desktop app.
- llama-server health and capability probes exposed via daemon health/engine status.
- Conversation search, JSON import/export (including embedded attachment blobs
  and run metadata), persisted model download jobs, background download queue,
  in-flight download cancellation, and Hub license / remote-code acknowledgement.
- Media hydration: `brazier_blob` attachments expand to OpenAI `image_url` data
  URLs for vision models; honest capability advertising (mmproj → image only).
- whisper.cpp **batch ASR** (source builds, inventory, activation) with Whisper
  model download/discovery and `POST /v1/audio/transcriptions`.
- Video preprocess via system ffmpeg (frame sampling into the active vision
  model) plus optional soundtrack transcription when batch ASR is available.
- Distinct audio interface taxonomy in architecture and
  `/api/v1/capabilities` (`batch_asr`, `native_model_audio`, `streaming_asr`,
  planned `realtime_voice`).
- Conservative `audio_input: native` detection for known audio-LLM checkpoints
  (separate from Whisper ASR weights), with automatic fallback to batch ASR when
  the chat engine rejects `input_audio`.
- **Streaming ASR** via managed Python env + NVIDIA Nemotron ASR Streaming
  snapshots; `POST /v1/audio/transcriptions` with `stream=true` emits partial
  transcript SSE events.
- **whisper.cpp managed prebuilts** on Linux/Windows (official CLI release
  assets); macOS continues to use source builds (XCFramework-only releases).
- **stable-diffusion.cpp** image/video generation (managed + source builds),
  `POST /api/v1/generate/{image,video}`, chat tools `generate_image` /
  `generate_video`, and a dedicated Generate workspace mode.
- **Realtime voice (PersonaPlex / Moshi protocol)** — engine recipe, session
  API, Voice workspace mode with persona text; Apple Silicon Moshi/MLX and
  deeper duplex polish remain follow-ons.
- **Voice, chat, and agent in one conversation** — a session coordinator between
  the three, with the daemon's message graph as the only store: messages record
  which surface produced them, which turn they belong to, and what became of
  them. Response ownership is explicit, PersonaPlex owns nothing and is treated
  as untrusted output, and stopping speech, dropping an answer, and cancelling a
  task stay three separate controls. Voice supplies the two things the Moshi
  protocol lacks: user transcripts, by segmenting the microphone and
  transcribing each utterance, and spoken answers, rendered verbatim through the
  platform synthesizer with the model's own audio gated off so one question
  never gets two replies. The streaming ASR worker keeps its model loaded
  between utterances (3.1 s to 0.18 s a turn).
- **Agent mode** — interactive coding and system agent as a fourth workspace
  mode. Pi (`@earendil-works/pi-*`, MIT) is the first runtime, installed as a
  pinned dependency and reachable only through an adapter whose boundary a test
  enforces. Brazier owns the rest: 19 filesystem, shell, workspace, and
  permission tools with daemon-served schemas; a policy broker with `ask`,
  `sandbox-only`, and `skip-permissions` modes; approvals bound to a session,
  tool, and argument hash; Seatbelt and Bubblewrap sandboxes that report their
  isolation honestly (and refuse to pretend when there is none); session,
  transcript, tool-ledger, approval, grant, and artifact persistence;
  cancellation, compaction, and argument repair for weak local models. The agent
  runtime runs in its own `utilityProcess` and reaches the machine only through
  `POST /api/v1/agent/exec`.

## Alpha

- Alpha scope is complete on macOS Apple Silicon (MLX) and cross-platform for
  llama.cpp. Remaining engine work moves to the workshop track below.

## Voice follow-ons

The nearest-term track. Voice, chat, and the agent share one conversation now;
what is left is mostly the difference between working and trustworthy.

- **Whisper as the alternative transcription path.** *Unblocked and measured;
  the verdict needs a machine with both installed.* Batch whisper could not
  transcribe a spoken turn at all: capture runs at 24 kHz, whisper.cpp reads
  only 16 kHz and refuses rather than resamples, and a `.wav` extension was
  taken as proof the audio was already right. WAVs are now inspected and
  converted in process (downmix, decode, windowed-sinc resample) with no ffmpeg
  in the way of a microphone. Every utterance is timed and attributed to the
  interface that actually served it, and the live pane shows what each is
  costing — last, average, and multiple of real time — so whether one binary
  invocation beats a resident Python worker is now a reading rather than an
  argument.
- **Turn latency.** *Transcription moved inside the silence window; the window
  itself is unchanged.* A turn used to wait for 700 ms of silence and then wait
  again while the audio decoded. Transcription now starts at the first 300 ms
  pause, and when that pause turns out to be the end of the turn — which is most
  of them — the transcript is already in hand when the gate closes, so the
  second wait is gone. The audio is byte-identical to what closing delivers, so
  the early transcript can be the final one; when speech resumes instead, it is
  shown as a partial and discarded, which is what the coordinator's unused
  partial path was for. What each utterance waited for is measured beside what
  it cost. Still open: continuous feeding of the streaming endpoint, which needs
  a session protocol in the Python worker rather than a file per request, and
  would give word-incremental partials and let the close window itself shrink.
- **Voice activity detection that adapts to the room.** *Structure in place;
  the constants still want real rooms.* The gate now sits above a tracked noise
  floor instead of at a fixed level. The earlier attempt failed by learning only
  from frames below the gate — the audio a fan never produces — so this learns
  from every frame and decides by shape instead: the estimate falls freely and
  rises only while the last second looks flat, because room noise is steady and
  speech is not. It is bounded below by the level quiet rooms already worked at
  and above by where ordinary speech lives, so a very noisy room becomes
  unreliable rather than deaf. An utterance that opened before the room was
  learned is re-judged against it at close rather than sent to be transcribed.
  The gate and the room's level are shown live in the voice pane, and
  `adaptive: false` restores the fixed floor exactly. What is still owed is
  measurement against recorded rooms: fall and rise rates, the flatness
  threshold, and the speech-to-noise bar are reasoned about, not fitted.
- **Spoken confirmation before destructive agent actions.** *Held calls are read
  back and answered in words; the warning stays.* A call the permission broker
  holds is now spoken aloud — what it will do, and whether it is inside the
  sandbox — and the next thing said answers it: an unmistakable yes allows it,
  a no refuses it, and anything else leaves it held and says so, because a
  qualified answer is not consent and the transcript comes from a microphone, an
  energy gate, and a recogniser. Decisions are one-shot and written into the
  conversation, and the voice pane shows the held call with buttons, since
  speech should be the convenient path and not the only one. What this does not
  do is cover a session set to skip permissions: nothing is held there, so
  nothing is read back. Removing the warning would also need the transcript
  itself to be trustworthy, which is the VAD and latency work above.
- **Handing PersonaPlex the answer to speak.** *Still open; the voice is at
  least chosen now.* Its audio is gated off and the platform synthesizer speaks
  instead, because the persona is a process launch flag and the socket carries
  audio only — so the voice identity is lost for exactly the sentences that
  matter. The synthesizer's voice and speaking rate are now picked, previewed,
  and remembered rather than being whatever the operating system defaulted to,
  which narrows the gap without closing it. Constrained rendering still needs a
  client-to-server text frame in the runtime, which means owning a patch to the
  recipe rather than consuming it, and testing it needs the upstream tree and a
  GPU — neither is guesswork worth committing blind.

## Engine workshop

- Detect and install user-scoped toolchains; guide users through platform SDKs
  and drivers without silent elevation.
- Add Linux vLLM, remote OpenAI-compatible connections, engine diagnostics, and
  hardware-specific compatibility tests.
- Optional mlx-whisper and bundled ffmpeg for stronger offline media prep.
- Moshi MLX (Apple Silicon) flavor for realtime voice; interruption UX polish;
  Nemotron VoiceChat if open self-host packaging lands.
- Broader generation families beyond sd.cpp (e.g. Diffusers for Hunyuan/CogVideoX)
  if needed.

## Agent mode follow-ons

- Verify Agent mode inside a packaged build: the worker is an ESM bundle that
  imports Pi from `node_modules`, so `asarUnpack` coverage needs checking on all
  three platforms.
- Windows sandboxing. There is no backend today, so the daemon reports
  `isolated: false` and command execution is treated as host execution.
- Live output streaming for `shell_run` (long commands currently report at
  completion; `shell_start` plus `shell_output` covers the interactive case).
- Model-generated compaction summaries; V1 builds the digest deterministically
  from the transcript and the tool ledger.
- Optional MCP tools inside agent sessions, reusing the existing MCP client
  behind the same policy broker.

## Tool packs and release hardening

- Add the permission broker, optional browser automation, WASI code runtimes,
  and optional OCI execution (safe built-in tools and bounded web retrieval
  are implemented).
- Add API-key management, configurable external binding and CORS, redacted
  support bundles, SBOMs, signed updates, notarization, and release packaging.
- Complete public naming review before stable application identifiers ship.
