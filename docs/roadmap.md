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
  transcribing each utterance, plus an experimental result handoff. PersonaPlex
  is the only audible voice; the background answer remains in chat while a
  selectable strategy leaves the stream alone or reconnects/restarts it with a
  result prompt and optionally replays the exact triggering utterance. The
  background submission itself is selectable: Auto uses a local no-model
  classifier to leave lightweight conversation with PersonaPlex, Always keeps
  the original behavior, and Explicit requires a work cue. Short speech boost
  accepts Silero-confirmed 100 ms bursts, pads both sides for ASR, and retries
  an empty short result with the other recognizer when available. The streaming
  ASR worker keeps its model loaded
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
- **Model-based voice activity detection.** *Silero VAD v5 is bundled and in the
  live capture path.* The small ONNX model makes the normal speech/no-speech
  decision from the microphone frames already going to PersonaPlex, so a fan,
  keyboard, knock, or background audio does not take the assistant's turn just
  because it is loud. Inference is serialized and its 16 kHz / 512-sample
  windows are mapped back onto the original 24 kHz capture frames without
  opening a second microphone graph. The former adaptive energy detector stays
  as a visible meter, as extra echo protection while the assistant speaks, and
  as a recoverable fallback if WebAssembly or the model cannot initialize.
  Still owed: a checked-in room/noise corpus and false-interruption measurements
  across the microphones supported for release.
- **Spoken confirmation before destructive agent actions.** *Held calls are
  shown and can be answered in words; the warning stays.* A call the permission
  broker holds is shown prominently — what it will do, and whether it is inside
  the sandbox — and the next thing said answers it: an unmistakable yes allows
  it, a no refuses it, and anything else leaves it held, because a
  qualified answer is not consent and the transcript comes from a microphone, an
  energy gate, and a recogniser. Decisions are one-shot and written into the
  conversation, and the voice pane shows the held call with buttons, since
  speech should be the convenient path and not the only one. What this does not
  do is cover a session set to skip permissions: nothing is held there.
  Removing the warning would also need the transcript
  itself to be trustworthy, which is the VAD and latency work above.
- **PersonaPlex result-handoff experiments.** *Ready for hardware-backed
  comparison.* Platform TTS is gone, so PersonaPlex never competes with a second
  voice. A dropdown compares continuous conversation against same-process
  reconnects with direct or service-style result prompts, with and without
  realtime replay of the correlated utterance, plus a full process restart as a
  control. The upstream servers accept a new prompt when a WebSocket connection
  begins but do not support mutating the prompt of an active generation. Trial
  reports should identify the selected strategy, whether PersonaPlex first
  acknowledged or answered independently, restart/reconnect delay, and how
  faithfully it used the checked result. Hardware tuning should use the new
  background-routing, pre-handoff mute timing, and short-speech controls
  independently; current trial feedback favors the full process restart over
  same-process reconnects.

## Engine workshop

- **Toolchain detection** now looks where user-scoped installs actually go —
  `~/.local/bin`, `~/.cargo/bin`, Homebrew, Linuxbrew, Flatpak exports, Windows
  Apps — instead of only the `PATH` a windowed application inherits, which on
  macOS is four system directories and nothing else. What it finds is what gets
  run: ffmpeg is invoked at its resolved path rather than by name. The Runtimes
  build form lists each prerequisite with where it was found or the install
  command for the detected package manager; nothing is installed or elevated on
  the user's behalf. Still open: installing a user-scoped toolchain from inside
  the application, and guiding through GPU SDKs and drivers beyond the
  per-target hints that already exist.
- **Remote OpenAI-compatible connections** are in: named connections (base URL,
  optional key, on/off), their models listed as `remote:{connection}/{model}`
  beside local ones, chat and tool rounds routed to them with the model name the
  server itself uses, and keys stored with restricted file permissions and never
  returned by the API. A server that is asleep contributes nothing to the model
  list rather than emptying it. Not attempted: probing what a remote can
  actually do — capabilities are advertised as plain text in, text out, since
  the protocol says nothing and claiming vision would fail at the server.
- Add Linux vLLM (buildable today, not yet servable), engine diagnostics, and
  hardware-specific compatibility tests.
- Optional mlx-whisper and bundled ffmpeg for stronger offline media prep.
- Interruption UX polish; Nemotron VoiceChat if open self-host packaging lands.
  (Moshi MLX for Apple Silicon shipped: the `personaplex-mlx` recipe, the
  `local_web` backend, and activation are in place.)
- Broader generation families beyond sd.cpp (e.g. Diffusers for Hunyuan/CogVideoX)
  if needed.

## Agent mode follow-ons

- **`asarUnpack` coverage is fixed and enforced**, though a packaged build on
  each platform has still not been run. The worker keeps every import external,
  so what it needs at run time is Pi *and its whole dependency closure* — 94
  packages, mostly provider SDKs — while only `@earendil-works/**` was unpacked.
  `node_modules/**` is now unpacked rather than an enumerated list that would
  rot on the next upgrade, and a test walks the real closure and fails with the
  names of anything a narrower glob would miss. What remains is running the
  packaged application on macOS, Linux, and Windows and starting an agent
  session in it.
- Windows sandboxing. There is no backend today, so the daemon reports
  `isolated: false` and command execution is treated as host execution.
- **Live output streaming for `shell_run` is in.** Foreground commands now
  publish stdout and labelled stderr chunks into the existing agent tool
  timeline while they run; the bounded, persisted final result remains the
  authoritative output handed back to the model. `shell_start` plus
  `shell_output` remains the interactive-process path.
- **Model-generated compaction summaries** are in, alongside the deterministic
  digest rather than instead of it: the session's own model writes what was
  attempted and why an approach was abandoned, and the machine-built facts —
  files changed, commands run, unresolved failures — are appended verbatim
  underneath, so a model that forgets a file cannot erase it from the session.
  Every failure of the request (unreachable, slow, empty, malformed) falls back
  to the digest alone, because compaction usually runs when the context is
  already full and that is the worst moment to fail. Which half produced a
  summary is recorded on the session, and the narrative is cut at a sentence
  boundary if a model answers an eight-sentence instruction with an essay.
- **Optional MCP tools are available inside agent sessions.** Enabled servers'
  advertised schemas join the agent catalog and system prompt, and the worker
  refreshes that catalog when it opens a session. Calls reuse the existing MCP
  client but still enter through the agent execution broker: an MCP server is
  reported honestly as a host process with network reach, Ask mode holds every
  call for one-shot approval, Sandbox-only mode refuses it, and disabled or
  unadvertised tools cannot be invoked.

## Tool packs and release hardening

- Add the permission broker, optional browser automation, WASI code runtimes,
  and optional OCI execution (safe built-in tools and bounded web retrieval
  are implemented).
- **Configurable CORS** is in: `--allowed-origin` (repeatable, or
  comma-separated in `BRAZIER_ALLOWED_ORIGINS`) names extra browser origins
  beside the packaged UI and the dev server, validated at startup so a typo
  fails at the command line rather than at the first request. A wildcard is
  refused outright — this daemon holds a machine's conversations and can execute
  tools, so widening it stays deliberate and visible in the launch command.
  External binding already exists (`--host`, with keyless non-loopback access
  requiring `--allow-insecure-remote`).
- Add API-key management (rotation and per-client keys; today there is one key,
  generated at startup or supplied), redacted support bundles, SBOMs, signed
  updates, notarization, and release packaging.
- Complete public naming review before stable application identifiers ship.
