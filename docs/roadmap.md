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

## Alpha

- Alpha scope is complete on macOS Apple Silicon (MLX) and cross-platform for
  llama.cpp. Remaining engine work moves to the workshop track below.

## Engine workshop

- Detect and install user-scoped toolchains; guide users through platform SDKs
  and drivers without silent elevation.
- Add Linux vLLM, remote OpenAI-compatible connections, engine diagnostics, and
  hardware-specific compatibility tests.
- Optional mlx-whisper and bundled ffmpeg for stronger offline media prep.
- **Realtime voice / PersonaPlex** — full-duplex speech-to-speech with persona
  control (NVIDIA PersonaPlex / Nemotron VoiceChat class). Separate product
  surface from file-attach chat; requires dedicated streaming I/O, not only
  blob hydration.
- **Image and video generation models** — discover/install generation
  checkpoints and invoke them either as an inbuilt tool call from chat or via
  a direct management/API action (not mixed into the chat completion path by
  default).

## Tool packs and release hardening

- Add the permission broker, optional browser automation, WASI code runtimes,
  and optional OCI execution (safe built-in tools and bounded web retrieval
  are implemented).
- Add API-key management, configurable external binding and CORS, redacted
  support bundles, SBOMs, signed updates, notarization, and release packaging.
- Complete public naming review before stable application identifiers ship.
