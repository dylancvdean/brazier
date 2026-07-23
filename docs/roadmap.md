# Roadmap

## Implemented foundation

- Rust daemon with authenticated loopback startup and graceful shutdown.
- SQLite conversation and message-branch persistence.
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
  binaries) with activation and deletion.
- Bundled safe tools (current time, calculator, bounded web fetch with a
  private-network guard) executed server-side in a multi-round tool loop for
  streamed and non-streamed chat.

## Alpha

- Replace in-memory attachment payloads with content-addressed storage and
  bounded media imports.
- Add llama.cpp binary discovery, managed releases, process lifecycle, health
  checks, and capability probes.
- Add Apple-Silicon mlx-lm and mlx-vlm virtual environments and server adapters.
- Add model download jobs, resume/checksum support, gated-model authentication,
  license acknowledgement, and remote-code warnings.
- Add immutable run snapshots, model settings, reasoning records, tool calls,
  import/export, search, and database migrations.

## Engine workshop

- Add build cancellation and richer failure diagnostics to source builds
  (execution, isolated prefixes, and streamed logs are implemented).
- Detect and install user-scoped toolchains; guide users through platform SDKs
  and drivers without silent elevation.
- Add Linux vLLM, remote OpenAI-compatible connections, engine diagnostics, and
  hardware-specific compatibility tests.

## Tool packs and release hardening

- Add the permission broker, optional browser automation, WASI code runtimes,
  and optional OCI execution (safe built-in tools and bounded web retrieval
  are implemented).
- Add API-key management, configurable external binding and CORS, redacted
  support bundles, SBOMs, signed updates, notarization, and release packaging.
- Complete public naming review before stable application identifiers ship.
