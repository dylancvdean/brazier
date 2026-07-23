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

- Execute approved build plans with cancellation, logs, isolated prefixes, and
  rollback to the prior installation.
- Detect and install user-scoped toolchains; guide users through platform SDKs
  and drivers without silent elevation.
- Add Linux vLLM, remote OpenAI-compatible connections, engine diagnostics, and
  hardware-specific compatibility tests.

## Tool packs and release hardening

- Add the permission broker, safe built-in tools, bounded web retrieval,
  optional browser automation, WASI code runtimes, and optional OCI execution.
- Add API-key management, configurable external binding and CORS, redacted
  support bundles, SBOMs, signed updates, notarization, and release packaging.
- Complete public naming review before stable application identifiers ship.
