# Brazier

Brazier is an MIT-licensed desktop client and local API for running open language
models. It is designed around a sandboxed Electron interface and an independent
Rust daemon so the same model, conversation, and tool infrastructure works in
the desktop app and in headless deployments.

The project is at the early implementation stage. On Linux and macOS Apple Silicon, the vertical slice
provides:

- local conversation persistence with branching;
- OpenAI-compatible model, Chat Completions, and Responses endpoints;
- a deterministic development engine plus **llama.cpp** inference via managed or
  discovered `llama-server` and on-disk GGUF models;
- on **macOS Apple Silicon**, **mlx-lm** and **mlx-vlm** Python runtimes built into
  isolated virtual environments with OpenAI-compatible HTTP servers;
- Hugging Face discovery (engine-filtered) and GGUF download into the app data
  directory;
- runtime management: managed release installs, llama.cpp builds from source
  into isolated prefixes, and explicit runtime activation;
- bundled safe tools (time, calculator, bounded web fetch, QuickJS sandbox)
  executed by the daemon in a multi-round tool loop;
- a capability model for text, vision, audio, video, reasoning, and tools;
- an **Agent** workspace mode for coding and system tasks: the agent edits files
  and runs commands inside a workspace folder you choose, sandboxed with Seatbelt
  on macOS or Bubblewrap on Linux, with per-action approvals and a full record of
  what ran;
- a cross-platform Electron chat interface with separate surfaces for usage
  (model picker, inference settings) and management (library, downloads,
  runtimes, engine configuration).

See [docs/architecture.md](docs/architecture.md) and
[docs/roadmap.md](docs/roadmap.md) for the implementation boundaries and
remaining engine work.

## Development

Prerequisites are Node.js 24+, pnpm 11, and Rust 1.93.

```sh
pnpm install
cargo test --workspace
pnpm dev
```

The first launch shows a welcome checklist for host tools (git, cmake, a C++
toolchain, uv, ffmpeg). To reopen it during development:

```sh
pnpm dev:welcome
# or: BRAZIER_FORCE_WELCOME=1 pnpm dev
```

Run the daemon independently:

```sh
cargo run -p brazierd -- --no-auth
```

The daemon prints a `BRAZIER_READY` JSON record containing the selected address.
Authentication is enabled by default and a random session key is emitted in
that record for the desktop process.

The application icons are derived from `assets/brazier-logo.png` and committed,
so building needs no image tooling. After changing the artwork, regenerate them
and commit the result:

```sh
uv run --with pillow apps/desktop/scripts/make-icons.py
```

## Security

Brazier collects no telemetry. Model engines and source forks execute native
code and must not be considered a security boundary. Non-upstream forks,
remote-code models, network tools, and code execution require explicit trust.

Agent mode runs model-chosen commands on your machine. It defaults to asking
before anything writes, executes, or leaves the workspace, and the interface
states plainly when no OS sandbox is available — on Windows there is none yet, so
commands there are treated as host execution. `skip-permissions` mode exists and
still refuses credential paths, but it removes the prompts; use it deliberately.

Please report vulnerabilities as described in [SECURITY.md](SECURITY.md).
