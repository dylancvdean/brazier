# Brazier

Brazier is an MIT-licensed desktop client and local API for running open language
models. It is designed around a sandboxed Electron interface and an independent
Rust daemon so the same model, conversation, and tool infrastructure works in
the desktop app and in headless deployments.

The project is at the early implementation stage. On Linux, the vertical slice
provides:

- local conversation persistence with branching;
- OpenAI-compatible model, Chat Completions, and Responses endpoints;
- a deterministic development engine plus **llama.cpp** inference via managed or
  discovered `llama-server` and on-disk GGUF models;
- Hugging Face discovery (engine-filtered) and GGUF download into the app data
  directory;
- runtime management: managed release installs, llama.cpp builds from source
  into isolated prefixes, and explicit runtime activation;
- bundled safe tools (time, calculator, bounded web fetch, QuickJS sandbox)
  executed by the daemon in a multi-round tool loop;
- a capability model for text, vision, audio, video, reasoning, and tools;
- a cross-platform Electron chat interface with separate surfaces for usage
  (model picker, inference settings) and management (library, downloads,
  runtimes, engine configuration).

See [docs/architecture.md](docs/architecture.md) and
[docs/roadmap.md](docs/roadmap.md) for the implementation boundaries and
remaining engine work.

## Development

Prerequisites are Node.js 24+, pnpm 10, and Rust 1.93.

```sh
pnpm install
cargo test --workspace
pnpm dev
```

Run the daemon independently:

```sh
cargo run -p brazierd -- --no-auth
```

The daemon prints a `BRAZIER_READY` JSON record containing the selected address.
Authentication is enabled by default and a random session key is emitted in
that record for the desktop process.

## Security

Brazier collects no telemetry. Model engines and source forks execute native
code and must not be considered a security boundary. Non-upstream forks,
remote-code models, network tools, and code execution require explicit trust.
Please report vulnerabilities as described in [SECURITY.md](SECURITY.md).
