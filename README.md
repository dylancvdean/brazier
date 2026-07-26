# Brazier

Brazier is an MIT-licensed desktop client and local API for running open weight AI
models. It is designed around a sandboxed Electron interface and an independent
Rust daemon so the same model, conversation, and tool infrastructure works in
the desktop app and in headless deployments.

## Current features

| Feature | Status |
| --- | --- |
| Platforms | 🟨 Alpha on macOS and Linux; Windows support is partial |
| Desktop app | ✅ Electron workspaces for chat, generation, voice, agents, and model management |
| Engines | ✅ llama.cpp, MLX-LM/VLM, stable-diffusion.cpp, whisper.cpp, WhisperKit, Nemotron ASR, PersonaPlex, and remote OpenAI-compatible servers |
| Custom engine builds | ✅ Build llama.cpp, MLX, whisper.cpp, stable-diffusion.cpp, and PersonaPlex forks from a repository URL on supported platforms |
| Runtime management | ✅ Managed installs, source builds, system runtime discovery, activation, logs, and removal |
| Model discovery | 🟨 Hardware-sized recommendations and Hugging Face search; some media and voice recommendations are still placeholders |
| Chat | ✅ Persistent branching conversations, search, attachments, cancellation, run history, and import/export |
| API | ✅ OpenAI-compatible Models, Chat Completions, and Responses, plus transcription, generation, and agent endpoints |
| Tools and MCP | ✅ Built-in utilities, media generation, multi-round tool use, and custom stdio MCP servers |
| Multimodal input | ✅ Image, audio, and sampled video input with capability-aware fallback |
| Agent | 🟨 Sandboxed Pi core on macOS and Linux; Windows has no agent sandbox and agent MCP support is still planned |
| Media generation | 🟨 Curated image/video models and managed runtimes; backend reliability varies and AMD APU Vulkan support is experimental |
| Speech recognition | ✅ whisper.cpp, WhisperKit, and Nemotron streaming ASR |
| Multi-GPU | 🟨 Manual llama.cpp GPU splits are supported; automatic and cross-engine placement are not |
| Voice | 🟨 PersonaPlex bidirectional voice and shared chat/agent conversations work; voice safety and VAD remain alpha-quality |

## Development

Prerequisites are Node.js 24+, pnpm 11, and Rust 1.93.

```sh
pnpm install
cargo test --workspace
pnpm dev
```

The window has no menu bar, so development builds bind the two things a menu
would normally provide: `Cmd/Ctrl+R` reloads the renderer and
`Cmd/Ctrl+Alt+I` toggles developer tools.


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


## Security

Brazier collects no telemetry. Non-upstream forks,
remote-code models, network tools, and code execution require explicit trust.

Agent mode runs model-chosen commands on your machine. It defaults to asking
before anything writes, executes, or leaves the workspace, and the interface
states plainly when no OS sandbox is available. On Windows there is none yet. `skip-permissions` mode exists and
still refuses credential paths, but it removes the prompts.