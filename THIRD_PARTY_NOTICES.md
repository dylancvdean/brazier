# Third-party notices

Brazier is MIT licensed. It ships and links against third-party components with
their own terms. This file records the components Brazier depends on directly and
the notices that must travel with a distribution. The complete transitive
inventory is generated per release from `pnpm-lock.yaml` and `Cargo.lock`; this
file covers the components that carry attribution requirements or that execute as
part of the product.

Model weights are not covered here. Hugging Face repositories carry their own
licenses, which the application surfaces and requires acknowledging before a
download.

## Agent runtime

**Pi** — `@earendil-works/pi-agent-core`, `@earendil-works/pi-ai`
Copyright (c) Mario Zechner
License: MIT
Source: https://github.com/badlogic/pi-mono

Pi provides the agent orchestration loop used by Agent mode. It is installed as a
normal package dependency at a pinned version and is never vendored into this
repository. It is reached only through the adapter in
`apps/desktop/src/agent/pi/`.

The packages were published as `@mariozechner/pi-agent-core` and
`@mariozechner/pi-ai` up to 0.73.x; those names are deprecated upstream in favour
of the `@earendil-works` scope, which is what Brazier depends on.

```
MIT License

Copyright (c) Mario Zechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Pi depends on provider SDKs (`openai`, `@anthropic-ai/sdk`,
`@aws-sdk/client-bedrock-runtime`, `@google/genai`, `@mistralai/mistralai`), all
Apache-2.0 or MIT. Brazier's Agent mode uses only the OpenAI-compatible path
against the local daemon; the other providers are never invoked.

## Application shell

**Electron** — MIT (Electron), with bundled Chromium (BSD-3-Clause and others)
and Node.js (MIT). Electron's own `LICENSES.chromium.html` must ship with a
packaged build; `electron-builder` includes it.

**React**, **React DOM** — MIT
**lucide-react** — ISC
**Inter** (`@fontsource-variable/inter`) — SIL Open Font License 1.1

## Daemon

Rust dependencies are MIT, Apache-2.0, or dual licensed under both, including
`tokio`, `axum`, `sqlx`, `reqwest`, `serde`, and `rquickjs`. `rquickjs` embeds
QuickJS (MIT).

## Engines and external tools

Model engines are installed by the user at run time and are not distributed with
Brazier. They carry their own licenses:

- `llama.cpp`, `whisper.cpp`, `stable-diffusion.cpp` — MIT
- `mlx-lm`, `mlx-vlm` — MIT
- NVIDIA PersonaPlex and Nemotron ASR checkpoints — NVIDIA model licenses
- `ffmpeg` — system-provided; LGPL-2.1 or GPL-2.0 depending on the build
- `bubblewrap` (Linux sandbox) — LGPL-2.0; system-provided
