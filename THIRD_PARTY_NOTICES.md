# Third-party notices

Brazier is MIT licensed. It ships and links against third-party components with
their own terms. This file records the components Brazier depends on directly and
the notices that must travel with a distribution. The complete transitive
inventory is generated per release from `pnpm-lock.yaml` and `Cargo.lock`; this
file covers the components that carry attribution requirements or that execute as
part of the product.

User-downloaded model weights are not covered here. Hugging Face repositories
carry their own licenses, which the application surfaces and requires
acknowledging before a download. The bundled Silero VAD model is listed below.

## Application shell

**Electron** — MIT (Electron), with bundled Chromium (BSD-3-Clause and others)
and Node.js (MIT). Electron's own `LICENSES.chromium.html` must ship with a
packaged build; `electron-builder` includes it.

**React**, **React DOM** — MIT
**KaTeX** (`katex`) — MIT
**lucide-react** — ISC
**Inter** (`@fontsource-variable/inter`) — SIL Open Font License 1.1

**Silero VAD v5** — MIT model (`silero_vad_v5.onnx`)
Source: https://github.com/snakers4/silero-vad

**ONNX Runtime Web** — MIT
Source: https://github.com/microsoft/onnxruntime

These components run the bundled speech detector locally in the renderer.
Captured microphone audio does not leave the machine for voice activity
detection.

## Daemon

Rust dependencies are permissively licensed, primarily MIT, Apache-2.0, or dual
licensed under both (including `tokio`, `axum`, `sqlx`, `reqwest`, `serde`,
`sha2`, `uuid`, `tar`, `flate2`, `zip`, and `rquickjs`, which embeds QuickJS
under MIT), alongside the following permissive notice components:

- **`subtle`** — BSD-3-Clause (Copyright (c) 2016-2024 Isis Agora Lovecruft, Henry de Valence)
- **`ring`** (via `rustls`) — ISC, MIT, and OpenSSL/SSLeay (BoringSSL / Eric Young / OpenSSL contributors)
- **`webpki-roots`** (via `hyper-rustls`) — CDLA-Permissive-2.0 (Mozilla root CA bundle)
- **ICU4X / Unicode data crates** (`icu_*`, `unicode-ident`) — Unicode-3.0
- **`zlib-rs`** (via `flate2`) — Zlib
- **`libsqlite3-sys`** (via `sqlx`) — MIT bindings over SQLite (Public Domain)

## Engines and external tools

Model engines are installed by the user at run time and are not distributed with
Brazier. They carry their own licenses:

- `llama.cpp`, `whisper.cpp`, `stable-diffusion.cpp` — MIT
- `mlx-lm`, `mlx-vlm` — MIT
- NVIDIA PersonaPlex and Nemotron ASR checkpoints — NVIDIA model licenses
- `ffmpeg` — system-provided; LGPL-2.1 or GPL-2.0 depending on the build
- `bubblewrap` (Linux sandbox) — LGPL-2.0; system-provided
