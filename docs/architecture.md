# Architecture

Brazier separates user interface concerns from model and tool execution.

## Processes and trust boundaries

The Electron main process starts `brazierd` on an ephemeral loopback port. A
random bearer token is passed to the sandboxed renderer through the minimal
preload bridge. The renderer has no Node.js integration and cannot create child
processes.

`brazierd` owns:

- the SQLite conversation graph and attachment index;
- engine installation, lifecycle, health checks, and capability negotiation;
- Hugging Face metadata and downloads;
- the OpenAI-compatible and Brazier management APIs;
- tool permission decisions and isolated workers.

Model engines are always separate child processes. An engine crash must end an
in-flight run without corrupting conversations or terminating the daemon.
Engine adapters translate the canonical Brazier request into the engine's native
wire protocol. Unsupported modalities or controls fail before inference.

## Data model

A conversation is a container for a directed acyclic message graph. Every
message has an optional parent. A displayed branch is the path from a selected
tip to a root; sending while viewing an older message creates another child.
Generation settings, engine identity, model revision, tool definitions, usage,
and errors will be recorded as immutable run snapshots.

SQLite stores metadata in WAL mode. Large attachments and models belong in
content-addressed stores under the application data directory. Credentials and
Hugging Face tokens belong in the operating-system credential store, never in
SQLite or renderer storage. Application preferences that must span renderer
origins also live in SQLite; onboarding completion, for example, is shared by
the development HTTP renderer and packaged `file://` renderer. The renderer
promotes the previous localStorage flag on first read for upgrade compatibility,
and migration 10 treats an already-existing database as an onboarded
installation so cross-origin upgrades do not replay the welcome flow.

## APIs

Public compatibility endpoints are `/v1/models`, `/v1/chat/completions`, and
`/v1/responses`. Brazier management endpoints are versioned below `/api/v1`.
Streaming uses Server-Sent Events. The daemon binds to loopback and requires a
random key by default. External binding and CORS policy require explicit
configuration. Keyless mode is restricted to loopback and cannot be combined
with a remote listener; `--allow-insecure-remote` only acknowledges plaintext
transport and does not disable authentication.

The canonical capability record separately describes input modalities, output
modalities, streaming, reasoning, and tools. Platform support is attached to an
engine installation, not inferred from model metadata alone.

## Audio and media interfaces

Brazier treats several different “audio” paths as distinct products. Do not
collapse them into a single Audio badge meaning.

1. **Batch ASR (preprocess)** — Implemented via `whisper.cpp`. An attached
   audio file is transcribed offline; the transcript is injected as text into
   the chat request. Works with ordinary text chat models. Exposed as
   `features.asr` / `features.audio_interfaces.batch_asr` and
   `POST /v1/audio/transcriptions`.

2. **Native model audio** — The selected chat model accepts audio tokens or
   OpenAI-style `input_audio` parts (`capabilities.audio_input = "native"`).
   Brazier hydrates blobs to `input_audio` instead of running Whisper. Detection
   is conservative (name/config heuristics for known audio-LLMs). If the chat
   engine rejects `input_audio`, Brazier falls back once to batch ASR using the
   original blob metadata.

3. **Streaming ASR** — Low-latency chunked transcription via the managed
   `streaming-asr` Python engine (NVIDIA Nemotron ASR Streaming / Transformers).
   Exposed as `audio_interfaces.streaming_asr` and
   `POST /v1/audio/transcriptions` with `stream=true` (SSE partials). Distinct
   from chat attachment hydration.

   The worker process is resident: it loads the model once and then takes one
   request per line on stdin, because loading Nemotron costs about three seconds
   and paying it per utterance is the difference between 3.1 s and 0.18 s a turn.
   A worker that dies, or that holds a different model, is replaced on the next
   request — one cold request, then warm again. Its source is put on
   `PYTHONPATH` from the recipe directory, so the worker always matches the
   daemon shipping it rather than whatever copy the last runtime build installed.

   A streaming decoder needs trailing audio to emit its final tokens, so
   utterances are padded with silence before transcription; without it
   "which test is failing" comes back as "which test".

   Audio arrives from the capture graph at 24 kHz, and whisper.cpp reads only
   16 kHz — it refuses rather than resamples — so WAVs are inspected and, when
   they need it, converted in the daemon (`wav.rs`) rather than through ffmpeg:
   a microphone should not need a system media toolchain to be heard. Every
   transcription response carries the engine that served it and the
   milliseconds it took, which is how a machine with both interfaces installed
   answers which one it should use.

4. **Realtime voice / PersonaPlex-class** — Full-duplex speech-to-speech with
   persona control over the Moshi WebSocket protocol (NVIDIA PersonaPlex
   primary flavor). Dedicated Voice workspace mode and
   `/api/v1/voice/sessions` — not file-attach chat. Apple Silicon Moshi/MLX
   is a planned follow-on.

**Remote servers.** A configured OpenAI-compatible endpoint is a fourth source
of chat models beside llama.cpp, MLX, and the local catalogue. Connections are
explicit — base URL, optional key, on/off — and their models appear as
`remote:{connection}/{model}`, so a conversation records where its answers came
from. Nothing is discovered on the network, requests carry the model name the
remote uses rather than Brazier's id, and keys live in `remote/connections.json`
with restricted permissions and are never returned by the API.

**Vision** attachments hydrate to `image_url` data URLs when the chat model
advertises `image`. **Video** uses system ffmpeg to sample frames into that
vision path, optionally with batch ASR on the soundtrack.

## Generation interfaces

Image and video *generation* are separate from vision *input* hydration and
from `/v1/chat/completions` modalities:

- Engine: `stable-diffusion.cpp` (`sd-cli`) with managed GitHub release
  binaries and source/fork builds.
- Models use `sdcpp-image:` / `sdcpp-video:` ids and optional `manifest.json`
  for multi-component checkpoints.
- Invoke via `POST /api/v1/generate/image|video`, bundled chat tools, or the
  Generate workspace mode. Defaults live in runtime settings
  (`default_image_gen_model`, `default_video_gen_model`).

Unsupported modalities fail in `brazierd` before inference.

## Acceleration targets

The target is picked from what the machine actually has, and one case is not a
preference but a safety property. An AMD GPU outside a ROCm build's compiled
architectures still enumerates as a ROCm device, so llama.cpp commits to the HIP
backend and then dispatches a kernel with no code object for the hardware: the
HSA queue wedges and the process aborts with a GPU hang rather than failing.

The only thing that knows which architectures are covered is the build itself.
HIP embeds device code as a fat binary whose bundles carry their target ids as
plain strings, so `rocm.rs` reads them out of the installed files and compares
against the architectures the kernel reports through KFD topology
(`/sys/class/kfd`, published by amdgpu with no ROCm userspace installed). A
mismatch is refused at install and activation, pointing at Vulkan. Nothing in
this repository lists which GPUs ROCm supports, so nothing goes stale when AMD
ships a new part or llama.cpp changes its release matrix.

Before a ROCm build exists there is nothing authoritative to check, so AMD
machines are recommended Vulkan — it runs on all of them, ROCm does not. ROCm
stays selectable and says plainly that the builds do not generally cover
integrated graphics. Whether a GPU is an APU is used only for that wording:
some APUs are covered and some discrete cards are not, so it never decides.

## Engine builds

Build recipes in `engine-recipes` are shipped and reviewed with Brazier. A user
may replace the Git origin and revision, but a fork cannot replace the command
list. Git hooks are disabled, commands are spawned with argument arrays rather
than a shell, and installations receive separate source, build, virtual
environment, and prefix directories.

This boundary reduces accidental command injection but does not make source
builds safe: compilers and Python build backends execute code from the selected
fork. The UI must show the complete plan and an untrusted-native-code warning
for every non-whitelisted origin before execution.

## Shared conversation: chat, voice, and the agent

Chat, Voice, and Agent mode can work on one conversation instead of three. A
session coordinator (`apps/desktop/src/renderer/src/session/`) sits between them
and owns the rules; the three subsystems keep their own responsibilities and are
reached only through adapters.

The conversation is the daemon's existing message graph — there is no second
store. A message now records the surface that produced it (`source`), the turn
it belongs to (`correlation_id`), and what became of it (`status`: `partial`,
`final`, `cancelled`, `superseded`, `failed`). A conversation records which
agent session its turns go to, plus the compact summary a voice session is
seeded with.

Ownership is explicit, never inferred from timing. A typed turn goes to the
agent when a session is bound to the conversation and to chat otherwise; a
spoken turn goes wherever Voice mode is pointed, and each destination there
names exactly one place, so it is always answerable which model replied and
where the next turn will land. PersonaPlex owns nothing. It may acknowledge,
but it never decides a tool result, a completion, or a fact. Its own generated text is treated as untrusted model
output: shown in the voice pane, never stored as an answer, never parsed as a
command.

Voice is a workspace mode like Agent and Generate: it takes the whole surface,
renders the shared conversation itself, and has no text box. Speech already has
a chosen destination, so a second input alongside it would be a second,
unlabelled one.

Pointing speech at the agent is marked experimental in the interface, and the
warning is not decoration. A misheard word reaches a subsystem that edits files
and runs commands, and the transcript still passes through both speech detection
and recognition. Silero VAD is materially more selective than a loudness gate,
but the words the agent acts on remain the least reliable input in the
application. The permission layer still judges every call, and held calls
receive the explicit confirmation described below.

A tool call the permission broker holds during a spoken turn is shown before it
runs — what it will do and whether it is inside the sandbox — and the next
utterance may answer it. Only an unmistakable yes allows it; anything qualified
leaves the call held, because the request came from a microphone and a
recogniser, and mishearing a refusal as consent runs a command that cannot be
taken back. Decisions are one-shot, recorded in the conversation, and equally
available as buttons.

Three cancellations stay separate, because they are different decisions:
stopping the audio, dropping the current answer, and abandoning the agent task.
Talking over the assistant does the first only — a long task survives a
barge-in — unless the user explicitly asks for cancellation.

Two things the plan for this needs are not in the Moshi protocol, and the voice
adapter supplies them:

- **User transcripts.** The socket's text frames are the model's own speech, so
  the user's words come from segmenting the captured microphone stream and
  transcribing each finished utterance through `/v1/audio/transcriptions`.
  Transcription starts at the first 300 ms pause rather than waiting for the
  700 ms that closes the utterance, so the decoding happens inside the silence
  window instead of after it. If the speaker carries on, the early transcript is
  shown as a partial and discarded; if they were done, it *is* the final one,
  byte-identical audio, and the turn starts without a second wait.

  Bundled Silero VAD v5 normally decides what is speech. Its stateful 16 kHz
  windows are aligned back to the original capture frames and run locally
  through ONNX Runtime Web. The earlier adaptive energy gate remains as an echo
  guard and as a recoverable fallback when model initialization fails. The
  active detector and its current reading are shown in the voice pane, because a
  session that has stopped hearing you should be able to say why. With short
  speech boost enabled, a Silero-confirmed 100 ms burst is retained rather than
  discarded, ASR receives deterministic leading and trailing silence, and an
  empty clip is tried once on the other installed recognizer. Standard mode
  keeps the former 200 ms / trailing-only behavior as an A/B control.
- **Routing without another router model.** PersonaPlex has already heard every
  utterance. A local lexical gate decides whether the transcript also needs the
  selected chat or agent model: Always preserves the original behavior, Auto
  keeps short conversational turns with PersonaPlex while routing workspace,
  tool, current-fact, and active-task language, and Explicit requires a concrete
  work cue. Skipped turns invoke and record nothing; the voice pane reports the
  decision so the classifier can be tuned from real sessions.
- **Experimentally handing back a result.** PersonaPlex is the only audible
  voice; platform TTS is not used. Both supported servers accept `text_prompt`
  when a WebSocket connection begins, although neither can mutate the prompt of
  an active generation. The adapter can therefore leave the current stream
  continuous, reconnect to the same loaded process with direct or service-style
  result information, optionally replay the exact correlated utterance at
  realtime pace, or restart the process as a comparison. The background answer
  remains authoritative in chat regardless of what PersonaPlex says. A separate
  pre-handoff timing control can leave the old stream audible, mute it when a
  speculative or final transcript routes to background work, or mute it as soon
  as sustained speech begins. A local or empty final transcript reopens the old
  stream; a background turn remains silent until the adapter has stopped it and
  opened the replacement, preventing the unchecked answer from leaking during
  the restart.

Voice-session renewal (duration, context size, a runtime restart) replaces the
PersonaPlex process at a safe conversational boundary and re-seeds it from the
bounded summary. The conversation and the agent session are untouched, and the
agent run lives in the worker process, so voice failing or restarting never
loses the task.

See [voice-agent-integration.md](voice-agent-integration.md) for the component
map and the mismatches this design had to resolve.

## Agent mode

Agent mode is a workspace mode beside Chat, Voice, and Generate, not a separate
application. It reuses model selection, engines, persistence, and the daemon
API. Four processes are involved:

```text
renderer (sandboxed)  →  main  →  agent worker (utilityProcess)
                                      │
                                      ▼
                                 Pi adapter
                              (broker tools)
                                      │
                                      ▼
                                   brazierd
                              policy / sandbox / models
```

Agent frameworks are pluggable runtimes selected by `runtime_id`. Two modes
share the Pi adapter today (`default simple`):

- **Simple** (`simple`) — the standard broker-sandboxed tool set: files, shell,
  git, subagents. This is the everyday surface.
- **Powerful** (`powerful`) — the same base plus the operator-enabled power tools
  (web search, web fetch, LSP diagnostics, …) toggled under Manage → Agent.
  Power tools are metadata in the catalog now; executors land in a later build.

- **Pi** (`@earendil-works/pi-agent-core`, `@earendil-works/pi-ai`, MIT) — the
  orchestration loop both modes run: tool-call parsing, streaming, context
  tracking, cancellation, and completion detection. Reached exclusively through
  `apps/desktop/src/agent/pi/`. Everything else — tool definitions, permission
  policy, sandboxing, execution, persistence, and the event stream — is Brazier's
  under `apps/desktop/src/agent/core/`. `pi` remains a legacy alias so sessions
  created before modes existed still open.

The daemon decides each session's tool set at creation from its mode: `simple`
gets the base catalog, `powerful` adds the power tools the operator enabled. An
explicit per-session tool list wins but must respect the mode — `simple`
sessions refuse power tools.

A boundary test fails the build if framework imports escape their adapter
directories. The worker selects an adapter per session from
`session.runtime_id`.

The worker holds no host privileges: its only route to the machine is
`POST /api/v1/agent/exec` on the daemon, and the daemon decides.
Tool schemas and the agent system prompt are served by the daemon too
(`/api/v1/agent/tools`, `/api/v1/agent/sessions/{id}/prompt`), so the contract a
Pi model sees always matches the executor and the policy behind it. Sessions
persist transcripts through the daemon for UI continuity and may register
Brazier-only MCP tools as broker tools.

Agent system prompts are workspace-scoped settings in the daemon database, so
all tasks grouped under the same workspace share one override. Agent mode's
header opens the prompt editor. The editable value is a template composed from
named shortcuts such as `{workspace}`, `{system_info}`, and `{tools}`. The editor
lists each shortcut below the template and lets users inspect its current,
read-only expansion. The daemon resolves shortcuts from live session state when
the worker opens a task; unknown shortcuts remain literal. With no override the
editor displays the complete default template—not a blank extension field.
Resetting removes the override and restores that template. Worktree tasks use
their source repository as the settings key. Execution permissions remain
enforced by the daemon policy broker independently of prompt text.

Agent mode also exposes the shared grouped tool picker. Its selection is stored
on the agent session and supplied both to the runtime's tool catalog and the
prompt template's `{tools}` expansion. Pre-task selections are written when the
session is created; changing an existing idle task rebuilds its worker session
so the catalog and prompt cannot drift apart. The daemon rejects tool names that
are not in its current built-in and MCP catalog.

### Policy and approvals

Every call is judged by `agent_policy` from the session's permission mode, the
tool's risk level, the paths in its arguments, and whether an OS sandbox actually
exists. The result is allow, ask, or refuse:

- `ask` — reads inside the workspace proceed; writes, execution, network use,
  and anything outside the workspace ask first.
- `sandbox-only` — sandboxed work proceeds without prompts and host access is
  refused outright.
- `skip-permissions` — sandboxed work is auto-approved; host actions still need
  their own separate opt-in.

An approval is a daemon-side record bound to one session, tool, and argument
hash. The worker cannot fabricate or reuse one: a grant issued for different
arguments, already spent, expired, or belonging to another session is refused.
Destructive and host actions never accept a session-wide grant. Credential paths
(`~/.ssh`, `~/.aws`, keychains, `~/.git-credentials`, and Brazier's own data
directory) are refused in every mode, including `skip-permissions`, and the
attempt is recorded.

### Sandbox

Sandboxing is per platform and reported honestly. macOS uses Seatbelt
(`sandbox-exec`) with a generated profile; Linux uses Bubblewrap (`bwrap`) 0.10+
(or a distribution build with the 0.6.3 secure-FD backport) with a read-only
root, a tmpfs over `$HOME`, and the workspace bound back from an open directory
descriptor. Writes are confined to the workspace and a per-session scratch
directory, which is also the only `TMPDIR` a tool sees — `/tmp` itself is not
writable, so a tool that hardcodes it fails visibly instead of escaping. Network
access is off unless the profile grants it.

Where no backend exists or its startup probe fails (including Linux with an old
Bubblewrap), the daemon reports `backend: "none"`, `isolated: false`, and the UI
says "No sandbox" verbatim. Running a program is then treated as host execution:
it is refused in `sandbox-only` mode and needs the host opt-in elsewhere. Nothing
in the stack may claim isolation it did not apply. Detection occurs when the
daemon starts, so installing or upgrading the backend requires a daemon restart.

Filesystem tools run in the daemon rather than the sandbox, and enforce their own
boundary: paths are normalized, then compared in canonical form so a symlink
pointing out of the workspace counts as outside it and requires elevation.

### Sessions and recovery

Agent sessions are separate from chat conversations. The daemon stores the
transcript in runtime-neutral form, plus the tool-execution ledger, approvals,
standing grants, and stored artifacts for output too large to send to a model.
Restoring a session rebuilds context and re-checks the workspace; it never
re-runs a command. Compaction rewrites the transcript into a digest that keeps
goals, decisions, changed files, commands, and unresolved failures.

## Tools

Tool execution uses the same capability model as engines. Every tool declares
network, filesystem, process, and secret requirements. Grants are scoped to one
call, conversation, domain, or user-selected directory.

The web pack uses an SSRF-aware fetcher. Computer Use mode adds a model-agnostic
observe–act loop: adapters (Fara XML today; generic tool dialects later) emit
normalized `ComputerAction` values, and a daemon broker executes them against a
browser target or — when OS permissions allow — a desktop target. Browser
sessions run in isolated headless Chromium processes controlled through CDP;
if no working Chromium installation is available, the broker reports the target
as unavailable instead of fabricating observations or successful actions. The
daemon durably stores Computer Use sessions, steps, memories, and pending
approvals so an interrupted task can be resumed after restart. Desktop drivers probe macOS Screen
Recording/Accessibility and Linux X11/Wayland portals and fail closed until both
capture and input are granted.

Workspace mode visibility (Chat, Agent, Generate, Voice, Computer) and app
update preferences live under Manage → Customization.

The code pack will prefer WASI and can delegate broader workloads to an
OCI runtime. Workers receive no network, host filesystem, or secrets unless the
specific grant provides them.
