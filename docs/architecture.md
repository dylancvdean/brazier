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
SQLite or renderer storage.

## APIs

Public compatibility endpoints are `/v1/models`, `/v1/chat/completions`, and
`/v1/responses`. Brazier management endpoints are versioned below `/api/v1`.
Streaming uses Server-Sent Events. The daemon binds to loopback and requires a
random key by default. External binding, CORS policy, and keyless access require
explicit configuration.
The headless CLI requires `--allow-insecure-remote` in addition to `--no-auth`
before it will expose a keyless API beyond loopback.

The canonical capability record separately describes input modalities, output
modalities, streaming, reasoning, and tools. Platform support is attached to an
engine installation, not inferred from model metadata alone.

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

## Tools

Tool execution uses the same capability model as engines. Every tool declares
network, filesystem, process, and secret requirements. Grants are scoped to one
call, conversation, domain, or user-selected directory.

The web pack will use an SSRF-aware fetcher and an optional isolated browser
worker. The code pack will prefer WASI and can delegate broader workloads to an
OCI runtime. Workers receive no network, host filesystem, or secrets unless the
specific grant provides them.
