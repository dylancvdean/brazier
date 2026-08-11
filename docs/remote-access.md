# Remote daemon trust

`brazierd` speaks plaintext HTTP. Its safe default is therefore a loopback
listener. Authentication prevents unauthorized calls, but it does not encrypt
bearer credentials, conversations, attachments, or model output in transit.

For remote access, keep Brazier on `127.0.0.1:7614` and put one of these in
front of it:

- an HTTPS reverse proxy with a certificate the client validates; or
- an encrypted private-network tunnel such as Tailscale/WireGuard, with access
  controls limiting which peers can reach the daemon.

A direct non-loopback HTTP listener is an explicit unsafe escape hatch:

```sh
brazierd --service --host 0.0.0.0 --allow-insecure-remote
```

The flag is required even when authentication is enabled because a bearer sent
over ordinary HTTP can be read and replayed by anyone able to observe the
connection. Do not expose that listener to the public internet.

## Owner and paired clients

Service mode creates an owner-only bootstrap key at
`<data-dir>/service/api-key` when deployment tooling does not supply one. Owner
keys retain full access for compatibility. Remote clients should receive their
own paired credentials instead of a copy of that owner key.

The service key and ready descriptor are created with owner-only permissions on
Unix and a protected owner-only DACL on Windows. Desktop connection profiles
encrypt remote credentials through the operating-system credential store and
never return them to the renderer; the main process injects the active bearer
only for requests to the exact selected daemon origin. If a secure credential
store is locked or unavailable (including Electron's Linux `basic_text`
fallback), Brazier starts in Local recovery mode and refuses to consume a
one-time pairing code or persist a remote credential.

Each data directory also receives a stable daemon instance UUID. Set a human
label with `--daemon-name` or `BRAZIER_DAEMON_NAME`; authenticated
`/api/v1/daemon/info` responses include that identity plus platform and
architecture so clients can name the machine on which work will execute.

Each paired client has a stable ID and name plus explicit scopes:

- `inference`: OpenAI-compatible inference, conversations, attachments,
  memories, generation, voice, capabilities, and the tool catalogue;
- `management`: model/runtime/download/preferences/MCP configuration, support,
  and client/pairing administration;
- `agent`: Agent and Computer Use session and approval APIs.

The daemon stores only SHA-256 digests of high-entropy client credentials. A
credential is returned once when pairing is claimed. Revocation is effective
on the next request and does not require a daemon restart.

## Pairing API

An authenticated management client starts a short-lived pairing request:

```http
POST /api/v1/auth/pairings
Authorization: Bearer <owner-or-management-key>
Content-Type: application/json

{
  "client_name": "Dylan's laptop",
  "scopes": ["inference", "agent"],
  "ttl_seconds": 300
}
```

The response shows the pairing code once. Transfer it through the intended
client's pairing UI or another authenticated channel. The client claims it at:

```http
POST /api/v1/auth/pairings/<pairing-id>/claim
Content-Type: application/json

{ "code": "<one-time-code>" }
```

The claim route is code-authorized rather than bearer-authenticated. Codes
expire, are single-use, and stop accepting claims after eight incorrect
attempts. The successful response contains `api_key` once; store it in the
client's credential store. A paired management client can create another
pairing only with a subset of its own scopes; only an owner credential can
delegate arbitrary scopes.

Management clients can inspect or cancel pairing requests and inspect or revoke
clients with:

- `GET /api/v1/auth/pairings`
- `DELETE /api/v1/auth/pairings/<pairing-id>`
- `GET /api/v1/auth/clients`
- `DELETE /api/v1/auth/clients/<client-id>`

Invalid or revoked credentials receive `401 Unauthorized`. A valid client that
lacks the required scope receives `403 Forbidden`.

Voice stream tickets are bound to the paired client that created the session;
another inference client cannot list that ticket or terminate the session.
Agent and Computer approval decisions include the expected daemon execution
location, and the daemon rejects a stale or mismatched location before changing
the approval or grant ledger.
