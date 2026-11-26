# Remote daemon trust

`brazierd` speaks plaintext HTTP to loopback callers and HTTPS to other
devices. Its default is a loopback listener. Authentication prevents unauthorized calls, but it does not encrypt
bearer credentials, conversations, attachments, or model output in transit.

For remote access, keep Brazier on `127.0.0.1:7614` and put one of these in
front of it:

- an HTTPS reverse proxy with a certificate the client validates; or
- an encrypted private-network tunnel such as Tailscale/WireGuard, with access
  controls limiting which peers can reach the daemon.

If the proxy filters paths, forward the pairing preview and claim routes for
pairing. Desktop checks that the advertised origin reaches the same daemon by
sending an empty-code preview to the reserved all-zero pairing UUID before
issuing a real invite; it does not require public access to `/health`.
If desktop can verify the origin only through `/health`, it warns that the
preview route could not be verified. Check your proxy and daemon version before
sharing. Confirmation is still required.

A direct non-loopback listener serves HTTPS with a self-signed certificate
created once per data directory (`<data-dir>/tls/`). Its SHA-256 is printed in
the ready line as `tls_certificate_sha256`, and pairing invites created by the
desktop for its local daemon carry it as `certificateSha256`. Desktop and mobile
accept exactly that certificate for the invite's host and port, so LAN pairing
on `https://<lan-ip>:<port>` works without a proxy or a CA:

```sh
brazierd --service --host 0.0.0.0
```

Desktop opens its own daemon to the network the first time you create an invite:
it restarts the daemon on `0.0.0.0` at the configured server port with API keys
required, even if the server settings say localhost only. It stays reachable while
any paired client is active or an invite is pending, and returns to loopback at the
next launch after the last client is revoked.

Loopback callers can still use plaintext HTTP on the same port. Plaintext from
other devices gets `426 Upgrade Required` unless `--allow-insecure-remote` is
set. That flag is an unsafe escape hatch: a bearer sent over ordinary HTTP can
be read and replayed by anyone able to observe the connection. Do not expose
that listener to the public internet.

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

The unauthenticated `/health` probe returns only `status` and `database` to
non-loopback peers and to requests carrying `Forwarded`, `X-Forwarded-For` or
`X-Real-IP` (so a same-host reverse proxy does not count as local); the daemon
version, engine and boot id are reported to direct loopback callers and in
authenticated `/api/v1/daemon/info`.
The pairing compatibility probe exposes the stable instance UUID, not the
per-start boot ID, so desktop and mobile can verify an origin without exposing
restart times.

Each paired client has a stable ID and name plus explicit scopes:

- `inference`: OpenAI-compatible inference, conversations, attachments,
  memories, generation, voice, capabilities, and the tool catalogue;
- `management`: model/runtime/download/preferences/MCP configuration, support,
  and client/pairing administration;
- `agent`: Agent and Computer Use session and approval APIs.

The daemon stores only SHA-256 digests of high-entropy client credentials. A
credential is returned once when pairing is claimed. Revocation is effective
on the next request and does not require a daemon restart.

## Connect a desktop

1. On a desktop connected to the host with management access, open **Manage → Connections**. Enter the daemon's reachable HTTPS origin and choose the device name and scopes. Create an invite to show a five-minute QR code and **Copy invite** fallback. Share either only with the intended device.
2. On the receiving desktop, open **Manage → Connections** and paste the invite into the pairing ID field. Legacy copied text is also accepted; supply the reachable origin if the old text omits it. Choose **Review invite** to check the host, origin, and granted scopes before you send the claim.
3. Choose **Confirm and save**, then **Connect now**. The model library groups files on the active host separately from models served by each configured inference server. Changing the active connection in the library switches the entire desktop.

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
client's pairing UI or another authenticated channel. The client can check the
daemon identity, scopes, and expiry without consuming the code by sending
`POST /api/v1/auth/pairings/<pairing-id>/preview` with the same code body as the
claim. Include the preview's `daemon.instance_id` as
`expected_daemon_instance_id` in the claim to reject a changed host. The client
claims it at:

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
Revoking the client also invalidates its voice ticket and stops its live stream.
Agent and Computer approval decisions include the expected daemon execution
location, and the daemon rejects a stale or mismatched location before changing
the approval or grant ledger.
