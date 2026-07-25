# Security policy

Please do not open public issues for suspected vulnerabilities. Until a
dedicated security mailbox is established, use GitHub private vulnerability
reporting on the canonical repository.

Brazier treats model runtimes, downloaded model code, source forks, tool
runtimes, and remote media as untrusted. The renderer never receives Node.js
integration, secrets are not passed to child engines, and external network
binding is opt-in.

## Agent mode

Agent mode executes actions a model chose. The trust boundaries are:

- The agent runtime runs in a separate `utilityProcess` with no host privileges.
  Its only path to the machine is `POST /api/v1/agent/exec` on the daemon, which
  applies the policy broker, the sandbox, and the executors in that order. A
  runtime that ignored the application's own tool wrappers still could not reach
  the filesystem or a shell.
- Approvals are daemon-side records bound to one session, tool, and argument
  hash. They cannot be replayed, reused after being spent, transplanted to
  another session, or applied to different arguments.
- Credential paths (`~/.ssh`, `~/.aws`, keychains, `~/.git-credentials`, browser
  profiles) and Brazier's own data directory are refused in every permission
  mode, including `skip-permissions`. Environment variables whose names look like
  credentials are stripped before a command starts.
- Sandboxing uses Seatbelt on macOS and Bubblewrap on Linux. Writes are confined
  to the workspace and a per-session scratch directory. Where no backend exists,
  the daemon reports `isolated: false` and the interface says so; command
  execution is then treated as host execution.

Agent mode is not a substitute for reviewing what an agent did. A sandbox limits
reach; it does not make an arbitrary command safe. Treat a workspace you hand to
an agent as writable by it.
