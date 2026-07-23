# Security policy

Please do not open public issues for suspected vulnerabilities. Until a
dedicated security mailbox is established, use GitHub private vulnerability
reporting on the canonical repository.

Brazier treats model runtimes, downloaded model code, source forks, tool
runtimes, and remote media as untrusted. The renderer never receives Node.js
integration, secrets are not passed to child engines, and external network
binding is opt-in.
