export type SmokeConnection = {
  address: string
  apiKey: string | null
}

export type PackageSmokeResult = {
  schema_version: 1
  kind: 'package-smoke'
  commit: string
  passed: boolean
  platform: 'macos' | 'linux' | 'windows'
  arch: string
  artifact: 'dmg' | 'appimage' | 'nsis'
  checks: {
    daemon_started: boolean
    safety_helper_present: boolean
    windows_sandbox_probe: boolean
    worker_loaded: boolean
    session_opened: boolean
    worker_shutdown: boolean
    session_deleted: boolean
    daemon_stopped: boolean
    clean_shutdown: boolean
  }
  metrics: {
    agent_worker_ready_ms: number
    agent_session_open_ms: number
  }
  error?: string
}

export type PackageSmokeDependencies = {
  connection: SmokeConnection
  warmWorker: () => Promise<void>
  openSession: (sessionId: string) => Promise<void>
  shutdownWorker: () => Promise<void>
  shutdownDaemon: () => Promise<void>
  checkSafetyHelper: () => Promise<void>
  fetch?: typeof fetch
  now?: () => number
  commit: string
  platform: PackageSmokeResult['platform']
  arch: string
  artifact: PackageSmokeResult['artifact']
}

function headers(connection: SmokeConnection): Headers {
  const value = new Headers({ 'content-type': 'application/json' })
  if (connection.apiKey) value.set('authorization', `Bearer ${connection.apiKey}`)
  return value
}

/**
 * Exercise the dependency path a source build cannot prove: the installed
 * daemon, the unpacked Pi dependency closure, and worker session hydration.
 * No model is called and the temporary daemon record is removed afterwards.
 */
export async function runPackageSmoke(
  dependencies: PackageSmokeDependencies
): Promise<PackageSmokeResult> {
  const now = dependencies.now ?? Date.now
  const fetch_ = dependencies.fetch ?? fetch
  const checks = {
    daemon_started: true,
    safety_helper_present: false,
    windows_sandbox_probe: dependencies.platform !== 'windows',
    worker_loaded: false,
    session_opened: false,
    worker_shutdown: false,
    session_deleted: false,
    daemon_stopped: false,
    clean_shutdown: false
  }
  const metrics = { agent_worker_ready_ms: 0, agent_session_open_ms: 0 }
  let sessionId: string | null = null
  let error: string | undefined

  try {
    await dependencies.checkSafetyHelper()
    checks.safety_helper_present = true

    if (dependencies.platform === 'windows') {
      const response = await fetch_(
        `${dependencies.connection.address}/api/v1/agent/capabilities`,
        {
          headers: headers(dependencies.connection),
          signal: AbortSignal.timeout(30_000)
        }
      )
      if (!response.ok) {
        throw new Error(`probing the packaged Windows sandbox returned ${response.status}`)
      }
      const capabilities = (await response.json()) as {
        sandbox?: { sandboxed_execution?: unknown }
      }
      if (capabilities.sandbox?.sandboxed_execution !== true) {
        throw new Error('the packaged Windows AppContainer launcher probe did not pass')
      }
      checks.windows_sandbox_probe = true
    }

    const response = await fetch_(`${dependencies.connection.address}/api/v1/agent/sessions`, {
      method: 'POST',
      headers: headers(dependencies.connection),
      body: JSON.stringify({
        title: 'Packaged agent smoke',
        workspace_path: null,
        model: 'smoke:no-model',
        runtime_id: 'simple',
        permission_mode: 'sandbox-only'
      }),
      signal: AbortSignal.timeout(10_000)
    })
    if (!response.ok) throw new Error(`creating the smoke session returned ${response.status}`)
    const created = (await response.json()) as { id?: unknown }
    if (typeof created.id !== 'string' || !created.id) {
      throw new Error('the daemon did not return a smoke session id')
    }
    sessionId = created.id

    let startedAt = now()
    await dependencies.warmWorker()
    metrics.agent_worker_ready_ms = now() - startedAt
    checks.worker_loaded = true

    startedAt = now()
    await dependencies.openSession(sessionId)
    metrics.agent_session_open_ms = now() - startedAt
    checks.session_opened = true
  } catch (cause) {
    error = cause instanceof Error ? cause.message : String(cause)
  } finally {
    const cleanupErrors: string[] = []
    // Let the worker persist its final session status before deleting the
    // temporary daemon record; reversing this order creates a misleading 404
    // during an otherwise successful packaged-app smoke.
    await dependencies.shutdownWorker().then(() => {
      checks.worker_shutdown = true
    }).catch((cause) => {
      cleanupErrors.push(`worker shutdown failed: ${cause instanceof Error ? cause.message : String(cause)}`)
    })
    if (sessionId) {
      const response = await fetch_(
        `${dependencies.connection.address}/api/v1/agent/sessions/${encodeURIComponent(sessionId)}`,
        {
          method: 'DELETE',
          headers: headers(dependencies.connection),
          signal: AbortSignal.timeout(10_000)
        }
      ).catch(() => null)
      checks.session_deleted = response?.ok === true
      if (!checks.session_deleted) {
        cleanupErrors.push(
          `deleting the smoke session ${response ? `returned ${response.status}` : 'failed'}`
        )
      }
    } else {
      // There is no durable session to remove when creation itself failed.
      checks.session_deleted = true
    }
    await dependencies.shutdownDaemon().then(() => {
      checks.daemon_stopped = true
    }).catch((cause) => {
      cleanupErrors.push(`daemon shutdown failed: ${cause instanceof Error ? cause.message : String(cause)}`)
    })
    checks.clean_shutdown =
      checks.worker_shutdown && checks.session_deleted && checks.daemon_stopped
    if (cleanupErrors.length > 0) {
      const cleanupError = cleanupErrors.join('; ')
      error = error ? `${error}; ${cleanupError}` : cleanupError
    }
  }

  return {
    schema_version: 1,
    kind: 'package-smoke',
    commit: dependencies.commit,
    passed: Object.values(checks).every(Boolean) && !error,
    platform: dependencies.platform,
    arch: dependencies.arch,
    artifact: dependencies.artifact,
    checks,
    metrics,
    ...(error ? { error } : {})
  }
}
