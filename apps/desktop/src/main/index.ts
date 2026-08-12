import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { accessSync, constants, createWriteStream, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync, type WriteStream } from 'node:fs'
import { randomBytes, randomUUID } from 'node:crypto'
import { writeFile } from 'node:fs/promises'
import { homedir, totalmem } from 'node:os'
import { join, resolve } from 'node:path'
import { app, BrowserWindow, clipboard, dialog, globalShortcut, ipcMain, Menu, safeStorage, screen, session, shell, type IpcMainInvokeEvent } from 'electron'

import { AgentSupervisor, registerAgentIpc } from './agent'
import {
  ConnectionProfileManager,
  ConnectionProfileStore,
  viewConnectionProfile,
  type ConnectionCredentialCodec,
  type ConnectionTestResult,
  type ClaimedConnection,
  type ConnectionProfileSummary,
  type DaemonConnection,
  type PairingClaimInput,
  type RemoteConnectionProfileInput
} from './connections'
import {
  getUpdateSettings,
  saveUpdateSettings,
  startUpdates,
  type UpdateCheckResult
} from './updates'
import { runPackageSmoke, type PackageSmokeResult } from './packageSmoke'
import { parseNvidiaSmiMemoryGib } from './qualificationHost'
import { isSafeExternalUrl, isTrustedRendererUrl } from './rendererTrust'

declare const __BRAZIER_BUILD_COMMIT__: string

function rendererIndexPath(): string {
  return join(__dirname, '../renderer/index.html')
}

function rendererTrustOptions(): { developmentUrl?: string; packagedIndexPath: string } {
  return {
    ...(process.env.ELECTRON_RENDERER_URL
      ? { developmentUrl: process.env.ELECTRON_RENDERER_URL }
      : {}),
    packagedIndexPath: rendererIndexPath()
  }
}

function assertTrustedIpcSender(event: IpcMainInvokeEvent): void {
  const frame = event.senderFrame
  if (
    !frame ||
    frame !== event.sender.mainFrame ||
    !isTrustedRendererUrl(frame.url, rendererTrustOptions())
  ) {
    throw new Error('Refusing privileged IPC from an untrusted renderer document.')
  }
}

function handleTrusted<Args extends unknown[], Result>(
  channel: string,
  listener: (event: IpcMainInvokeEvent, ...args: Args) => Result
): void {
  ipcMain.handle(channel, (event, ...args) => {
    assertTrustedIpcSender(event)
    return listener(event, ...(args as Args))
  })
}

// Electron may render through XWayland for stability, but the daemon must see
// the actual login session to choose the correct desktop-control driver. The
// dev launcher intentionally unsets WAYLAND_DISPLAY, so recover the compositor
// socket from its per-user runtime directory in that case.
function computerWaylandDisplay(): string | undefined {
  if (process.env.WAYLAND_DISPLAY) return process.env.WAYLAND_DISPLAY
  if (process.platform !== 'linux' || process.env.XDG_SESSION_TYPE !== 'wayland') return undefined
  try {
    return readdirSync(process.env.XDG_RUNTIME_DIR ?? '').find((entry) => /^wayland-\d+$/.test(entry))
  } catch {
    return undefined
  }
}
const COMPUTER_WAYLAND_DISPLAY = computerWaylandDisplay()
// A distro package runs `electron /usr/lib/brazier`, which Electron considers
// an unpackaged app even though its renderer and daemon are installed there.
const installedApp = app.isPackaged || process.env.BRAZIER_INSTALLED === '1'

/** Session log, opened on first write. See `appendLog`. */
let logStream: WriteStream | undefined

// Names the process for menus, notifications, and crash reports. The Dock
// label in development comes from the Electron bundle instead, which
// `scripts/ensure-electron.mjs` renames.
app.setName('Brazier')

/**
 * Linux launch flags must run before app.ready.
 *
 * Observed failure mode on this host (KDE Wayland + tmpfs /dev/shm|/tmp with
 * usrquota): Chromium child processes cannot create POSIX shared memory
 * (access(W_OK|X_OK) returns ESRCH), the GPU/zygote path melts down, and the
 * window stays the backgroundColor black shell.
 *
 * Rendering mitigations retained here do not disable Chromium's process
 * sandbox. A package whose sandbox helper is misconfigured must fail visibly
 * instead of silently running web content without isolation.
 * - disable-dev-shm-usage: avoid /dev/shm
 * - force X11 (XWayland) instead of Ozone/Wayland color-management path
 * - software compositing to avoid the crashing GPU process
 */
if (process.platform === 'linux') {
  // Prefer X11 even when WAYLAND_DISPLAY is set. Electron reads the env var.
  if (!process.env.ELECTRON_OZONE_PLATFORM_HINT) {
    process.env.ELECTRON_OZONE_PLATFORM_HINT = 'x11'
  }
  if (process.env.WAYLAND_DISPLAY && process.env.ELECTRON_OZONE_PLATFORM_HINT === 'x11') {
    // Keep DISPLAY (XWayland) but stop Chromium auto-selecting Wayland.
    delete process.env.WAYLAND_DISPLAY
  }

  app.commandLine.appendSwitch('disable-dev-shm-usage')
  app.commandLine.appendSwitch('ozone-platform-hint', process.env.ELECTRON_OZONE_PLATFORM_HINT)
  if (process.env.ELECTRON_OZONE_PLATFORM_HINT === 'x11') {
    app.commandLine.appendSwitch('ozone-platform', 'x11')
  }

  // Default to software GL on Linux. Override with BRAZIER_ELECTRON_SOFTWARE_GL=0
  // for hardware acceleration once the host paints correctly.
  if (process.env.BRAZIER_ELECTRON_SOFTWARE_GL !== '0') {
    app.disableHardwareAcceleration()
    app.commandLine.appendSwitch('disable-gpu')
  }
}

type Connection = {
  address: string
  api_key: string | null
  local_control_key: string | null
}

let computerSafetyOverlay: BrowserWindow | null = null
let nativeComputerSafety: ChildProcessWithoutNullStreams | null = null
let computerSafetyStarting: Promise<void> | null = null
let computerSafetyGeneration = 0
let computerOverlayWatchdog: NodeJS.Timeout | null = null
let computerUseActive = false

type InputGuardStatus = {
  supported: boolean
  installed: boolean
  secure: boolean
  ready: boolean
  current: boolean
  version: string | null
  detail: string
}

const INPUT_GUARD_INSTALLED_PATH = '/usr/lib/brazier-input-guard'

function inputGuardSourcePath(): string {
  const executable = 'brazier-input-guard'
  const installedSource = join(__dirname, '../..', executable)
  if (process.env.BRAZIER_INSTALLED === '1' || existsSync(installedSource)) {
    return installedSource
  }
  if (app.isPackaged) return join(process.resourcesPath, 'bin', executable)
  return join(repositoryRoot(), 'target', 'debug', executable)
}

function runChild(
  command: string,
  args: string[],
  timeoutMs = 15_000
): Promise<{ code: number | null; signal: NodeJS.Signals | null; stdout: string; stderr: string }> {
  return new Promise((resolveChild, rejectChild) => {
    const child = spawn(command, args, {
      cwd: installedApp ? undefined : repositoryRoot(),
      env: {
        ...process.env,
        ...(COMPUTER_WAYLAND_DISPLAY ? { WAYLAND_DISPLAY: COMPUTER_WAYLAND_DISPLAY } : {})
      },
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true
    })
    let stdout = ''
    let stderr = ''
    const timeout = setTimeout(() => {
      child.kill('SIGTERM')
      rejectChild(new Error(`${command} timed out.`))
    }, timeoutMs)
    child.stdout.on('data', (chunk) => { stdout = `${stdout}${chunk.toString()}`.slice(-8192) })
    child.stderr.on('data', (chunk) => { stderr = `${stderr}${chunk.toString()}`.slice(-8192) })
    child.once('error', (error) => {
      clearTimeout(timeout)
      rejectChild(error)
    })
    child.once('exit', (code, signal) => {
      clearTimeout(timeout)
      resolveChild({ code, signal, stdout, stderr })
    })
  })
}

async function inputGuardStatus(): Promise<InputGuardStatus> {
  if (process.platform !== 'linux') {
    return {
      supported: false,
      installed: false,
      secure: false,
      ready: false,
      current: false,
      version: null,
      detail: 'The privileged keyboard safety fallback is only used on Linux.'
    }
  }

  let metadata
  try {
    metadata = lstatSync(INPUT_GUARD_INSTALLED_PATH)
  } catch {
    return {
      supported: true,
      installed: false,
      secure: false,
      ready: false,
      current: false,
      version: null,
      detail: 'Not installed. The Wayland portal remains the preferred emergency shortcut.'
    }
  }
  const mode = metadata.mode & 0o7777
  const secure = metadata.isFile() && metadata.uid === 0 && (mode & 0o2000) !== 0 && (mode & 0o022) === 0
  if (!secure) {
    return {
      supported: true,
      installed: true,
      secure: false,
      ready: false,
      current: false,
      version: null,
      detail: `${INPUT_GUARD_INSTALLED_PATH} has unsafe ownership or permissions. Repair the installation before using it.`
    }
  }

  try {
    const result = await runChild(INPUT_GUARD_INSTALLED_PATH, ['--probe'], 5_000)
    const readyLine = result.stdout.split('\n').find((line) => line.startsWith('READY '))
    const version = readyLine?.slice('READY '.length).trim() || null
    const ready = result.code === 0 && version !== null
    const current = version === app.getVersion()
    return {
      supported: true,
      installed: true,
      secure: true,
      ready,
      current,
      version,
      detail: ready
        ? current
          ? 'Ready. It runs only while Computer Use is active and reports only Ctrl+Shift+Esc.'
          : `Version ${version ?? 'unknown'} is installed; update it to match Brazier ${app.getVersion()}.`
        : result.stderr.trim() || 'The installed fallback could not open a keyboard input device.'
    }
  } catch (cause) {
    return {
      supported: true,
      installed: true,
      secure: true,
      ready: false,
      current: false,
      version: null,
      detail: cause instanceof Error ? cause.message : String(cause)
    }
  }
}

async function ensureInputGuardSource(): Promise<string> {
  const source = inputGuardSourcePath()
  if (!installedApp && !existsSync(source)) {
    const build = await runChild('cargo', ['build', '-p', 'brazier-input-guard'], 120_000)
    if (build.code !== 0) {
      throw new Error(build.stderr.trim() || 'Could not build the keyboard safety fallback.')
    }
  }
  if (!existsSync(source)) {
    throw new Error('This Brazier installation does not contain brazier-input-guard.')
  }
  return source
}

async function setupInputGuard(): Promise<InputGuardStatus> {
  if (process.platform !== 'linux') throw new Error('The input guard is only available on Linux.')
  const source = await ensureInputGuardSource()
  const pkexec = existsSync('/usr/bin/pkexec') ? '/usr/bin/pkexec' : 'pkexec'
  const install = await runChild(
    pkexec,
    ['/usr/bin/install', '-o', 'root', '-g', 'input', '-m', '2755', source, INPUT_GUARD_INSTALLED_PATH],
    120_000
  )
  if (install.code !== 0) {
    const detail = install.stderr.trim()
    throw new Error(detail || 'Administrator authorization was cancelled or the input guard could not be installed.')
  }
  const status = await inputGuardStatus()
  if (!status.ready) throw new Error(status.detail)
  return status
}

function positionComputerSafetyOverlay(): void {
  if (!computerSafetyOverlay || computerSafetyOverlay.isDestroyed()) return
  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
  const { x, y, width } = display.workArea
  const [overlayWidth] = computerSafetyOverlay.getSize()
  computerSafetyOverlay.setPosition(Math.round(x + (width - overlayWidth) / 2), y + 16, false)
}

function stopComputerOverlayWatchdog(): void {
  if (computerOverlayWatchdog) clearInterval(computerOverlayWatchdog)
  computerOverlayWatchdog = null
}

function stopNativeComputerSafety(): void {
  const child = nativeComputerSafety
  nativeComputerSafety = null
  if (child && child.exitCode === null && child.signalCode === null) child.kill('SIGTERM')
}

function safetyOverlayMarkerPath(): string {
  return join(dataDirectory(), 'computer-safety-overlay.ready')
}

function writeSafetyOverlayMarker(): void {
  const directory = dataDirectory()
  mkdirSync(directory, { recursive: true })
  writeFileSync(safetyOverlayMarkerPath(), 'ready\n', { mode: 0o600 })
}

function clearSafetyOverlayMarker(): void {
  try {
    rmSync(safetyOverlayMarkerPath(), { force: true })
  } catch {
    // Best-effort; authority revoke below is the hard stop.
  }
}

/** Revoke every session's desktop authority even if the renderer is wedged. */
async function revokeAllDesktopAuthority(): Promise<void> {
  clearSafetyOverlayMarker()
  try {
    const profiles = connectionProfiles
    if (!profiles) return
    const ready = await profiles.connection()
    const headers = new Headers({ 'content-type': 'application/json' })
    if (ready.api_key) headers.set('authorization', `Bearer ${ready.api_key}`)
    const response = await fetch(`${ready.address}/api/v1/computer/desktop-authority/revoke-all`, {
      method: 'POST',
      headers,
      body: '{}',
      signal: AbortSignal.timeout(5_000)
    })
    if (!response.ok) {
      report(
        `[computer-safety] daemon revoke-all failed with status ${response.status}`,
        'warn'
      )
    }
  } catch (error) {
    report(
      `[computer-safety] daemon revoke-all failed: ${error instanceof Error ? error.message : String(error)}`,
      'warn'
    )
  }
}

function broadcastComputerEscape(reason?: string): void {
  if (!computerUseActive && !computerSafetyStarting) return
  if (reason) report(`[computer-safety] ${reason}`, 'warn')
  computerUseActive = false
  computerSafetyGeneration += 1
  computerSafetyStarting = null
  stopNativeComputerSafety()
  stopComputerOverlayWatchdog()
  computerSafetyOverlay?.hide()
  globalShortcut.unregister('Escape')
  // Main clears daemon authority directly. The renderer stop path is best-effort
  // UX; a crashed renderer must not leave desktop injection live for API clients.
  void revokeAllDesktopAuthority()
  for (const candidate of BrowserWindow.getAllWindows()) {
    if (candidate !== computerSafetyOverlay && !candidate.isDestroyed()) {
      candidate.webContents.send('brazier:computer:escape')
    }
  }
}

function nativeSafetyCommand(prepare: boolean): { command: string; args: string[] } {
  const executable = process.platform === 'win32' ? 'brazier-safety.exe' : 'brazier-safety'
  const installedSafety = join(__dirname, '../..', executable)
  const useInstalledSafety = process.env.BRAZIER_INSTALLED === '1' || existsSync(installedSafety)
  if (useInstalledSafety) {
    return { command: installedSafety, args: prepare ? ['--prepare'] : [] }
  }
  if (app.isPackaged) {
    return {
      command: join(process.resourcesPath, 'bin', executable),
      args: prepare ? ['--prepare'] : []
    }
  }
  return {
    command: 'cargo',
    args: ['run', '-q', '-p', 'brazier-safety', '--', ...(prepare ? ['--prepare'] : [])]
  }
}

function launchNativeComputerSafety(prepare: boolean): Promise<ChildProcessWithoutNullStreams> {
  const { command, args } = nativeSafetyCommand(prepare)
  const child = spawn(command, args, {
    cwd: installedApp ? undefined : repositoryRoot(),
    env: {
      ...process.env,
      ...(COMPUTER_WAYLAND_DISPLAY ? { WAYLAND_DISPLAY: COMPUTER_WAYLAND_DISPLAY } : {}),
      RUSTUP_TOOLCHAIN: process.env.RUSTUP_TOOLCHAIN ?? 'stable'
    },
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true
  })
  let errorOutput = ''
  child.stderr.on('data', (chunk) => {
    errorOutput = `${errorOutput}${chunk.toString()}`.slice(-4096)
    const line = `[computer-safety] ${chunk}`
    process.stderr.write(line)
    appendLog(line)
  })

  return new Promise((resolveSafety, rejectSafety) => {
    let ready = false
    let buffer = ''
    const diagnostic = (): string => {
      const detail = errorOutput.trim()
      return detail ? ` ${detail}` : ''
    }
    const timeout = setTimeout(() => {
      if (ready) return
      child.kill('SIGTERM')
      rejectSafety(new Error(
        `Timed out establishing the always-visible Computer Use safety overlay and Esc emergency stop.${diagnostic()}`
      ))
    }, 90_000)
    const failBeforeReady = (message: string): void => {
      if (ready) return
      clearTimeout(timeout)
      rejectSafety(new Error(message))
    }
    child.once('error', (error) => {
      failBeforeReady(`Could not start the Computer Use safety helper: ${error.message}`)
    })
    child.on('exit', (code, signal) => {
      const status = signal ? `signal ${signal}` : `status ${code ?? 'unknown'}`
      if (!ready) {
        failBeforeReady(`Computer Use safety helper exited before it was ready (${status}).${diagnostic()}`)
      } else if (!prepare && nativeComputerSafety === child && computerUseActive) {
        broadcastComputerEscape(`Safety helper exited unexpectedly (${status}); computer use was stopped.`)
      }
    })
    child.stdout.on('data', (chunk) => {
      buffer += chunk.toString()
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (line === 'READY' && !ready) {
          ready = true
          clearTimeout(timeout)
          resolveSafety(child)
        } else if (line === 'ESC' && !prepare) {
          broadcastComputerEscape('Esc emergency stop pressed.')
        }
      }
    })
  })
}

async function prepareComputerSafety(): Promise<void> {
  if (process.platform !== 'linux') return
  const child = await launchNativeComputerSafety(true)
  // --prepare exits as soon as the compositor has stored the shortcut. Close
  // its input as a second guarantee that no preparation process can linger.
  child.stdin.end()
}

/** Establish the security indicator and Escape hatch before desktop authority
 * is granted. Linux uses a separate native process so compositor or renderer
 * failure revokes control instead of leaving an invisible session running. */
async function setComputerUseActive(active: boolean): Promise<void> {
  if (!active) {
    computerUseActive = false
    computerSafetyGeneration += 1
    computerSafetyStarting = null
    stopNativeComputerSafety()
    stopComputerOverlayWatchdog()
    computerSafetyOverlay?.hide()
    globalShortcut.unregister('Escape')
    await revokeAllDesktopAuthority()
    return
  }
  if (computerUseActive) return
  if (computerSafetyStarting) return computerSafetyStarting

  const generation = ++computerSafetyGeneration
  if (process.platform === 'linux') {
    computerSafetyStarting = (async () => {
      const child = await launchNativeComputerSafety(false)
      if (generation !== computerSafetyGeneration) {
        child.kill('SIGTERM')
        throw new Error('Computer Use safety startup was cancelled.')
      }
      nativeComputerSafety = child
      writeSafetyOverlayMarker()
      computerUseActive = true
    })()
    try {
      await computerSafetyStarting
    } finally {
      if (generation === computerSafetyGeneration) computerSafetyStarting = null
    }
    return
  }

  const shortcutRegistered = globalShortcut.register('Escape', () => {
    broadcastComputerEscape('Esc emergency stop pressed.')
  })
  if (!shortcutRegistered) {
    throw new Error('Could not reserve Esc as the Computer Use emergency stop; desktop control was not started.')
  }
  if (!computerSafetyOverlay || computerSafetyOverlay.isDestroyed()) {
    computerSafetyOverlay = new BrowserWindow({
      width: 286, height: 48, frame: false, transparent: true,
      resizable: false, movable: false, focusable: false, skipTaskbar: true,
      alwaysOnTop: true,
      webPreferences: { sandbox: true, nodeIntegration: false, contextIsolation: true }
    })
    computerSafetyOverlay.setAlwaysOnTop(true, 'screen-saver')
    computerSafetyOverlay.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
    computerSafetyOverlay.setIgnoreMouseEvents(true, { forward: true })
    computerSafetyOverlay.on('ready-to-show', () => {
      positionComputerSafetyOverlay()
      computerSafetyOverlay?.setAlwaysOnTop(true, 'screen-saver')
      computerSafetyOverlay?.moveTop()
    })
    await computerSafetyOverlay.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(`<!doctype html><style>body{margin:0;background:transparent;font:600 14px -apple-system,BlinkMacSystemFont,sans-serif;color:#fff}div{height:48px;box-sizing:border-box;display:flex;align-items:center;justify-content:center;gap:10px;border:1px solid #ff8c69;border-radius:12px;background:#31140eeF;box-shadow:0 8px 30px #0008}b{color:#ffad91}kbd{padding:3px 7px;border:1px solid #ffffff55;border-radius:5px;background:#0008;font:600 12px inherit}</style><div><b>Computer Use active</b><span><kbd>Esc</kbd> to stop</span></div>`))
  }
  positionComputerSafetyOverlay()
  computerSafetyOverlay.setAlwaysOnTop(true, 'screen-saver')
  computerSafetyOverlay.showInactive()
  computerSafetyOverlay.moveTop()
  computerOverlayWatchdog = setInterval(() => {
    if (!computerUseActive || !computerSafetyOverlay || computerSafetyOverlay.isDestroyed()) return
    positionComputerSafetyOverlay()
    computerSafetyOverlay.setAlwaysOnTop(true, 'screen-saver')
    computerSafetyOverlay.showInactive()
    computerSafetyOverlay.moveTop()
  }, 250)
  writeSafetyOverlayMarker()
  computerUseActive = true
}

export type ServerKey = {
  id: string
  name: string
  createdAt: number
}

export type ServerSettings = {
  enabled: boolean
  port: number
  apiKeyEnabled: boolean
  hasApiKeys: boolean
  localhostOnly: boolean
  allowInsecureRemote: boolean
  jitLoading: boolean
  keys: ServerKey[]
}

type StoredApiKey = ServerKey & { value: string }
type PersistedApiKey = ServerKey & { encryptedValue: string }

type StoredServerSettings = Omit<ServerSettings, 'hasApiKeys' | 'keys'> & {
  apiKeys: StoredApiKey[]
}

type PersistedServerSettings = Omit<StoredServerSettings, 'apiKeys'> & {
  apiKeys: Array<PersistedApiKey | StoredApiKey>
}

const DEFAULT_SERVER_SETTINGS: StoredServerSettings = {
  enabled: false,
  port: 7614,
  apiKeyEnabled: true,
  apiKeys: [],
  localhostOnly: true,
  allowInsecureRemote: false,
  jitLoading: true
}

function serverSettingsPath(): string {
  return join(app.getPath('userData'), 'server-settings.json')
}

function credentialStorageAvailable(): boolean {
  if (!safeStorage.isEncryptionAvailable()) return false
  // Electron's Linux `basic_text` backend is reversible obfuscation, not an
  // operating-system credential store. Treat it as unavailable so bearer
  // credentials are never silently persisted in recoverable plaintext.
  return process.platform !== 'linux' || safeStorage.getSelectedStorageBackend() !== 'basic_text'
}

function validApiKeys(value: unknown): { keys: StoredApiKey[]; hadPlaintext: boolean } {
  if (!Array.isArray(value)) return { keys: [], hadPlaintext: false }
  const now = Date.now()
  let hadPlaintext = false
  const keys = value.flatMap((entry, index) => {
    if (typeof entry !== 'object' || entry === null) return []
    const candidate = entry as Record<string, unknown>
    let value_ = ''
    if (typeof candidate.encryptedValue === 'string' && candidate.encryptedValue) {
      if (!credentialStorageAvailable()) {
        throw new Error('Secure storage is unavailable for server API keys.')
      }
      value_ = safeStorage
        .decryptString(Buffer.from(candidate.encryptedValue, 'base64'))
        .trim()
    } else if (typeof candidate.value === 'string' && candidate.value.trim()) {
      value_ = candidate.value.trim()
      hadPlaintext = true
    }
    if (!value_) return []
    const name = typeof candidate.name === 'string' && candidate.name.trim() ? candidate.name.trim() : `Key ${index + 1}`
    const createdAt = typeof candidate.createdAt === 'number' && Number.isFinite(candidate.createdAt) ? candidate.createdAt : now
    const id = typeof candidate.id === 'string' && candidate.id ? candidate.id : randomUUID()
    return [{ id, name, value: value_, createdAt }]
  })
  return { keys, hadPlaintext }
}

function loadServerSettings(): StoredServerSettings {
  try {
    const parsed = JSON.parse(readFileSync(serverSettingsPath(), 'utf8')) as Partial<PersistedServerSettings>
    const port = Number(parsed.port)
    const loadedKeys = validApiKeys(parsed.apiKeys)
    const allowInsecureRemote = parsed.allowInsecureRemote === true
    const settings: StoredServerSettings = {
      enabled: parsed.enabled === true,
      port: Number.isInteger(port) && port >= 1 && port <= 65535 ? port : DEFAULT_SERVER_SETTINGS.port,
      apiKeyEnabled: parsed.apiKeyEnabled !== false,
      apiKeys: loadedKeys.keys,
      // Older builds defaulted this to every interface without a meaningful
      // acknowledgement. Migrate those settings back to loopback.
      localhostOnly: allowInsecureRemote ? parsed.localhostOnly !== false : true,
      allowInsecureRemote,
      jitLoading: parsed.jitLoading !== false
    }
    if (loadedKeys.hadPlaintext) saveServerSettings(settings)
    return settings
  } catch (cause) {
    if ((cause as NodeJS.ErrnoException).code !== 'ENOENT') {
      report(
        `[server] Saved server settings could not be opened; server mode is disabled: ${
          cause instanceof Error ? cause.message : String(cause)
        }`,
        'error'
      )
    }
    return { ...DEFAULT_SERVER_SETTINGS, apiKeys: [] }
  }
}

function publicServerSettings(settings: StoredServerSettings): ServerSettings {
  const { apiKeys, ...publicSettings } = settings
  return {
    ...publicSettings,
    hasApiKeys: apiKeys.length > 0,
    keys: apiKeys.map(({ id, name, createdAt }) => ({ id, name, createdAt }))
  }
}

function saveServerSettings(settings: StoredServerSettings): void {
  if (settings.apiKeys.length > 0 && !credentialStorageAvailable()) {
    throw new Error('Secure credential storage is unavailable; server API keys were not saved.')
  }
  const path = serverSettingsPath()
  mkdirSync(app.getPath('userData'), { recursive: true })
  const temporary = `${path}.${randomBytes(6).toString('hex')}.tmp`
  const persisted: PersistedServerSettings = {
    ...settings,
    apiKeys: settings.apiKeys.map(({ value, ...key }) => ({
      ...key,
      encryptedValue: safeStorage.encryptString(value).toString('base64')
    }))
  }
  writeFileSync(temporary, `${JSON.stringify(persisted, null, 2)}\n`, { mode: 0o600 })
  renameSync(temporary, path)
}

function generatedApiKey(): string {
  return `brazier_${randomBytes(32).toString('base64url')}`
}

let daemon: ChildProcessWithoutNullStreams | undefined
let connectionProfiles: ConnectionProfileManager | undefined
const agent = new AgentSupervisor()
let checkForUpdates: (() => Promise<UpdateCheckResult>) | undefined

function repositoryRoot(): string {
  const candidates = [
    resolve(process.cwd()),
    resolve(app.getAppPath(), '../..'),
    resolve(__dirname, '../../../..')
  ]
  return candidates.find((candidate) => existsSync(join(candidate, 'Cargo.toml'))) ?? process.cwd()
}

/**
 * Where brazierd keeps models, engines, downloads, and its database.
 *
 * This mirrors `default_data_dir` in crates/brazierd/src/main.rs so both entry
 * points agree. Electron's `userData` is deliberately not used: on Linux it
 * resolves under XDG_CONFIG_HOME, and this directory is data — it reaches tens
 * of gigabytes once models are downloaded, which does not belong in a config
 * directory. Honouring BRAZIER_DATA_DIR here also lets the desktop app run
 * against a throwaway profile; passing --data-dir unconditionally used to mask
 * the daemon's own reading of that variable.
 */
function dataDirectory(): string {
  const override = process.env.BRAZIER_DATA_DIR
  if (override) {
    return override
  }
  if (process.platform === 'win32') {
    const localAppData = process.env.LOCALAPPDATA
    return localAppData ? join(localAppData, 'Brazier') : app.getPath('userData')
  }
  if (process.platform === 'darwin') {
    return join(homedir(), 'Library', 'Application Support', 'Brazier')
  }
  const xdgDataHome = process.env.XDG_DATA_HOME
  return xdgDataHome ? join(xdgDataHome, 'brazier') : join(homedir(), '.local', 'share', 'brazier')
}

/**
 * Append a line to the session log.
 *
 * Diagnosing anything in the terminal meant quitting the app to copy what it
 * had printed — which killed whatever was still in flight, so the interesting
 * part was never in the paste. The same lines go to a file that can be read
 * while the app keeps running.
 */
function appendLog(line: string): void {
  try {
    logStream ??= createWriteStream(logPath(), { flags: 'a' })
    logStream.write(`${new Date().toISOString()} ${line.replace(/\s+$/, '')}\n`)
  } catch {
    // Logging must never be the thing that stops the app starting.
  }
}

function logPath(): string {
  const directory = join(dataDirectory(), 'logs')
  mkdirSync(directory, { recursive: true })
  return join(directory, 'brazier.log')
}

/** Print to the terminal and record it, so both places have the whole story. */
function report(line: string, level: 'log' | 'warn' | 'error' = 'log'): void {
  if (level === 'error') console.error(line)
  else if (level === 'warn') console.warn(line)
  else console.log(line)
  appendLog(line)
}

// Chromium and Node failures otherwise disappear into a terminal that is
// usually closed for packaged apps. Keep them beside daemon and updater events
// in the session log, without changing the application's recovery behaviour.
process.on('uncaughtException', (error) => {
  report(`[main] uncaught exception: ${error.stack ?? error.message}`, 'error')
})
process.on('unhandledRejection', (reason) => {
  report(`[main] unhandled rejection: ${reason instanceof Error ? reason.stack ?? reason.message : String(reason)}`, 'error')
})

function startDaemon(): Promise<Connection> {
  const directory = dataDirectory()
  const installedDaemon = join(__dirname, '../..', process.platform === 'win32' ? 'brazierd.exe' : 'brazierd')
  const useInstalledDaemon = process.env.BRAZIER_INSTALLED === '1' || existsSync(installedDaemon)
  const command = useInstalledDaemon
    ? installedDaemon
    : app.isPackaged
      ? join(process.resourcesPath, 'bin', process.platform === 'win32' ? 'brazierd.exe' : 'brazierd')
      : 'cargo'
  const args = (installedApp || useInstalledDaemon)
    ? ['--data-dir', directory]
    : ['run', '-q', '-p', 'brazierd', '--', '--data-dir', directory]
  const serverSettings = loadServerSettings()
  let daemonKeysForStdin: string[] = []
  if (serverSettings.enabled) {
    if (!serverSettings.localhostOnly && !serverSettings.apiKeyEnabled) {
      throw new Error('A non-loopback server must require API-key authentication.')
    }
    if (!serverSettings.localhostOnly && !serverSettings.allowInsecureRemote) {
      throw new Error('Non-loopback plaintext access has not been explicitly acknowledged.')
    }
    args.push('--host', serverSettings.localhostOnly ? '127.0.0.1' : '0.0.0.0', '--port', String(serverSettings.port))
    if (serverSettings.apiKeyEnabled) {
      // The desktop's own connection needs a credential too. If the user has
      // not named any keys yet, mint one and keep it stable across restarts so
      // a configured OpenAI client is not silently revoked.
      let apiKeys = serverSettings.apiKeys
      if (apiKeys.length === 0) {
        const value = generatedApiKey()
        apiKeys = [{ id: randomUUID(), name: 'Auto-generated', value, createdAt: Date.now() }]
        serverSettings.apiKeys = apiKeys
        saveServerSettings(serverSettings)
      }
      daemonKeysForStdin = apiKeys.map((key) => key.value)
      args.push('--api-keys-stdin')
    } else {
      args.push('--no-auth')
    }
    // Authentication prevents unauthorized use but does not encrypt bearer
    // credentials. The daemon requires this explicit acknowledgement for any
    // plaintext non-loopback listener, with or without auth.
    if (!serverSettings.localhostOnly && serverSettings.allowInsecureRemote) {
      args.push('--allow-insecure-remote')
    }
    args.push('--jit-loading', String(serverSettings.jitLoading))
  }
  const child = spawn(command, args, {
    cwd: installedApp || useInstalledDaemon ? undefined : repositoryRoot(),
    env: {
      ...process.env,
      ...(COMPUTER_WAYLAND_DISPLAY ? { WAYLAND_DISPLAY: COMPUTER_WAYLAND_DISPLAY } : {}),
      RUST_LOG: process.env.RUST_LOG ?? 'brazierd=info',
      RUSTUP_TOOLCHAIN: process.env.RUSTUP_TOOLCHAIN ?? 'stable'
    },
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true
  })
  daemon = child
  child.once('exit', () => {
    if (daemon === child) daemon = undefined
  })
  child.stdin.end(
    daemonKeysForStdin.length > 0 ? `${daemonKeysForStdin.join('\n')}\n` : undefined
  )
  child.stderr.on('data', (chunk) => {
    const text = `[brazierd] ${chunk}`
    process.stderr.write(text)
    appendLog(text)
  })

  return new Promise((resolveConnection, reject) => {
    let buffer = ''
    const timeout = setTimeout(
      () => reject(new Error('The Brazier daemon did not become ready in time.')),
      30_000
    )
    child.once('error', (error) => {
      clearTimeout(timeout)
      reject(error)
    })
    child.once('exit', (code) => {
      if (code && code !== 0) {
        clearTimeout(timeout)
        reject(new Error(`The Brazier daemon exited with status ${code}.`))
      }
    })
    child.stdout.on('data', (chunk) => {
      buffer += chunk.toString()
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('BRAZIER_READY ')) continue
        clearTimeout(timeout)
        resolveConnection(JSON.parse(line.slice('BRAZIER_READY '.length)) as Connection)
      }
    })
  })
}

function stopLocalDaemon(): void {
  daemon?.kill()
  daemon = undefined
}

/** Stop the package-smoke daemon and prove that the child actually exited. */
function stopLocalDaemonAndWait(timeoutMs = 10_000): Promise<void> {
  const child = daemon
  if (!child) return Promise.reject(new Error('the local daemon was not running at shutdown'))
  daemon = undefined
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve()

  return new Promise((resolveShutdown, rejectShutdown) => {
    const cleanup = (): void => {
      clearTimeout(timeout)
      child.off('exit', onExit)
      child.off('error', onError)
    }
    const onExit = (): void => {
      cleanup()
      resolveShutdown()
    }
    const onError = (cause: Error): void => {
      cleanup()
      rejectShutdown(cause)
    }
    const timeout = setTimeout(() => {
      cleanup()
      child.kill('SIGKILL')
      rejectShutdown(new Error(`the local daemon did not exit within ${timeoutMs} ms`))
    }, timeoutMs)
    child.once('exit', onExit)
    child.once('error', onError)
    if (!child.kill('SIGTERM')) {
      onError(new Error('the local daemon did not accept the shutdown signal'))
    }
  })
}

/** Confirm that the installed artifact contains its Computer Use safety helper. */
async function checkPackageSafetyHelper(): Promise<void> {
  const { command } = nativeSafetyCommand(false)
  const mode = process.platform === 'win32' ? constants.F_OK : constants.X_OK
  try {
    accessSync(command, mode)
  } catch {
    throw new Error(`the packaged Computer Use safety helper is missing or not executable: ${command}`)
  }
}

function connectionProfilesPath(): string {
  return join(app.getPath('userData'), 'connection-profiles.json')
}

function connectionCredentialCodec(): ConnectionCredentialCodec | undefined {
  if (!credentialStorageAvailable()) return undefined
  return {
    encrypt: (plaintext) => safeStorage.encryptString(plaintext).toString('base64'),
    decrypt: (ciphertext) => safeStorage.decryptString(Buffer.from(ciphertext, 'base64'))
  }
}

function openConnectionProfileStore(): ConnectionProfileStore {
  if (!credentialStorageAvailable()) {
    report('[connections] Secure credential storage is locked; using Local recovery mode.', 'error')
    return ConnectionProfileStore.localOnly(
      join(app.getPath('userData'), 'connection-profiles-recovery.json'),
      randomUUID
    )
  }
  const codec = connectionCredentialCodec()
  try {
    return new ConnectionProfileStore(connectionProfilesPath(), randomUUID, codec)
  } catch {
    // Leave the primary file untouched for recovery. An empty secondary store
    // keeps Local usable without silently accepting an undecryptable key.
    report('[connections] Saved remote credentials could not be opened; using Local recovery mode.', 'error')
    dialog.showErrorBox(
      'Remote credentials unavailable',
      'Brazier could not decrypt one or more saved remote API keys. The original profiles were left untouched. Local remains available; unlock your system credential store and restart, or pair the remote connection again.'
    )
    return ConnectionProfileStore.localOnly(
      join(app.getPath('userData'), 'connection-profiles-recovery.json'),
      randomUUID,
      codec
    )
  }
}

function rendererDevelopmentOrigin(): string | undefined {
  const value = process.env.ELECTRON_RENDERER_URL
  if (!value) return undefined
  try {
    return new URL(value).origin
  } catch {
    return undefined
  }
}

type QualificationHost = {
  commit: string
  platform: 'macos' | 'linux' | 'windows'
  arch: string
  memory_gib: number
  gpu_vram_gib: number | null
  gpu_vendor: string | null
}

async function qualificationHost(): Promise<QualificationHost> {
  const platform = process.platform === 'darwin'
    ? 'macos'
    : process.platform === 'win32'
      ? 'windows'
      : 'linux'
  let gpuVramGib: number | null = null
  let gpuVendor: string | null = null
  try {
    const info = await app.getGPUInfo('complete') as unknown as Record<string, unknown>
    const attributes = typeof info.auxAttributes === 'object' && info.auxAttributes !== null
      ? info.auxAttributes as Record<string, unknown>
      : {}
    const devices = Array.isArray(info.gpuDevice) ? info.gpuDevice as Array<Record<string, unknown>> : []
    const raw = attributes.videoMemory ?? attributes.video_memory ?? devices[0]?.videoMemory
    const megabytes = typeof raw === 'number' ? raw : Number(raw)
    if (Number.isFinite(megabytes) && megabytes > 0) gpuVramGib = megabytes / 1024
    const vendor = attributes.glVendor ?? attributes.driver_vendor ?? devices[0]?.vendorId
    const vendorText = String(vendor ?? '').toLowerCase()
    gpuVendor = vendorText.includes('nvidia') || vendorText === '4318' || vendorText === '0x10de'
      ? 'nvidia'
      : vendorText.includes('amd') || vendorText === '4098' || vendorText === '0x1002'
        ? 'amd'
        : vendorText.includes('intel') || vendorText === '32902' || vendorText === '0x8086'
          ? 'intel'
          : vendorText.includes('apple') || vendorText === '4203' || vendorText === '0x106b'
            ? 'apple'
            : vendorText || null
  } catch {
    // Some software renderers and CI hosts do not expose dedicated VRAM.
  }
  // Linux normally runs Electron with GPU acceleration disabled for renderer
  // compatibility. In that mode Chromium reports its software renderer rather
  // than the physical CUDA device, so query the driver's stable CLI directly.
  if (process.platform === 'linux' && (gpuVendor !== 'nvidia' || gpuVramGib === null)) {
    const candidates = ['/usr/bin/nvidia-smi', '/usr/local/bin/nvidia-smi', 'nvidia-smi']
    for (const command of candidates) {
      if (command.startsWith('/') && !existsSync(command)) continue
      try {
        const result = await runChild(
          command,
          ['--query-gpu=memory.total', '--format=csv,noheader,nounits'],
          5_000
        )
        const detected = result.code === 0 ? parseNvidiaSmiMemoryGib(result.stdout) : null
        if (detected !== null) {
          gpuVendor = 'nvidia'
          gpuVramGib = detected
          break
        }
      } catch {
        // Try the next conventional location; the verifier still fails closed.
      }
    }
  }
  return {
    commit:
      process.env.BRAZIER_BUILD_COMMIT?.trim() ||
      process.env.GITHUB_SHA?.trim() ||
      __BRAZIER_BUILD_COMMIT__ ||
      `version:${app.getVersion()}`,
    platform,
    arch: process.arch,
    memory_gib: totalmem() / 1024 ** 3,
    gpu_vram_gib: gpuVramGib,
    gpu_vendor: gpuVendor
  }
}

/**
 * The renderer talks directly to its selected daemon so streaming responses
 * retain browser backpressure and cancellation. The document CSP permits the
 * relevant schemes; this main-process boundary narrows them to the active
 * profile (plus Vite's own origin in development).
 */
function installRendererConnectionGuard(profiles: ConnectionProfileManager): void {
  const developmentOrigin = rendererDevelopmentOrigin()
  const filter = { urls: ['http://*/*', 'https://*/*', 'ws://*/*', 'wss://*/*'] }
  session.defaultSession.webRequest.onBeforeRequest(
    filter,
    (details, callback) => {
      // Updater and other main-process requests have no renderer owner and do
      // not carry profile credentials. This guard is specifically the
      // renderer's direct daemon boundary.
      if (!details.webContentsId) {
        callback({ cancel: false })
        return
      }
      callback({ cancel: !profiles.allowsRendererNetworkUrl(details.url, developmentOrigin) })
    }
  )
  session.defaultSession.webRequest.onBeforeSendHeaders(filter, (details, callback) => {
    const requestHeaders = { ...details.requestHeaders }
    for (const name of Object.keys(requestHeaders)) {
      if (
        name.toLowerCase() === 'authorization' ||
        name.toLowerCase() === 'x-brazier-local-control'
      ) {
        delete requestHeaders[name]
      }
    }
    if (!details.webContentsId) {
      callback({ requestHeaders })
      return
    }
    void Promise.all([
      profiles.rendererApiKeyForUrl(details.url),
      profiles.rendererLocalControlKeyForUrl(details.url)
    ])
      .then(([apiKey, localControlKey]) => {
        if (apiKey) requestHeaders.Authorization = `Bearer ${apiKey}`
        if (localControlKey) requestHeaders['X-Brazier-Local-Control'] = localControlKey
        callback({ requestHeaders })
      })
      .catch(() => callback({ cancel: true, requestHeaders }))
  })
}

function broadcastConnectionChanged(summary: ConnectionProfileSummary): void {
  for (const window of BrowserWindow.getAllWindows()) {
    if (!window.isDestroyed()) window.webContents.send('brazier:connection-changed', summary)
  }
}

async function resetForConnectionSwitch(ready: DaemonConnection): Promise<void> {
  // An agent worker is initialized with one immutable daemon credential. Stop
  // it before publishing the new selection so no request can cross profiles.
  await agent.shutdown()
  broadcastConnectionChanged(ready.profile)
  // The event invalidates small API caches immediately. Reloading is the reset
  // boundary for long-lived streams, voice graphs, and daemon-owned ids held by
  // mounted React components. Delay slightly so the IPC reply can be delivered.
  setTimeout(() => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.reload()
    }
  }, 50)
}

function attachContextMenu(window: BrowserWindow): void {
  window.webContents.on('context-menu', (_event, params) => {
    const template: Electron.MenuItemConstructorOptions[] = []

    if (params.isEditable) {
      template.push(
        { role: 'cut', enabled: params.editFlags.canCut },
        { role: 'copy', enabled: params.editFlags.canCopy },
        { role: 'paste', enabled: params.editFlags.canPaste },
        { type: 'separator' },
        { role: 'selectAll', enabled: params.editFlags.canSelectAll }
      )
    } else {
      template.push(
        {
          role: 'copy',
          enabled: params.selectionText.length > 0 || params.editFlags.canCopy
        },
        { role: 'selectAll' }
      )
    }

    if (params.linkURL && isSafeExternalUrl(params.linkURL)) {
      template.push({ type: 'separator' })
      template.push({
        label: 'Open Link',
        click: () => {
          void shell.openExternal(params.linkURL)
        }
      })
    }

    Menu.buildFromTemplate(template).popup({ window })
  })
}

/**
 * Application icon for the window, taskbar, and dock.
 *
 * electron-builder bakes these into the packaged app, but a running window
 * needs a real file: on Linux and Windows the taskbar entry otherwise shows
 * Electron's default logo, including in development.
 *
 * macOS gets the squircle cut, since the Dock draws every icon inside that
 * shape and a full-bleed square reads as oversized beside its neighbours.
 * Windows and Linux use the square artwork, which is their convention.
 */
function iconPath(): string | undefined {
  const file = process.platform === 'darwin' ? 'icon-mac.png' : 'icon.png'
  const installedIcon = join(__dirname, '../..', file)
  const candidates = process.env.BRAZIER_INSTALLED === '1' || existsSync(installedIcon)
    ? [installedIcon]
    : app.isPackaged
      ? [join(process.resourcesPath, file), join(process.resourcesPath, 'icon.png')]
      : [join(repositoryRoot(), 'apps', 'desktop', 'build', file)]
  return candidates.find((candidate) => existsSync(candidate))
}

async function createWindow(): Promise<void> {
  const icon = iconPath()
  const window = new BrowserWindow({
    width: 1280,
    height: 820,
    // Small enough for a half-screen snap on a 13" display; the renderer's
    // narrow-width breakpoints keep the chrome usable down to this size.
    minWidth: 560,
    minHeight: 540,
    show: false,
    backgroundColor: '#000000',
    autoHideMenuBar: true,
    ...(icon ? { icon } : {}),
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    ...(process.platform === 'darwin'
      ? { trafficLightPosition: { x: 15, y: 20 } }
      : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // Avoid offscreen compositing paths that need shared memory.
      offscreen: false,
      backgroundThrottling: false
    }
  })

  attachContextMenu(window)

  window.once('ready-to-show', () => {
    window.show()
    window.focus()
  })
  setTimeout(() => {
    if (!window.isDestroyed() && !window.isVisible()) {
      window.show()
      window.focus()
    }
  }, 1_500)

  window.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternalUrl(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  const guardMainFrameNavigation = (event: Electron.Event, url: string): void => {
    if (!isTrustedRendererUrl(url, rendererTrustOptions())) event.preventDefault()
  }
  window.webContents.on('will-navigate', guardMainFrameNavigation)
  window.webContents.on('will-redirect', guardMainFrameNavigation)
  // The app deliberately has no menu bar, and the menu bar is what normally
  // carries reload and developer tools. Without them the only way to pick up a
  // renderer change is quitting: Vite's module updates reach the page, but
  // long-lived objects built once per mount — the session coordinator, an open
  // audio graph — keep running the code they were constructed with.
  if (!app.isPackaged) {
    window.webContents.on('before-input-event', (event, input) => {
      if (input.type !== 'keyDown') return
      const accelerator = process.platform === 'darwin' ? input.meta : input.control
      if (!accelerator) return
      // Matched on physical key: Option+I on macOS produces a dead key rather
      // than the letter.
      if (input.code === 'KeyR') {
        window.webContents.reload()
        event.preventDefault()
        return
      }
      if (input.alt && input.code === 'KeyI') {
        window.webContents.toggleDevTools()
        event.preventDefault()
      }
    })
  }

  // Renderer logs go to the developer tools console, which is a different place
  // from the terminal `pnpm dev` prints to — so "the console is silent" can mean
  // either. Development builds forward them, making the terminal the one place
  // to look.
  if (!app.isPackaged) {
    window.webContents.on('console-message', (event) => {
      const level = event.level === 'error' ? 'error' : event.level === 'warning' ? 'warn' : 'log'
      report(`[renderer] ${event.message}`, level)
    })
  }

  window.webContents.on('did-fail-load', (_event, code, description, validatedURL) => {
    report(`[brazier] renderer failed to load (${code}): ${description} @ ${validatedURL}`, 'error')
    if (!window.isDestroyed() && !window.isVisible()) window.show()
  })
  window.webContents.on('did-finish-load', () => {
    // Says where the log is, so reading it does not require quitting first.
    report(`[brazier] renderer finished load — session log: ${logPath()}`)
    if (!window.isDestroyed() && !window.isVisible()) window.show()
  })
  window.webContents.on('render-process-gone', (_event, details) => {
    console.error('[brazier] renderer process gone', details)
  })
  window.webContents.on('console-message', (event) => {
    // Electron ≥39: prefer the event object form.
    const level = 'level' in event ? Number(event.level) : 0
    const message = 'message' in event ? String(event.message) : ''
    const line = 'line' in event ? Number(event.line) : 0
    const sourceId = 'sourceId' in event ? String(event.sourceId) : ''
    if (level >= 2) {
      report(`[renderer:${level}] ${message} (${sourceId}:${line})`, 'error')
    }
  })

  try {
    if (process.env.ELECTRON_RENDERER_URL) {
      await window.loadURL(process.env.ELECTRON_RENDERER_URL)
    } else {
      await window.loadFile(rendererIndexPath())
    }
  } catch (error) {
    report(`[brazier] failed to load renderer: ${error instanceof Error ? error.stack ?? error.message : String(error)}`, 'error')
    if (!window.isDestroyed() && !window.isVisible()) window.show()
  }

  // DevTools off by default (detach windows can confuse focus); opt-in.
  if (process.env.BRAZIER_DEVTOOLS === '1') {
    window.webContents.openDevTools({ mode: 'detach' })
  }
}

function forceWelcomeRequested(): boolean {
  if (process.env.BRAZIER_FORCE_WELCOME === '1') return true
  return process.argv.some((arg) => arg === '--welcome' || arg === '--force-welcome')
}

function packageSmokeTarget(): Pick<PackageSmokeResult, 'platform' | 'artifact'> {
  if (process.platform === 'darwin') return { platform: 'macos', artifact: 'dmg' }
  if (process.platform === 'win32') return { platform: 'windows', artifact: 'nsis' }
  return { platform: 'linux', artifact: 'appimage' }
}

async function runRequestedPackageSmoke(profiles: ConnectionProfileManager): Promise<void> {
  const ready = await profiles.select('local')
  agent.setConnection({ address: ready.address, apiKey: ready.api_key })
  const target = packageSmokeTarget()
  const result = await runPackageSmoke({
    connection: { address: ready.address, apiKey: ready.api_key },
    warmWorker: () => agent.warmup(),
    openSession: async (sessionId) => {
      await agent.invoke({ type: 'open-session', sessionId })
    },
    shutdownWorker: () => agent.shutdown(),
    shutdownDaemon: () => stopLocalDaemonAndWait(),
    checkSafetyHelper: () => checkPackageSafetyHelper(),
    commit:
      process.env.BRAZIER_BUILD_COMMIT?.trim() ||
      process.env.GITHUB_SHA?.trim() ||
      __BRAZIER_BUILD_COMMIT__ ||
      'development',
    platform: target.platform,
    arch: process.arch,
    artifact: target.artifact
  })
  const serialized = `${JSON.stringify(result, null, 2)}\n`
  const output = process.env.BRAZIER_PACKAGE_SMOKE_OUTPUT
  if (output) await writeFile(output, serialized, { mode: 0o600 })
  else process.stdout.write(serialized)
  profiles.shutdown()
  app.exit(result.passed ? 0 : 1)
}

app.whenReady().then(async () => {
  // No File/Edit/View application menu — the app is a self-contained shell.
  Menu.setApplicationMenu(null)
  // A packaged macOS app takes its dock icon from the bundle; an unpackaged one
  // shows Electron's unless it is set here.
  if (process.platform === 'darwin' && !app.isPackaged) {
    const icon = iconPath()
    if (icon) app.dock?.setIcon(icon)
  }
  const profiles = new ConnectionProfileManager(
    openConnectionProfileStore(),
    { startLocal: startDaemon, stopLocal: stopLocalDaemon }
  )
  connectionProfiles = profiles
  installRendererConnectionGuard(profiles)

  if (process.env.BRAZIER_PACKAGE_SMOKE === '1') {
    try {
      await runRequestedPackageSmoke(profiles)
    } catch (cause) {
      report(
        `[package-smoke] ${cause instanceof Error ? cause.stack ?? cause.message : String(cause)}`,
        'error'
      )
      profiles.shutdown()
      app.exit(1)
    }
    return
  }

  const readyConnection = async (): Promise<DaemonConnection> => {
    const ready = await profiles.connection()
    agent.setConnection({ address: ready.address, apiKey: ready.api_key })
    return ready
  }
  const requireLocalFilesystem = (action: string): void => {
    const current = profiles.current()
    if (current.kind === 'remote') {
      throw new Error(
        `${action} is unavailable for ${current.name}; its filesystem is on ${current.hostLabel}. Enter a daemon-host path instead.`
      )
    }
  }

  handleTrusted('brazier:connection', async () => {
    const { api_key: _secret, ...connection } = await readyConnection()
    return connection
  })
  handleTrusted('brazier:connection-profiles:list', () => profiles.list())
  handleTrusted('brazier:connection-profiles:current', () => profiles.current())
  handleTrusted(
    'brazier:connection-profiles:upsert',
    async (_event, input: RemoteConnectionProfileInput) => {
      const activeId = profiles.current().id
      const profile = await profiles.upsert(input)
      if (profile.id === activeId) {
        const ready = await profiles.connection()
        await resetForConnectionSwitch(ready)
      }
      return viewConnectionProfile(profile)
    }
  )
  handleTrusted(
    'brazier:connection-profiles:test',
    async (_event, idOrProfile: string | RemoteConnectionProfileInput): Promise<ConnectionTestResult> => {
      const ready = await profiles.test(idOrProfile)
      return { profile: ready.profile, daemon: ready.daemon }
    }
  )
  handleTrusted(
    'brazier:connection-profiles:claim-pairing',
    async (_event, input: PairingClaimInput): Promise<ClaimedConnection> => {
      const activeId = profiles.current().id
      const claimed = await profiles.claimAndSave(input)
      if (claimed.profile.id === activeId) {
        await resetForConnectionSwitch(await profiles.connection())
      }
      return claimed
    }
  )
  handleTrusted('brazier:connection-profiles:select', async (_event, id: string) => {
    const previousId = profiles.current().id
    const ready = await profiles.select(id)
    if (ready.profile.id !== previousId) await resetForConnectionSwitch(ready)
    else agent.setConnection({ address: ready.address, apiKey: ready.api_key })
    return ready.profile
  })
  handleTrusted('brazier:connection-profiles:delete', async (_event, id: string) => {
    const wasActive = profiles.current().id === id
    const deleted = await profiles.delete(id)
    if (deleted && wasActive) await resetForConnectionSwitch(await profiles.connection())
    return deleted
  })
  handleTrusted('brazier:computer:set-active', (_event, active: boolean) =>
    setComputerUseActive(active === true))
  handleTrusted('brazier:computer:prepare-safety', () => prepareComputerSafety())
  handleTrusted('brazier:computer:input-guard-status', () => inputGuardStatus())
  handleTrusted('brazier:computer:setup-input-guard', () => setupInputGuard())
  handleTrusted('brazier:server-settings', (): ServerSettings => publicServerSettings(loadServerSettings()))
  handleTrusted(
    'brazier:save-server-settings',
    (_event, requested: Omit<ServerSettings, 'hasApiKeys' | 'keys'>) => {
      const port = Number(requested.port)
      if (!Number.isInteger(port) || port < 1 || port > 65535) {
        throw new Error('Server port must be between 1 and 65535.')
      }
      const enabled = requested.enabled === true
      const localhostOnly = requested.localhostOnly !== false
      const apiKeyEnabled = requested.apiKeyEnabled !== false
      const allowInsecureRemote = !localhostOnly && requested.allowInsecureRemote === true
      if (enabled && !localhostOnly && !apiKeyEnabled) {
        throw new Error('A non-loopback server must require an API key.')
      }
      if (enabled && !localhostOnly && !allowInsecureRemote) {
        throw new Error('Acknowledge plaintext network exposure before listening beyond localhost.')
      }
      const next: StoredServerSettings = {
        enabled,
        port,
        apiKeyEnabled,
        localhostOnly,
        allowInsecureRemote,
        jitLoading: requested.jitLoading !== false,
        apiKeys: loadServerSettings().apiKeys
      }
      saveServerSettings(next)
      return publicServerSettings(next)
    }
  )
  handleTrusted(
    'brazier:add-server-api-key',
    (_event, name: string): StoredApiKey => {
      const current = loadServerSettings()
      const label = (name ?? '').trim() || `Key ${current.apiKeys.length + 1}`
      const key: StoredApiKey = { id: randomUUID(), name: label, value: generatedApiKey(), createdAt: Date.now() }
      current.apiKeys = [...current.apiKeys, key]
      saveServerSettings(current)
      return { id: key.id, name: key.name, value: key.value, createdAt: key.createdAt }
    }
  )
  handleTrusted('brazier:remove-server-api-key', (_event, id: string): ServerSettings => {
    const current = loadServerSettings()
    current.apiKeys = current.apiKeys.filter((key) => key.id !== id)
    saveServerSettings(current)
    return publicServerSettings(current)
  })
  handleTrusted('brazier:copy-text', (_event, text: string) => {
    clipboard.writeText(String(text ?? ''))
  })
  handleTrusted('brazier:flags', () => ({
    forceWelcome: forceWelcomeRequested()
  }))
  handleTrusted('brazier:qualification-host', () => qualificationHost())
  handleTrusted('brazier:check-for-updates', async () => {
    if (!checkForUpdates) return { supported: false }
    return checkForUpdates()
  })
  handleTrusted('brazier:get-update-settings', () => getUpdateSettings())
  handleTrusted(
    'brazier:save-update-settings',
    (
      _event,
      settings: { checkOnStartup?: boolean; autoDownload?: boolean }
    ) => saveUpdateSettings(settings ?? {})
  )
  handleTrusted('brazier:select-directory', async (event) => {
    requireLocalFilesystem('The folder picker')
    const window = BrowserWindow.fromWebContents(event.sender)
    const options = {
      properties: ['openDirectory'] as ('openDirectory')[],
      title: 'Choose a model folder'
    }
    const result = window
      ? await dialog.showOpenDialog(window, options)
      : await dialog.showOpenDialog(options)
    if (result.canceled || result.filePaths.length === 0) {
      return null
    }
    return result.filePaths[0] ?? null
  })
  // Choosing a LoRA, a ControlNet, or a reference image already on disk.
  // Adapters are usually shared with another tool, so they are pointed at
  // where they live rather than copied into the application's own library.
  handleTrusted(
    'brazier:select-file',
    async (event, title: string, filters: Electron.FileFilter[]) => {
      requireLocalFilesystem('The file picker')
      const window = BrowserWindow.fromWebContents(event.sender)
      const options = {
        // A directory is a valid MLX adapter, so both are selectable and the
        // daemon decides which engines can read whichever was chosen.
        properties: ['openFile', 'openDirectory'] as ('openFile' | 'openDirectory')[],
        title,
        filters
      }
      const result = window
        ? await dialog.showOpenDialog(window, options)
        : await dialog.showOpenDialog(options)
      if (result.canceled || result.filePaths.length === 0) {
        return null
      }
      return result.filePaths[0] ?? null
    }
  )
  // Saving generated media. The renderer holds the bytes already, so this only
  // has to ask where they go — writing from the main process keeps the renderer
  // without filesystem access.
  handleTrusted(
    'brazier:save-file',
    async (event, suggestedName: string, data: ArrayBuffer): Promise<string | null> => {
      const window = BrowserWindow.fromWebContents(event.sender)
      const options = {
        title: 'Save generated media',
        defaultPath: join(app.getPath('downloads'), suggestedName)
      }
      const result = window
        ? await dialog.showSaveDialog(window, options)
        : await dialog.showSaveDialog(options)
      if (result.canceled || !result.filePath) return null
      await writeFile(result.filePath, Buffer.from(data))
      return result.filePath
    }
  )
  handleTrusted('brazier:reveal-file', (_event, path: string) => {
    requireLocalFilesystem('Reveal in file manager')
    shell.showItemInFolder(path)
  })
  // Agent mode reaches the machine only through the daemon, so the worker gets
  // the loopback address and bearer token once the daemon is ready.
  registerAgentIpc(agent, assertTrustedIpcSender)
  handleTrusted('brazier:agent:status', () => agent.status())
  handleTrusted('brazier:select-workspace', async (event) => {
    requireLocalFilesystem('The workspace picker')
    const window = BrowserWindow.fromWebContents(event.sender)
    const options = {
      properties: ['openDirectory' as const],
      title: 'Choose a workspace folder for the agent'
    }
    const result = window
      ? await dialog.showOpenDialog(window, options)
      : await dialog.showOpenDialog(options)
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0] ?? null
  })
  await createWindow()
  checkForUpdates = startUpdates(report).checkForUpdates
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) void createWindow()
  })
})

app.on('before-quit', () => {
  computerUseActive = false
  computerSafetyGeneration += 1
  stopNativeComputerSafety()
  stopComputerOverlayWatchdog()
  globalShortcut.unregister('Escape')
  clearSafetyOverlayMarker()
  void agent.shutdown()
  connectionProfiles?.shutdown()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
