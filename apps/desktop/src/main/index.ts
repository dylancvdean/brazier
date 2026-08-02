import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { createWriteStream, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync, type WriteStream } from 'node:fs'
import { randomBytes } from 'node:crypto'
import { writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import { app, BrowserWindow, dialog, globalShortcut, ipcMain, Menu, screen, shell } from 'electron'

import { AgentSupervisor, registerAgentIpc } from './agent'
import {
  getUpdateSettings,
  saveUpdateSettings,
  startUpdates,
  type UpdateCheckResult
} from './updates'

/**
 * Where Electron kept per-app state before the rename below.
 *
 * `app.setName` moves `userData`, so this is read first: an install that
 * predates the rename still has its models and conversations under the old
 * name, and `migrateLegacyDataDirectory` needs to be able to find them.
 */
const LEGACY_USER_DATA = app.getPath('userData')
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
 * Mitigations that work in practice:
 * - no-sandbox: node_modules chrome-sandbox is never setuid
 * - disable-dev-shm-usage: avoid /dev/shm
 * - force X11 (XWayland) instead of Ozone/Wayland color-management path
 * - software compositing + in-process GPU to avoid the crashing GPU process
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

  app.commandLine.appendSwitch('no-sandbox')
  app.commandLine.appendSwitch('no-zygote')
  app.commandLine.appendSwitch('disable-dev-shm-usage')
  app.commandLine.appendSwitch('disable-gpu-sandbox')
  app.commandLine.appendSwitch('ozone-platform-hint', process.env.ELECTRON_OZONE_PLATFORM_HINT)
  if (process.env.ELECTRON_OZONE_PLATFORM_HINT === 'x11') {
    app.commandLine.appendSwitch('ozone-platform', 'x11')
  }

  // Default to software GL on Linux. Override with BRAZIER_ELECTRON_SOFTWARE_GL=0
  // for hardware acceleration once the host paints correctly.
  if (process.env.BRAZIER_ELECTRON_SOFTWARE_GL !== '0') {
    app.disableHardwareAcceleration()
    app.commandLine.appendSwitch('disable-gpu')
    app.commandLine.appendSwitch('in-process-gpu')
  }
}

type Connection = {
  address: string
  api_key: string | null
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
  computerUseActive = true
}

export type ServerSettings = {
  enabled: boolean
  port: number
  apiKeyEnabled: boolean
  hasApiKey: boolean
  jitLoading: boolean
}

type StoredServerSettings = Omit<ServerSettings, 'hasApiKey'> & { apiKey: string | null }

const DEFAULT_SERVER_SETTINGS: StoredServerSettings = {
  enabled: false,
  port: 7614,
  apiKeyEnabled: true,
  apiKey: null,
  jitLoading: true
}

function serverSettingsPath(): string {
  return join(app.getPath('userData'), 'server-settings.json')
}

function loadServerSettings(): StoredServerSettings {
  try {
    const parsed = JSON.parse(readFileSync(serverSettingsPath(), 'utf8')) as Partial<StoredServerSettings>
    const port = Number(parsed.port)
    return {
      enabled: parsed.enabled === true,
      port: Number.isInteger(port) && port >= 1 && port <= 65535 ? port : DEFAULT_SERVER_SETTINGS.port,
      apiKeyEnabled: parsed.apiKeyEnabled !== false,
      apiKey: typeof parsed.apiKey === 'string' && parsed.apiKey.trim() ? parsed.apiKey : null,
      jitLoading: parsed.jitLoading !== false
    }
  } catch {
    return { ...DEFAULT_SERVER_SETTINGS }
  }
}

function publicServerSettings(settings: StoredServerSettings): ServerSettings {
  const { apiKey: _apiKey, ...publicSettings } = settings
  return { ...publicSettings, hasApiKey: Boolean(settings.apiKey) }
}

function saveServerSettings(settings: StoredServerSettings): void {
  const path = serverSettingsPath()
  mkdirSync(app.getPath('userData'), { recursive: true })
  const temporary = `${path}.${randomBytes(6).toString('hex')}.tmp`
  writeFileSync(temporary, `${JSON.stringify(settings, null, 2)}\n`, { mode: 0o600 })
  renameSync(temporary, path)
}

function generatedApiKey(): string {
  return `brazier_${randomBytes(32).toString('base64url')}`
}

let daemon: ChildProcessWithoutNullStreams | undefined
let connection: Promise<Connection>
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
 * State owned by brazierd rather than by Electron.
 *
 * `userData` holds both, so a migration must move these entries individually.
 * Moving the whole directory would take Chromium's Preferences, Cookies, and
 * caches with it, and Electron is already using them by this point.
 */
const DAEMON_STATE = ['brazier.sqlite', 'runtime-settings.json', 'models', 'engines', 'downloads']

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
 * Move daemon state out of the pre-XDG location once, so existing installs keep
 * their models and conversations instead of silently starting empty.
 *
 * Same-filesystem renames make this instant even for a multi-gigabyte models
 * directory. Anything that fails to move is left where it is: a partial
 * migration that still launches beats refusing to start.
 */
function migrateLegacyDataDirectory(target: string): void {
  const legacy = LEGACY_USER_DATA
  if (legacy === target) {
    return
  }
  const pending = DAEMON_STATE.filter(
    (entry) => existsSync(join(legacy, entry)) && !existsSync(join(target, entry))
  )
  if (pending.length === 0) {
    return
  }
  try {
    mkdirSync(target, { recursive: true })
  } catch (error) {
    console.error('[brazier] could not create data directory', target, error)
    return
  }
  for (const entry of pending) {
    try {
      renameSync(join(legacy, entry), join(target, entry))
      console.log(`[brazier] migrated ${entry} to ${target}`)
    } catch (error) {
      console.error(`[brazier] could not migrate ${entry} from ${legacy}`, error)
    }
  }
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
  migrateLegacyDataDirectory(directory)
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
  if (serverSettings.enabled) {
    args.push('--host', '0.0.0.0', '--port', String(serverSettings.port))
    if (serverSettings.apiKeyEnabled) {
      const apiKey = serverSettings.apiKey ?? generatedApiKey()
      args.push('--api-key', apiKey)
      // Keep an auto-generated key stable across desktop restarts; otherwise
      // every configured OpenAI client would be revoked without warning.
      if (!serverSettings.apiKey) {
        serverSettings.apiKey = apiKey
        saveServerSettings(serverSettings)
      }
    } else {
      args.push('--no-auth', '--allow-insecure-remote')
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
  child.stdin.end()
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

    if (params.linkURL) {
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
    if (url.startsWith('https://')) void shell.openExternal(url)
    return { action: 'deny' }
  })
  window.webContents.on('will-navigate', (event, url) => {
    const allowed = process.env.ELECTRON_RENDERER_URL
    if (allowed && url.startsWith(allowed)) return
    if (url.startsWith('file://')) return
    event.preventDefault()
  })
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
      await window.loadFile(join(__dirname, '../renderer/index.html'))
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

app.whenReady().then(async () => {
  // No File/Edit/View application menu — the app is a self-contained shell.
  Menu.setApplicationMenu(null)
  // A packaged macOS app takes its dock icon from the bundle; an unpackaged one
  // shows Electron's unless it is set here.
  if (process.platform === 'darwin' && !app.isPackaged) {
    const icon = iconPath()
    if (icon) app.dock?.setIcon(icon)
  }
  connection = startDaemon()
  ipcMain.handle('brazier:connection', () => connection)
  ipcMain.handle('brazier:computer:set-active', (_event, active: boolean) =>
    setComputerUseActive(active === true))
  ipcMain.handle('brazier:computer:prepare-safety', () => prepareComputerSafety())
  ipcMain.handle('brazier:computer:input-guard-status', () => inputGuardStatus())
  ipcMain.handle('brazier:computer:setup-input-guard', () => setupInputGuard())
  ipcMain.handle('brazier:server-settings', (): ServerSettings => publicServerSettings(loadServerSettings()))
  ipcMain.handle(
    'brazier:save-server-settings',
    (_event, requested: Omit<ServerSettings, 'hasApiKey'> & { apiKey?: string | null }) => {
      const port = Number(requested.port)
      if (!Number.isInteger(port) || port < 1 || port > 65535) {
        throw new Error('Server port must be between 1 and 65535.')
      }
      const current = loadServerSettings()
      const apiKey = requested.apiKey === undefined ? current.apiKey : requested.apiKey?.trim() || null
      const next: StoredServerSettings = {
        enabled: requested.enabled === true,
        port,
        apiKeyEnabled: requested.apiKeyEnabled !== false,
        jitLoading: requested.jitLoading !== false,
        apiKey
      }
      saveServerSettings(next)
      return publicServerSettings(next)
    }
  )
  ipcMain.handle('brazier:generate-server-api-key', (): string => generatedApiKey())
  ipcMain.handle('brazier:flags', () => ({
    forceWelcome: forceWelcomeRequested()
  }))
  ipcMain.handle('brazier:check-for-updates', async () => {
    if (!checkForUpdates) return { supported: false }
    return checkForUpdates()
  })
  ipcMain.handle('brazier:get-update-settings', () => getUpdateSettings())
  ipcMain.handle(
    'brazier:save-update-settings',
    (
      _event,
      settings: { checkOnStartup?: boolean; autoDownload?: boolean }
    ) => saveUpdateSettings(settings ?? {})
  )
  ipcMain.handle('brazier:select-directory', async (event) => {
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
  ipcMain.handle(
    'brazier:select-file',
    async (event, title: string, filters: Electron.FileFilter[]) => {
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
  ipcMain.handle(
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
  ipcMain.handle('brazier:reveal-file', (_event, path: string) => {
    shell.showItemInFolder(path)
  })
  // Agent mode reaches the machine only through the daemon, so the worker gets
  // the loopback address and bearer token once the daemon is ready.
  registerAgentIpc(agent)
  ipcMain.handle('brazier:agent:status', () => agent.status())
  ipcMain.handle('brazier:select-workspace', async (event) => {
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
  connection
    .then((ready) => {
      agent.setConnection({ address: ready.address, apiKey: ready.api_key })
    })
    .catch(() => {
      // Reported below; Agent mode surfaces the daemon error when first used.
    })
  connection.catch((error: unknown) => {
    report(`[brazier] daemon failed to start: ${error instanceof Error ? error.stack ?? error.message : String(error)}`, 'error')
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
  void agent.shutdown()
  daemon?.kill()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
