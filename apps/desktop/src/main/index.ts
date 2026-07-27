import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { createWriteStream, existsSync, mkdirSync, renameSync, type WriteStream } from 'node:fs'
import { writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from 'electron'

import { AgentSupervisor, registerAgentIpc } from './agent'

/**
 * Where Electron kept per-app state before the rename below.
 *
 * `app.setName` moves `userData`, so this is read first: an install that
 * predates the rename still has its models and conversations under the old
 * name, and `migrateLegacyDataDirectory` needs to be able to find them.
 */
const LEGACY_USER_DATA = app.getPath('userData')

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

let daemon: ChildProcessWithoutNullStreams | undefined
let connection: Promise<Connection>
const agent = new AgentSupervisor()

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

function startDaemon(): Promise<Connection> {
  const directory = dataDirectory()
  migrateLegacyDataDirectory(directory)
  const command = app.isPackaged
    ? join(process.resourcesPath, 'bin', process.platform === 'win32' ? 'brazierd.exe' : 'brazierd')
    : 'cargo'
  const args = app.isPackaged
    ? ['--data-dir', directory]
    : ['run', '-q', '-p', 'brazierd', '--', '--data-dir', directory]
  const child = spawn(command, args, {
    cwd: app.isPackaged ? undefined : repositoryRoot(),
    env: {
      ...process.env,
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
  const candidates = app.isPackaged
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
    console.error(`[brazier] renderer failed to load (${code}): ${description} @ ${validatedURL}`)
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
      console.error(`[renderer:${level}] ${message} (${sourceId}:${line})`)
    }
  })

  try {
    if (process.env.ELECTRON_RENDERER_URL) {
      await window.loadURL(process.env.ELECTRON_RENDERER_URL)
    } else {
      await window.loadFile(join(__dirname, '../renderer/index.html'))
    }
  } catch (error) {
    console.error('[brazier] failed to load renderer', error)
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
  ipcMain.handle('brazier:flags', () => ({
    forceWelcome: forceWelcomeRequested()
  }))
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
    console.error('[brazier] daemon failed to start', error)
  })
  await createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) void createWindow()
  })
})

app.on('before-quit', () => {
  void agent.shutdown()
  daemon?.kill()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
