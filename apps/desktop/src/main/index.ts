import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from 'electron'

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

function repositoryRoot(): string {
  const candidates = [
    resolve(process.cwd()),
    resolve(app.getAppPath(), '../..'),
    resolve(__dirname, '../../../..')
  ]
  return candidates.find((candidate) => existsSync(join(candidate, 'Cargo.toml'))) ?? process.cwd()
}

function startDaemon(): Promise<Connection> {
  const dataDirectory = app.getPath('userData')
  const command = app.isPackaged
    ? join(process.resourcesPath, 'bin', process.platform === 'win32' ? 'brazierd.exe' : 'brazierd')
    : 'cargo'
  const args = app.isPackaged
    ? ['--data-dir', dataDirectory]
    : ['run', '-q', '-p', 'brazierd', '--', '--data-dir', dataDirectory]
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
  child.stderr.on('data', (chunk) => process.stderr.write(`[brazierd] ${chunk}`))

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

async function createWindow(): Promise<void> {
  const window = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 880,
    minHeight: 620,
    show: false,
    backgroundColor: '#10110f',
    autoHideMenuBar: true,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    ...(process.platform === 'darwin'
      ? { trafficLightPosition: { x: 16, y: 18 } }
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
  window.webContents.on('did-fail-load', (_event, code, description, validatedURL) => {
    console.error(`[brazier] renderer failed to load (${code}): ${description} @ ${validatedURL}`)
    if (!window.isDestroyed() && !window.isVisible()) window.show()
  })
  window.webContents.on('did-finish-load', () => {
    console.error('[brazier] renderer finished load')
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
  connection.catch((error: unknown) => {
    console.error('[brazier] daemon failed to start', error)
  })
  await createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) void createWindow()
  })
})

app.on('before-quit', () => {
  daemon?.kill()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
