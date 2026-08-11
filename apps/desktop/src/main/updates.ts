import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { randomBytes } from 'node:crypto'
import { app, dialog } from 'electron'
import { autoUpdater } from 'electron-updater'

type Report = (line: string, level?: 'log' | 'warn' | 'error') => void

export type UpdateCheckResult = {
  supported: boolean
}

export type UpdateSettings = {
  supported: boolean
  checkOnStartup: boolean
  autoDownload: boolean
}

type StoredUpdateSettings = {
  checkOnStartup: boolean
  autoDownload: boolean
}

const DEFAULT_STORED: StoredUpdateSettings = {
  checkOnStartup: true,
  autoDownload: false
}

const GITHUB_RELEASES_API = 'https://api.github.com/repos/dylancvdean/brazier/releases?per_page=100'

type GithubReleaseAsset = {
  name: string
  browser_download_url: string
}

type GithubRelease = {
  draft: boolean
  prerelease: boolean
  assets: GithubReleaseAsset[]
}

function updaterManifestName(): string {
  if (process.platform === 'darwin') return 'latest-mac.yml'
  if (process.platform === 'linux') return 'latest-linux.yml'
  return 'latest.yml'
}

/**
 * GitHub's /releases/latest can point at a partially published release.
 * Select the newest complete release for this platform instead, so an
 * experimental Windows failure cannot break update checks everywhere.
 */
async function configurePlatformUpdateFeed(report: Report): Promise<void> {
  const manifest = updaterManifestName()
  try {
    const response = await fetch(GITHUB_RELEASES_API, {
      headers: { Accept: 'application/vnd.github+json' }
    })
    if (!response.ok) throw new Error(`GitHub releases request failed (${response.status})`)
    const releases = (await response.json()) as GithubRelease[]
    const asset = releases
      .filter((release) => !release.draft && !release.prerelease)
      .flatMap((release) => release.assets)
      .find((candidate) => candidate.name === manifest)
    if (!asset) {
      report(`[updater] no published release contains ${manifest}; using the embedded GitHub feed`, 'warn')
      return
    }
    const slash = asset.browser_download_url.lastIndexOf('/')
    if (slash < 0) throw new Error(`invalid updater manifest URL: ${asset.browser_download_url}`)
    autoUpdater.setFeedURL({
      provider: 'generic',
      url: asset.browser_download_url.slice(0, slash + 1)
    })
  } catch (error) {
    // Retain electron-builder's embedded GitHub feed if the release listing
    // cannot be retrieved (offline use, rate limit, or a transient outage).
    report(
      `[updater] platform release lookup failed; using the embedded GitHub feed: ${error instanceof Error ? error.message : String(error)}`,
      'warn'
    )
  }
}

function updateSettingsPath(): string {
  return join(app.getPath('userData'), 'update-settings.json')
}

function loadStoredUpdateSettings(): StoredUpdateSettings {
  try {
    const parsed = JSON.parse(readFileSync(updateSettingsPath(), 'utf8')) as Partial<StoredUpdateSettings>
    return {
      checkOnStartup: parsed.checkOnStartup !== false,
      autoDownload: parsed.autoDownload === true
    }
  } catch {
    return { ...DEFAULT_STORED }
  }
}

function writeStoredUpdateSettings(settings: StoredUpdateSettings): void {
  const path = updateSettingsPath()
  mkdirSync(app.getPath('userData'), { recursive: true })
  const temporary = `${path}.${randomBytes(6).toString('hex')}.tmp`
  writeFileSync(temporary, `${JSON.stringify(settings, null, 2)}\n`, { mode: 0o600 })
  renameSync(temporary, path)
}

function updatesSupported(): boolean {
  return (
    app.isPackaged &&
    process.env.BRAZIER_DISABLE_UPDATES !== '1' &&
    process.env.BRAZIER_INSTALLED !== '1' &&
    !(process.platform === 'linux' && !process.env.APPIMAGE)
  )
}

/**
 * Update only from the GitHub Releases feed embedded by electron-builder.
 *
 * macOS additionally verifies the downloaded application's Developer ID
 * signature before it is installed. AppImage updates are checked against the
 * SHA-512 recorded in the release metadata by electron-updater.
 */
export function startUpdates(report: Report): { checkForUpdates: () => Promise<UpdateCheckResult> } {
  // Distro packages are updated by their package manager. In particular, the
  // Arch launcher sets BRAZIER_INSTALLED, so it must never replace files that
  // pacman owns. On Linux the updater is meaningful only for a running
  // AppImage: electron-updater uses APPIMAGE to replace that exact file.
  if (!updatesSupported()) {
    report('[updater] disabled for this installation')
    return { checkForUpdates: async () => ({ supported: false }) }
  }

  const settings = loadStoredUpdateSettings()

  // Availability is cheap to check, but an AppImage or macOS bundle can be a
  // large download. Never transfer it until its owner has said yes — unless
  // they opted into auto-download in Customization.
  autoUpdater.autoDownload = settings.autoDownload
  autoUpdater.autoInstallOnAppQuit = false

  let checking = false
  let interactiveCheck = false
  let downloadPromptOpen = false
  let downloading = false

  autoUpdater.on('error', async (error) => {
    // Update availability must never prevent an offline/local application from
    // launching. Record enough detail to diagnose a release-feed failure.
    report(`[updater] failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}`, 'warn')
    if (interactiveCheck) {
      await dialog.showMessageBox({
        type: 'error',
        title: 'Could not check for updates',
        message: 'Brazier could not check for an update.',
        detail: 'Check your connection and try again. The app will continue to work normally.'
      })
    }
  })
  autoUpdater.on('checking-for-update', () => report('[updater] checking for updates'))
  autoUpdater.on('update-not-available', (info) => {
    report(`[updater] already up to date (version ${info.version})`)
    if (interactiveCheck) {
      void dialog.showMessageBox({
        type: 'info',
        title: 'Brazier is up to date',
        message: `You already have the latest version (${info.version}).`
      })
    }
  })
  autoUpdater.on('download-progress', (progress) => {
    report(`[updater] download progress: ${Math.round(progress.percent)}% (${progress.transferred}/${progress.total} bytes)`)
  })
  autoUpdater.on('update-available', async (info) => {
    report(`[updater] update available: ${info.version}`)
    if (autoUpdater.autoDownload) {
      // Already downloading; skip the confirmation prompt.
      return
    }
    if (downloadPromptOpen || downloading) return
    downloadPromptOpen = true
    try {
      const result = await dialog.showMessageBox({
        type: 'info',
        buttons: ['Download update', 'Not now'],
        defaultId: 0,
        cancelId: 1,
        title: 'Update available',
        message: `Brazier ${info.version} is available.`,
        detail: 'Would you like to download the signed update now? You can keep using Brazier while it downloads.'
      })
      if (result.response !== 0) {
        report(`[updater] download declined for ${info.version}`)
        return
      }
      downloading = true
      report(`[updater] download approved for ${info.version}`)
      await autoUpdater.downloadUpdate()
    } catch (error) {
      report(
        `[updater] download failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}`,
        'error'
      )
    } finally {
      downloadPromptOpen = false
    }
  })
  autoUpdater.on('update-downloaded', async (info) => {
    downloading = false
    report(`[updater] update downloaded: ${info.version}`)
    const result = await dialog.showMessageBox({
      type: 'info',
      buttons: ['Restart and update', 'Later'],
      defaultId: 0,
      cancelId: 1,
      title: 'Update ready',
      message: `Brazier ${info.version} is ready to install.`,
      detail: 'Restarting installs the signed update. Your conversations and models stay in place.'
    })
    if (result.response === 0) {
      report(`[updater] installing ${info.version}`)
      autoUpdater.quitAndInstall()
    } else {
      report(`[updater] installation deferred for ${info.version}`)
    }
  })

  async function checkForUpdates(interactive = false): Promise<UpdateCheckResult> {
    if (checking) return { supported: true }
    checking = true
    interactiveCheck = interactive
    try {
      await configurePlatformUpdateFeed(report)
      await autoUpdater.checkForUpdates()
    } catch (error) {
      // electron-updater usually emits `error`, but retain this fallback so a
      // rejected check is never invisible in the log.
      report(
        `[updater] check rejected: ${error instanceof Error ? error.stack ?? error.message : String(error)}`,
        'warn'
      )
    } finally {
      checking = false
      interactiveCheck = false
    }
    return { supported: true }
  }

  if (settings.checkOnStartup) {
    void checkForUpdates()
  } else {
    report('[updater] startup check skipped (checkOnStartup=false)')
  }
  return { checkForUpdates: () => checkForUpdates(true) }
}

export function getUpdateSettings(): UpdateSettings {
  const stored = existsSync(updateSettingsPath())
    ? loadStoredUpdateSettings()
    : { ...DEFAULT_STORED }
  return {
    supported: updatesSupported(),
    checkOnStartup: stored.checkOnStartup,
    autoDownload: stored.autoDownload
  }
}

export function saveUpdateSettings(settings: {
  checkOnStartup?: boolean
  autoDownload?: boolean
}): UpdateSettings {
  const current = loadStoredUpdateSettings()
  const next: StoredUpdateSettings = {
    checkOnStartup:
      settings.checkOnStartup === undefined ? current.checkOnStartup : settings.checkOnStartup,
    autoDownload:
      settings.autoDownload === undefined ? current.autoDownload : settings.autoDownload
  }
  writeStoredUpdateSettings(next)
  if (updatesSupported()) {
    autoUpdater.autoDownload = next.autoDownload
  }
  return {
    supported: updatesSupported(),
    checkOnStartup: next.checkOnStartup,
    autoDownload: next.autoDownload
  }
}
