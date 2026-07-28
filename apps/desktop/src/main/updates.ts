import { app, dialog } from 'electron'
import { autoUpdater } from 'electron-updater'

/**
 * Update only from the GitHub Releases feed embedded by electron-builder.
 *
 * macOS additionally verifies the downloaded application's Developer ID
 * signature before it is installed. AppImage updates are checked against the
 * SHA-512 recorded in the release metadata by electron-updater.
 */
export function startUpdates(): void {
  // Distro packages are updated by their package manager. In particular, the
  // Arch launcher sets BRAZIER_INSTALLED, so it must never replace files that
  // pacman owns. On Linux the updater is meaningful only for a running
  // AppImage: electron-updater uses APPIMAGE to replace that exact file.
  if (
    !app.isPackaged ||
    process.env.BRAZIER_DISABLE_UPDATES === '1' ||
    process.env.BRAZIER_INSTALLED === '1' ||
    (process.platform === 'linux' && !process.env.APPIMAGE)
  ) {
    return
  }

  autoUpdater.autoDownload = true
  autoUpdater.autoInstallOnAppQuit = false

  autoUpdater.on('error', (error) => {
    // Update availability must never prevent an offline/local application from
    // launching. The release workflow's signed artifacts make failures
    // diagnosable without turning an unavailable network into a fatal error.
    console.warn('[brazier] update check failed', error)
  })
  autoUpdater.on('update-downloaded', async (info) => {
    const result = await dialog.showMessageBox({
      type: 'info',
      buttons: ['Restart and update', 'Later'],
      defaultId: 0,
      cancelId: 1,
      title: 'Update ready',
      message: `Brazier ${info.version} has been downloaded.`,
      detail: 'Restarting installs the signed update. Your conversations and models stay in place.'
    })
    if (result.response === 0) autoUpdater.quitAndInstall()
  })

  void autoUpdater.checkForUpdates()
}
