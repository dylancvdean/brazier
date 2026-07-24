import { contextBridge, ipcRenderer } from 'electron'

export type BrazierConnection = {
  address: string
  api_key: string | null
}

export type BrazierFlags = {
  forceWelcome: boolean
}

contextBridge.exposeInMainWorld('brazier', {
  getConnection: (): Promise<BrazierConnection> => ipcRenderer.invoke('brazier:connection'),
  getFlags: (): Promise<BrazierFlags> => ipcRenderer.invoke('brazier:flags'),
  platform: process.platform,
  selectDirectory: (): Promise<string | null> => ipcRenderer.invoke('brazier:select-directory')
})
