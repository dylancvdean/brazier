import { contextBridge, ipcRenderer } from 'electron'

export type BrazierConnection = {
  address: string
  api_key: string | null
}

contextBridge.exposeInMainWorld('brazier', {
  getConnection: (): Promise<BrazierConnection> => ipcRenderer.invoke('brazier:connection')
})
