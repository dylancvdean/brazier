import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'electron-vite'

export default defineConfig({
  main: {
    build: {
      rollupOptions: {
        input: resolve(__dirname, 'src/main/index.ts')
      }
    }
  },
  preload: {
    build: {
      rollupOptions: {
        input: resolve(__dirname, 'src/preload/index.ts')
      }
    }
  },
  renderer: {
    root: resolve(__dirname, 'src/renderer'),
    plugins: [react()],
    build: {
      // AudioWorklet modules have to stay real files served from the app
      // origin: the renderer's CSP allows `script-src 'self'` only, so an
      // inlined `data:` URL would be refused by the worklet loader.
      assetsInlineLimit: (filePath) =>
        filePath.endsWith('Worklet.js') ? false : undefined
    },
    server: {
      port: 5173,
      strictPort: true
    }
  }
})
