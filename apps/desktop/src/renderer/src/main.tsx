import '@fontsource-variable/inter'
import React from 'react'
import ReactDOM from 'react-dom/client'
import { App } from './App'
import './styles.css'

if (window.brazier.platform === 'darwin') {
  document.documentElement.classList.add('platform-darwin')
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
