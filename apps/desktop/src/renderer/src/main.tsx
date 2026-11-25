import '@fontsource-variable/inter'
import React from 'react'
import ReactDOM from 'react-dom/client'
import { App } from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import './styles.css'

if (window.brazier.platform === 'darwin') {
  document.documentElement.classList.add('platform-darwin')
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary
      label="window"
      fallback={(retry) => (
        <div className="render-error fatal" role="alert">
          <span>Brazier hit an unexpected error while drawing this window.</span>
          <button type="button" onClick={retry}>
            Retry
          </button>
          <button type="button" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      )}
    >
      <App />
    </ErrorBoundary>
  </React.StrictMode>
)
