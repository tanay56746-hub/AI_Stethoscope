// =============================================================================
// [LAYER: FRONTEND ENTRYPOINT] | File: frontend/src/main.jsx
// Purpose: React 19 DOM Root Entrypoint & Global Styles Mount
// =============================================================================
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
