// =============================================================================
// [LAYER: FRONTEND BUILD CONFIG] | File: frontend/vite.config.js
// Purpose: Vite Dev Server (Port 5173) & /api Proxy to Python Backend (Port 8000)
// =============================================================================
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load environment variables from project root (../.env) and frontend/
  const rootDir = path.resolve(__dirname, '..')
  const env = {
    ...loadEnv(mode, rootDir, ''),
    ...loadEnv(mode, __dirname, ''),
  }

  const frontendPort = parseInt(env.VITE_FRONTEND_PORT || '5173', 10)
  const backendUrl = env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'

  return {
    plugins: [react()],
    envDir: rootDir,
    server: {
      port: frontendPort,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
        },
      },
    },
  }
})
