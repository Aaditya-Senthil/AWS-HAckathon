import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  // Get the base URL from the environment variable (can be '/your-app/' or '/')
  const baseUrl = env.VITE_APP_BASE_URL || '/';  // default to root ('/') if not set

  return {
    base: baseUrl,  // Set the base URL for assets
    plugins: [react()],
    define: {
      'process.env': env
    }
  }
})
