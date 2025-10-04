import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path'; // NEW: Import the 'path' module

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  // NEW: Add the resolve alias configuration
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});