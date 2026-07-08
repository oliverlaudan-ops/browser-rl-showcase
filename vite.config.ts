import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    host: '127.0.0.1'
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          tfjs: ['@tensorflow/tfjs'],
          charts: ['chart.js']
        }
      }
    }
  }
});
