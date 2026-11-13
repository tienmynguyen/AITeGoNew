import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  root: '.',
  publicDir: 'public',
  resolve: {
    preserveSymlinks: true,
    alias: {
      'kalidokit': path.resolve(__dirname, 'node_modules/kalidokit/dist/kalidokit.es.js')
    }
  },
  server: {
    port: 5173,
    host: 'localhost',
    open: false,
    fs: {
      strict: false
    }
  },
  build: {
    outDir: 'dist',
    target: 'esnext'
  }
});
