import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Bind-mounted source over virtiofs does not deliver inotify events reliably
    // on macOS/Colima. Polling costs a little CPU and is the difference between
    // hot reload working and silently not working. See CLAUDE.md, "Platform support".
    watch: { usePolling: true, interval: 300 },
    proxy: {
      "/api": { target: "http://api:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
