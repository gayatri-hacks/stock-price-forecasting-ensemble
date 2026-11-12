import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(({mode}) => {
  const target = loadEnv(mode, process.cwd(), "VITE_").VITE_API_TARGET || "http://127.0.0.1:8000";
  return {
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target, changeOrigin: true },
      "/health": { target, changeOrigin: true },
      "/ready": { target, changeOrigin: true },
    },
  },
  test: {environment:"jsdom", setupFiles:["./src/setup.ts"]},
  };
});
