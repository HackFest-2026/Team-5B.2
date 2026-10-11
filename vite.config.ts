/// <reference types="vitest" />
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API = `http://127.0.0.1:${process.env.M360_API_PORT ?? "8010"}`;

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  // The UI only talks to the FastAPI backend; /api is proxied to it in development.
  server: { port: process.env.PORT ? Number(process.env.PORT) : 5173, proxy: { "/api": { target: API, changeOrigin: true } } },
  preview: { proxy: { "/api": { target: API, changeOrigin: true } } },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
