// Standalone build of the perception end-to-end harness (frontend/e2e/run.mjs). Not the console: it bundles only
// frontend/src/perception and serves the self-hosted MediaPipe files from frontend/public/mediapipe.
import path from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  root: __dirname,
  publicDir: path.resolve(__dirname, "../public"),
  resolve: { alias: { "@": path.resolve(__dirname, "../src") } },
  build: { outDir: path.resolve(__dirname, "dist"), emptyOutDir: true, sourcemap: false },
  logLevel: "warn",
});
