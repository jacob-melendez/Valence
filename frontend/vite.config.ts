// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - TanStack devtools (dev-only, first), tanstackStart, viteReact, tailwindcss, tsConfigPaths,
//     nitro (build-only using cloudflare as a default target), VITE_* env injection, @ path alias,
//     React/TanStack dedupe, error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";
// @ts-expect-error plain .mjs build helper without type declarations
import { writeSecurityHeaders } from "./scripts/security-headers.mjs";

// Valence Web is a static site: the console (and the in-browser engine) run entirely client-side, so there is no
// server worker. `nitro: false` skips the Cloudflare worker build (the Lovable sandbox still applies its own
// preset there), and the SPA shell is prerendered to dist/client/index.html. Deployed with wrangler.jsonc.
export default defineConfig({
  nitro: false,
  tanstackStart: {
    // Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
    server: { entry: "server" },
    spa: { enabled: true, prerender: { outputPath: "/index.html" } },
  },
  vite: {
    plugins: [
      {
        name: "valence-security-headers",
        apply: "build",
        // TanStack prerenders the shell after every build hook, so the headers (with hashes of the shell's inline
        // scripts) are written when the build process exits.
        buildApp: {
          order: "post",
          async handler() {
            process.once("exit", () => writeSecurityHeaders("dist/client"));
          },
        },
      },
    ],
  },
});
