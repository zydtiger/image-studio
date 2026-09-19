import { defineConfig } from "@playwright/test";

/**
 * Browser tests for the Image Studio frontend.
 *
 * Default mode runs against the Vite dev server with a transport-level
 * fake API mounted through route interception. Integration mode instead
 * points at a real backend serving the built frontend:
 *
 *   pnpm --dir frontend run build
 *   uv run image-studio serve --fake-runtime --fake-hub --port 7860
 *   E2E_BASE_URL=http://127.0.0.1:7860 pnpm --dir tests/e2e run test
 *
 * Browser binaries come from the shared Playwright cache; this package
 * never downloads them.
 */
const realBase = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: ".",
  timeout: 30_000,
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [["list"]],
  use: {
    baseURL: realBase ?? "http://127.0.0.1:5199",
  },
  webServer: realBase
    ? undefined
    : {
        command:
          "pnpm --dir ../../frontend run dev --port 5199 --strictPort",
        url: "http://127.0.0.1:5199",
        reuseExistingServer: true,
        timeout: 120_000,
      },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
