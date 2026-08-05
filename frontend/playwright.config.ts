import { defineConfig } from "@playwright/test";

// The browser E2E runs against the real chain:
//   Vite dev server (5173) -> backend (8000) -> inference (8100) -> docker PG
// The backend and inference services must be started before running.
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 20_000 },
  retries: 0,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:5173",
    headless: true,
    screenshot: "only-on-failure",
    trace: "off",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
