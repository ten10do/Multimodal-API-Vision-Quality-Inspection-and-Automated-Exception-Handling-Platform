import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

const apiRoot = path.resolve(__dirname, "../api");
const pythonExecutable =
  process.platform === "win32"
    ? path.join(apiRoot, ".venv", "Scripts", "python.exe")
    : "python";

export default defineConfig({
  testDir: "./e2e",
  outputDir: "test-results",
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `"${pythonExecutable}" -m uvicorn app.main:app --host 127.0.0.1 --port 18000`,
      cwd: apiRoot,
      url: "http://127.0.0.1:18000/ready",
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        APP_ENV: "test",
        AI_MODE: "mock",
        CELERY_TASK_ALWAYS_EAGER: "true",
        DATABASE_URL: "sqlite+aiosqlite:///./playwright-e2e.db",
        UPLOAD_DIR: "./playwright-e2e-uploads",
        API_CORS_ORIGINS: '["http://127.0.0.1:3100"]',
      },
    },
    {
      command: "node node_modules/next/dist/bin/next dev -p 3100",
      url: "http://127.0.0.1:3100",
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        NEXT_PUBLIC_API_BASE_URL: "http://127.0.0.1:18000/api/v1",
        NEXT_PUBLIC_OPERATOR_ROLE: "supervisor",
      },
    },
  ],
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
