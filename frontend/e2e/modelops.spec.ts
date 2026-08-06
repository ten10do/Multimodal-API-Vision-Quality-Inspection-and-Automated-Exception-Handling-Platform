/**
 * Phase 8 browser E2E: Model Operations page renders the registry, current
 * production, drift status and human feedback.
 */
import { test, expect } from "@playwright/test";

test("Model Operations page shows registry, production and drift", async ({ page }) => {
  await page.goto("http://127.0.0.1:5173/");
  await page.getByRole("button", { name: "Model Operations" }).click();
  await page.waitForSelector(".table", { timeout: 30000 });

  await expect(page.getByText("Model Registry（Phase 8）")).toBeVisible({ timeout: 15000 });
  await expect(page.getByText("Current Production Models")).toBeVisible();
  await expect(page.getByText("Registry", { exact: true })).toBeVisible();
  await expect(page.getByText("Production Metrics")).toBeVisible();
  await expect(page.getByText("Drift（8I）")).toBeVisible();
  await expect(page.getByText("Human Feedback（8H，按 model_version 关联）")).toBeVisible();

  // drift overall badge is one of NORMAL/WARNING/CRITICAL
  const overall = page.getByText(/^(NORMAL|WARNING|CRITICAL)$/).first();
  await expect(overall).toBeVisible({ timeout: 10000 });
});
