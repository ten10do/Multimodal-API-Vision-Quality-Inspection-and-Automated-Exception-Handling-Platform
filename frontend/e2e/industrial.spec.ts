/**
 * Phase 7 browser E2E: the inspection detail panel shows the industrial
 * state block (desired command / execution status / industrial state / PLC
 * adapter / PLC latency / MES sync / reason code) with distinct badges.
 *
 * Requires the full stack (backend 8000, frontend 5173, simulators).
 */
import { test, expect } from "@playwright/test";

const BACKEND = "http://127.0.0.1:8000";

async function anyInspection() {
  const r = await fetch(`${BACKEND}/api/v1/inspections?limit=50`);
  const list = await r.json();
  // status is the enum value ("completed"), lowercase
  const hit = list.find((i: any) => String(i.status).toLowerCase() === "completed");
  if (!hit) throw new Error("no completed inspection");
  return hit;
}

test("inspection detail shows the industrial state block", async ({ page }) => {
  const insp = await anyInspection();
  await page.goto("http://127.0.0.1:5173/");
  await page.getByRole("button", { name: "Quality Traceability" }).click();
  await page.waitForSelector(".table tbody tr", { timeout: 30000 });
  await page.getByText(insp.inspection_id).first().click();
  await page.waitForSelector(".modal", { timeout: 15000 });

  // industrial section header (Phase 7)
  await expect(page.getByText("工业执行（Phase 7）")).toBeVisible({ timeout: 10000 });
  await expect(page.getByText("Desired Command")).toBeVisible();
  await expect(page.getByText("Execution Status")).toBeVisible();
  await expect(page.getByText("Industrial State")).toBeVisible();
  await expect(page.getByText("PLC Adapter")).toBeVisible();
  await expect(page.getByText("PLC Latency")).toBeVisible();
  await expect(page.getByText("MES Sync")).toBeVisible();
  await expect(page.getByText("Reason Code")).toBeVisible();

  // the industrial state badge sits in the "Industrial State" row
  const badge = page.locator('dt:has-text("Industrial State") + dd .badge');
  await expect(badge).toBeVisible();
  const text = await badge.textContent();
  expect(["RELEASED", "REJECTED", "HELD", "SAFE_HOLD", "COMMAND_FAILED", "NOT_INTEGRATED"]).toContain(text);
});

test("overview shows industrial counters", async ({ page }) => {
  await page.goto("http://127.0.0.1:5173/");
  await page.waitForSelector(".metric-grid", { timeout: 30000 });
  await expect(page.getByText("Released", { exact: true }).first()).toBeVisible({ timeout: 10000 });
  await expect(page.getByText("Rejected", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Held", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Safe Hold", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Not Integrated", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("MES Sync Failed", { exact: true }).first()).toBeVisible();
});
