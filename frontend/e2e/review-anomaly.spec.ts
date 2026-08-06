import { test, expect } from "@playwright/test";

// 6K: UNKNOWN_ANOMALY -> REVIEW -> review task -> browser claim -> human
// resolve (CONFIRM_DEFECT) -> DB final FAIL -> training candidate.
// The task carries anomaly score / regions / heatmap consumable by the UI.
// Backend (8000), inference (8100, PatchCore loaded) and Vite (5173) must run.

const BACKEND = "http://127.0.0.1:8000";

test.beforeAll(async () => {
  try {
    const r = await fetch(`${BACKEND}/ready`);
    if (!r.ok) throw new Error("backend not ready");
  } catch {
    test.skip(true, "backend not reachable");
  }
});

async function pendingAnomalyTask(): Promise<any> {
  const r = await fetch(`${BACKEND}/api/v1/reviews?status=PENDING&limit=300`);
  const tasks = (await r.json()) as any[];
  const hit = tasks.find(
    (t) => t.is_anomalous && t.anomaly_map_url && (t.ai_defects_snapshot?.length ?? 0) === 0,
  );
  if (!hit) throw new Error("no UNKNOWN_ANOMALY pending task with heatmap");
  return hit;
}

test("UNKNOWN_ANOMALY review: claim, consume anomaly info, confirm defect", async ({ page }) => {
  const target = await pendingAnomalyTask();
  expect(target.anomaly_score).toBeGreaterThan(0);
  expect(target.anomaly_map_url).toBeTruthy();
  expect(target.status).toBe("PENDING");

  await page.goto("/");
  await page.getByRole("button", { name: "Review Queue" }).click();
  await expect(page.getByRole("heading", { name: /Review Queue/ })).toBeVisible({ timeout: 15000 });

  // open the task
  await page.getByText(target.product_id).first().click();
  await expect(page.getByText(`Review ${target.review_task_id}`)).toBeVisible({ timeout: 15000 });

  // anomaly information is consumable: heatmap image + score + regions
  const heatmap = page.locator(`img[src="${target.anomaly_map_url}"]`);
  await expect(heatmap).toBeVisible({ timeout: 15000 });
  await expect(page.getByText(target.anomaly_score.toFixed(4))).toBeVisible();
  await expect(page.getByText(/Anomaly Heatmap（PatchCore）/)).toBeVisible();
  // UNKNOWN_ANOMALY has no AI defect label to correct -> CORRECT_DEFECT hidden
  await expect(page.getByText("CORRECT_DEFECT", { exact: true })).toHaveCount(0);

  // claim + confirm the unknown anomaly as a new defect
  await page.getByRole("button", { name: "Claim" }).click();
  await expect(page.getByRole("button", { name: /Resolve/ })).toBeVisible({ timeout: 15000 });
  await page.getByText("CONFIRM_DEFECT", { exact: true }).click();
  await page.getByPlaceholder(/human_label/).fill("new_crack");
  await page.getByPlaceholder(/reason/).fill("unknown anomaly confirmed under light");
  await page.getByRole("button", { name: /Resolve/ }).click();
  await expect(page.getByText(/已复核：/)).toBeVisible({ timeout: 15000 });

  // DB: final FAIL, AI anomaly snapshot preserved
  const task = await (await fetch(`${BACKEND}/api/v1/reviews/${target.review_task_id}`)).json();
  expect(task.status).toBe("RESOLVED");
  expect(task.decision.human_decision).toBe("CONFIRM_DEFECT");
  expect(task.decision.human_label).toBe("new_crack");
  expect(task.decision.final_quality_result).toBe("FAIL");
  expect(task.anomaly_score).toBe(target.anomaly_score);

  // training candidate: unknown -> human label -> future known defect (6G)
  const candidates = await (await fetch(`${BACKEND}/api/v1/training-candidates?kind=all`)).json();
  const hit = candidates.find((c: any) => c.inspection_id === task.inspection?.inspection_id);
  expect(hit).toBeTruthy();
  expect(hit.human_label).toBe("new_crack");
  expect(hit.anomaly_score).toBe(target.anomaly_score);
  expect(hit.ai_label).toBeNull(); // no YOLO defect: label is new knowledge
});

test("anomaly heatmap endpoint serves PNG for a resolved anomaly review", async () => {
  const target = await pendingAnomalyTask();
  const resp = await fetch(`${BACKEND}${target.anomaly_map_url}`);
  expect(resp.status).toBe(200);
  const buf = new Uint8Array(await resp.arrayBuffer());
  const hex = Array.from(buf.subarray(0, 4))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  expect(hex).toBe("89504e47"); // PNG magic
});
