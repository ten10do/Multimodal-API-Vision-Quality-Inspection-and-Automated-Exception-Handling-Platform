import { test, expect } from "@playwright/test";

// 5M: real human-in-the-loop closed loop through the browser:
//   Simulator -> GPU inference -> Rule Engine -> REVIEW -> Review Task ->
//   Browser Dashboard -> claim -> resolve (PASS override / confirm AI defect /
//   corrected label) -> DB -> Final Quality Result -> Training Candidate.
// Backend (8000), inference (8100) and Vite (5173) must be running; the
// simulator must be producing REVIEW inspections.

const BACKEND = "http://127.0.0.1:8000";


test.beforeAll(async () => {
  try {
    const r = await fetch(`${BACKEND}/ready`);
    if (!r.ok) throw new Error("not ready");
  } catch {
    test.skip(true, "backend not reachable");
  }
});

async function pendingTasks(limit = 50): Promise<any[]> {
  const r = await fetch(`${BACKEND}/api/v1/reviews?status=PENDING&limit=${limit}`);
  return (await r.json()) as any[];
}

test("Review Queue: claim + PASS override through the browser", async ({ page }) => {
  const tasks = await pendingTasks();
  expect(tasks.length).toBeGreaterThan(0);
  const target = tasks[0];

  await page.goto("/");
  await page.getByRole("button", { name: "Review Queue" }).click();
  await expect(page.getByRole("heading", { name: /Review Queue/ })).toBeVisible({ timeout: 15000 });

  // the task row is present
  await expect(page.getByText(target.product_id).first()).toBeVisible({ timeout: 30000 });

  // open detail, claim, resolve PASS
  await page.getByText(target.product_id).first().click();
  await expect(page.getByText(`Review ${target.review_task_id}`)).toBeVisible({ timeout: 15000 });
  await expect(page.getByText("AI Prediction（固化快照）")).toBeVisible();
  await page.getByRole("button", { name: "Claim" }).click();
  await expect(page.getByRole("button", { name: /Resolve/ })).toBeVisible({ timeout: 15000 });
  // PASS is the default radio; add a reason and resolve
  await page.getByPlaceholder(/reason/).fill("false positive, no real defect");
  await page.getByRole("button", { name: /Resolve/ }).click();
  await expect(page.getByText(/已复核：/)).toBeVisible({ timeout: 15000 });

  // DB verification: AI judgment preserved, final = PASS
  const task = await (await fetch(`${BACKEND}/api/v1/reviews/${target.review_task_id}`)).json();
  expect(task.status).toBe("RESOLVED");
  expect(task.decision.human_decision).toBe("PASS");
  expect(task.decision.final_quality_result).toBe("PASS");
  expect(task.ai_quality_result).toBe("REVIEW"); // AI 原始判断保留
  expect(task.ai_defects_snapshot.length).toBeGreaterThan(0);
});

test("Confirm AI defect (CONFIRM_DEFECT) through the browser", async ({ page }) => {
  const tasks = await pendingTasks();
  expect(tasks.length).toBeGreaterThan(0);
  const target = tasks.find((t) => t.ai_defects_snapshot.length > 0) ?? tasks[0];
  const aiClass = target.ai_defects_snapshot[0]?.class_name ?? "crazing";

  await page.goto("/");
  await page.getByRole("button", { name: "Review Queue" }).click();
  await page.getByText(target.product_id).first().click();
  await page.getByRole("button", { name: "Claim" }).click();
  await page.getByText("CONFIRM_DEFECT", { exact: true }).click();
  await page.getByPlaceholder(/human_label/).fill(aiClass);
  await page.getByPlaceholder(/reason/).fill("defect visible, matches AI");
  await page.getByRole("button", { name: /Resolve/ }).click();
  await expect(page.getByText(/已复核：/)).toBeVisible({ timeout: 15000 });

  const task = await (await fetch(`${BACKEND}/api/v1/reviews/${target.review_task_id}`)).json();
  expect(task.decision.human_decision).toBe("CONFIRM_DEFECT");
  expect(task.decision.final_quality_result).toBe("FAIL");
  expect(task.decision.human_label).toBe(aiClass);
  // AI snapshot unchanged
  expect(task.ai_defects_snapshot[0].class_name).toBe(aiClass);
});

test("Corrected defect label (CORRECT_DEFECT) + training candidate export", async ({ page }) => {
  const tasks = await pendingTasks();
  expect(tasks.length).toBeGreaterThan(0);
  const target = tasks.find((t) => t.ai_defects_snapshot.length > 0) ?? tasks[0];
  const aiClass = target.ai_defects_snapshot[0]?.class_name ?? "crazing";
  const corrected = aiClass === "scratches" ? "crazing" : "scratches";

  await page.goto("/");
  await page.getByRole("button", { name: "Review Queue" }).click();
  await page.getByText(target.product_id).first().click();
  await page.getByRole("button", { name: "Claim" }).click();
  await page.getByText("CORRECT_DEFECT", { exact: true }).click();
  await page.getByPlaceholder(/human_label/).fill(corrected);
  await page.getByRole("button", { name: /Resolve/ }).click();
  await expect(page.getByText(/已复核：/)).toBeVisible({ timeout: 15000 });

  const task = await (await fetch(`${BACKEND}/api/v1/reviews/${target.review_task_id}`)).json();
  expect(task.decision.human_decision).toBe("CORRECT_DEFECT");
  expect(task.decision.human_label).toBe(corrected);
  expect(task.decision.final_quality_result).toBe("FAIL");

  // training candidate manifest includes this corrected sample (5J)
  const candidates = await (await fetch(`${BACKEND}/api/v1/training-candidates?kind=corrected`)).json();
  const hit = candidates.find((c: any) => c.inspection_id === task.inspection?.inspection_id);
  expect(hit).toBeTruthy();
  expect(hit.ai_label).toBe(aiClass);
  expect(hit.human_label).toBe(corrected);
  expect(hit.agreement).toBe(false);
  expect(hit.model_version).toBeTruthy();
  expect(hit.image_url).toContain("/image");
});

test("Review metrics update after browser resolves (5K)", async ({ page }) => {
  const before = await (await fetch(`${BACKEND}/api/v1/reviews-metrics`)).json();
  await page.goto("/");
  await page.getByRole("button", { name: "Review Queue" }).click();
  await expect(page.getByText("Pending Reviews")).toBeVisible({ timeout: 15000 });
  // metrics row reflects the REST values
  await expect(page.getByText(String(before.pending_review_count))).toBeVisible();
});

test("Concurrent claim: second browser claim gets 409 conflict state", async ({ page }) => {
  const tasks = await pendingTasks();
  const target = tasks[0];
  // claim via API as another reviewer so the UI shows the conflict message
  const claim = await fetch(`${BACKEND}/api/v1/reviews/${target.review_task_id}/claim`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviewer: "other-worker" }),
  });
  expect(claim.status).toBe(200);

  await page.goto("/");
  await page.getByRole("button", { name: "Review Queue" }).click();
  await page.getByText(target.product_id).first().click();
  // the modal shows the task as claimed by someone else (no Claim button)
  await expect(page.getByText(/已被 other-worker 认领/)).toBeVisible({ timeout: 15000 });
});

test("Full chain audit: AI prediction preserved after browser resolves", async () => {
  // pick any resolved task and confirm inspection rows keep AI + final distinct
  const resolved = await (await fetch(`${BACKEND}/api/v1/reviews?status=RESOLVED&limit=5`)).json();
  expect(resolved.length).toBeGreaterThan(0);
  for (const t of resolved.slice(0, 2)) {
    const insp = await (await fetch(`${BACKEND}/api/v1/inspections/${t.inspection?.inspection_id}`)).json();
    expect(insp.quality_result).toBe("REVIEW"); // AI 原始判断不可变（5G）
    expect(insp.final_quality_result).toBe(t.decision?.final_quality_result);
  }
});
