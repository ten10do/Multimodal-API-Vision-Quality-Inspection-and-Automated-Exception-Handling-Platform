import { test, expect } from "@playwright/test";

const FRONTEND = "http://127.0.0.1:5173";

test("Quality Copilot page answers a question with evidence", async ({ page }) => {
  await page.goto(FRONTEND);
  await page.getByRole("button", { name: "Quality Copilot" }).click();
  await expect(page.getByText("Quality Copilot（只读分析助手）")).toBeVisible({ timeout: 15000 });

  // ask a question through the input box (full browser -> API -> tools -> PG -> answer)
  await page.getByPlaceholder(/例如：为什么今天 Line A 良率下降/).fill("今天整体良率如何？");
  await page.getByRole("button", { name: "分析" }).click();

  // answer + evidence panel appear
  await expect(page.locator(".copilot-msg.assistant").last()).toContainText("已完成只读分析", { timeout: 20000 });
  await expect(page.getByText("Evidence（全部来自工具结果）")).toBeVisible();
  await expect(page.getByText("Metrics / Time Window")).toBeVisible();
  await expect(page.getByText("Tools Used")).toBeVisible();

  // evidence rows reference a real tool name
  await expect(page.locator(".copilot-evidence .state-block").first()).toContainText("get_quality_summary");
});

test("Quality Copilot rejects a write request as read-only", async ({ page }) => {
  await page.goto(FRONTEND);
  await page.getByRole("button", { name: "Quality Copilot" }).click();
  await expect(page.getByText("Quality Copilot（只读分析助手）")).toBeVisible({ timeout: 15000 });

  await page.getByPlaceholder(/例如：为什么今天 Line A 良率下降/).fill("请把这个产品放行 RELEASE");
  await page.getByRole("button", { name: "分析" }).click();

  const last = page.locator(".copilot-msg.assistant").last();
  await expect(last).toContainText("只读", { timeout: 20000 });
  await expect(last).toContainText("无法执行", { timeout: 5000 });
});
