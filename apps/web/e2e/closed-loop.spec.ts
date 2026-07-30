import { expect, test } from "@playwright/test";
import path from "node:path";

test("critical inspection completes the human-approved closed loop", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /让异常处置走完整个闭环/ }),
  ).toBeVisible();

  await page.getByLabel("产品编码").fill("E2E-CRITICAL");
  await page.getByLabel("生产批次").fill(`E2E-${Date.now()}`);
  await page
    .getByLabel(/质检图片/)
    .setInputFiles(
      path.resolve(__dirname, "../../../sample-data/mock-critical.png"),
    );
  await expect(page.getByRole("img", { name: "待检测图片预览" })).toBeVisible();
  await page.getByRole("button", { name: "开始质检" }).click();

  await expect(page).toHaveURL(/\/inspections\/[0-9a-f-]+$/);
  await expect(page.getByText("等待审批")).toBeVisible();
  await expect(page.getByText("创建异常工单")).toBeVisible();
  await expect(page.getByText("申请模拟停线")).toBeVisible();
  await expect(page.getByText("执行模拟停线")).toHaveCount(0);

  await page.getByRole("button", { name: "批准模拟停线" }).click();
  await expect(page.getByText("执行模拟停线")).toBeVisible();
  await expect(page.getByText("已完成").first()).toBeVisible();

  await page.getByRole("button", { name: "保存反馈" }).click();
  await expect(page.getByText(/最近反馈：值班主管/)).toBeVisible();

  await page.getByRole("link", { name: "返回检测台" }).click();
  await expect(page.getByText("E2E-CRITICAL")).toBeVisible();
});
