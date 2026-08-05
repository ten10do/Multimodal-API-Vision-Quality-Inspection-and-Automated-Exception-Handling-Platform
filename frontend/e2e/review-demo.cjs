// Phase 5 demo screenshots: Review Queue through a real chain.
const { chromium } = require("playwright");
const OUT = "D:/Multimodal API Vision Quality Inspection and Automated Exception Handling Platform/docs/screenshots";
const BACKEND = "http://127.0.0.1:8000";
const REVIEWER = "qc-worker-01";
const fs = require("fs");
fs.mkdirSync(OUT, { recursive: true });

async function shot(page, name) {
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log("screenshot:", name);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 940 } });

  // fetch a pending task
  const tasks = await (await fetch(`${BACKEND}/api/v1/reviews?status=PENDING&limit=100`)).json();
  const metrics = await (await fetch(`${BACKEND}/api/v1/reviews-metrics`)).json();
  console.log("pending tasks:", tasks.length, "| metrics:", JSON.stringify(metrics));
  if (tasks.length === 0) throw new Error("no pending review tasks (simulator must be running)");

  // 1. Review Queue list
  await page.goto("http://127.0.0.1:5173/");
  await page.getByRole("button", { name: "Review Queue" }).click();
  await page.waitForSelector(".table tbody tr", { timeout: 30000 });
  await page.waitForTimeout(800);
  await shot(page, "06-review-queue");

  // 2. Detail modal with image + BBox + AI prediction + review controls
  const target = tasks[0];
  await page.getByText(target.product_id).first().click();
  await page.waitForSelector(".modal", { timeout: 15000 });
  await page.waitForTimeout(800);
  await shot(page, "07-review-detail-ai");

  // 3. Claim + PASS resolve
  await page.getByRole("button", { name: "Claim" }).click();
  await page.waitForTimeout(400);
  await page.getByPlaceholder(/reason/).fill("human inspection: false positive");
  await shot(page, "08-review-controls-claimed");
  await page.getByRole("button", { name: /Resolve/ }).click();
  await page.waitForSelector("text=已复核：", { timeout: 15000 });
  await page.waitForTimeout(500);
  await shot(page, "09-review-resolved-pass");

  console.log("phase5 demo done");
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
