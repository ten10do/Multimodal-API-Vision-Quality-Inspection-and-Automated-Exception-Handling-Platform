// Phase 4 Demo: drives the real chain (Simulator -> Backend -> Inference ->
// Docker PG -> WebSocket -> Browser) and captures screenshots for each stage.
//
// Run:  NODE_OPTIONS= node node_modules/.bin/playwright... (see README)
const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:5173";
const OUT = "D:/Multimodal API Vision Quality Inspection and Automated Exception Handling Platform/docs/screenshots";
const fs = require("fs");
fs.mkdirSync(OUT, { recursive: true });

async function shot(page, name) {
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: false });
  console.log("screenshot:", name);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  // 1. Overview
  await page.goto(BASE);
  await page.waitForSelector(".metric-card", { timeout: 20000 });
  await page.waitForTimeout(1500);
  await shot(page, "01-overview");

  // 2. Live Inspection
  await page.getByRole("button", { name: "Live Inspection" }).click();
  await page.waitForSelector(".table tbody tr", { timeout: 20000 });
  await page.waitForTimeout(1000);
  await shot(page, "02-live-inspection");

  // 3. Traceability + detail with image & bbox
  await page.getByRole("button", { name: "Quality Traceability" }).click();
  await page.waitForSelector(".table.clickable tbody tr", { timeout: 20000 });
  await shot(page, "03-traceability");
  await page.locator(".table.clickable tbody tr").first().click();
  await page.waitForSelector(".modal", { timeout: 15000 });
  await page.waitForTimeout(800);
  await shot(page, "04-inspection-detail-bbox");
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);

  // 4. Simulator running state visible on overview
  await page.getByRole("button", { name: "Production Overview" }).click();
  await page.waitForTimeout(1200);
  await shot(page, "05-overview-running");

  console.log("demo screenshots done");
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
