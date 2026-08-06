const { chromium } = require("playwright");
const OUT = "D:/Multimodal API Vision Quality Inspection and Automated Exception Handling Platform/docs/screenshots/final";
const fs = require("fs");
fs.mkdirSync(OUT, { recursive: true });

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 940 } });

  // 09 Drift: Model Operations -> Drift panel
  await page.goto("http://127.0.0.1:5173/");
  await page.getByRole("button", { name: "Model Operations" }).click();
  await page.waitForSelector(".panel h3:has-text('Drift')", { timeout: 30000 });
  await page.waitForTimeout(1200);
  const drift = page.locator(".panel:has(h3:text('Drift'))");
  await drift.screenshot({ path: `${OUT}/09-drift.png` });
  console.log("09-drift.png");

  // 10 Quality Copilot: ask a question, show evidence panel
  await page.getByRole("button", { name: "Quality Copilot" }).click();
  await page.waitForSelector(".copilot-examples .chip-btn", { timeout: 30000 });
  await page.getByPlaceholder(/例如：为什么今天 Line A 良率下降/).fill("今天整体良率如何？");
  await page.getByRole("button", { name: "分析" }).click();
  await page.waitForSelector(".copilot-msg.assistant", { timeout: 30000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT}/10-quality-copilot.png` });
  console.log("10-quality-copilot.png");

  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
