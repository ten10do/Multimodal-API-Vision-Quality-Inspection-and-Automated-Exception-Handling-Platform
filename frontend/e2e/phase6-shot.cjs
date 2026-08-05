const { chromium } = require("playwright");
const OUT = "D:/Multimodal API Vision Quality Inspection and Automated Exception Handling Platform/docs/screenshots";
const BACKEND = "http://127.0.0.1:8000";
async function main() {
  const tasks = await (await fetch(`${BACKEND}/api/v1/reviews?status=PENDING&limit=300`)).json();
  const target = tasks.find((t) => t.is_anomalous && t.anomaly_map_url && (t.ai_defects_snapshot?.length ?? 0) === 0);
  if (!target) throw new Error("no anomaly task");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 940 } });
  await page.goto("http://127.0.0.1:5173/");
  await page.getByRole("button", { name: "Review Queue" }).click();
  await page.waitForSelector(".table tbody tr", { timeout: 30000 });
  await page.getByText(target.product_id).first().click();
  await page.waitForSelector(".modal", { timeout: 15000 });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/10-review-anomaly.png` });
  console.log("screenshot saved 10-review-anomaly.png, score:", target.anomaly_score);
  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
