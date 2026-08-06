const { chromium } = require("playwright");
const OUT = "D:/Multimodal API Vision Quality Inspection and Automated Exception Handling Platform/docs/screenshots";
async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 940 } });
  await page.goto("http://127.0.0.1:5173/");
  await page.getByRole("button", { name: "Model Operations" }).click();
  await page.waitForSelector(".table", { timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT}/12-phase8-modelops.png` });
  console.log("saved 12-phase8-modelops.png");
  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
