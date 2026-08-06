const { chromium } = require("playwright");
const OUT = "D:/Multimodal API Vision Quality Inspection and Automated Exception Handling Platform/docs/screenshots";
const BACKEND = "http://127.0.0.1:8000";
async function main() {
  const list = await (await fetch(`${BACKEND}/api/v1/inspections?limit=100`)).json();
  // prefer a HELD / RELEASED / REJECTED sample to show the industrial block
  const target =
    list.find((i) => ["HELD", "RELEASED", "REJECTED"].includes(i.industrial_final_state)) || list[0];
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 940 } });
  await page.goto("http://127.0.0.1:5173/");
  await page.getByRole("button", { name: "Quality Traceability" }).click();
  await page.waitForSelector(".table tbody tr", { timeout: 30000 });
  await page.getByText(target.inspection_id).first().click();
  await page.waitForSelector(".modal", { timeout: 15000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${OUT}/11-phase7-industrial-detail.png` });
  console.log(
    "saved 11-phase7-industrial-detail.png |",
    target.inspection_id,
    "| state:", target.industrial_final_state,
    "| desired:", target.desired_command,
    "| exec:", target.execution_status,
    "| mes:", target.mes_sync_status,
  );
  await browser.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
