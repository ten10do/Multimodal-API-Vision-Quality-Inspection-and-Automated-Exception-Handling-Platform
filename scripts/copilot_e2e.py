"""Phase 9T: real E2E of the Quality Copilot on live system data.

Browser-equivalent pipeline: HTTP POST /api/v1/copilot/query -> CopilotService
-> LLM provider (default offline fake) -> read-only tools -> PostgreSQL ->
evidence bundle -> grounded answer.

Seven required scenarios (9T):
  1. today's quality summary
  2. line anomaly analysis
  3. defect trend
  4. product rejection trace
  5. model drift
  6. review backlog
  7. PLC fault summary

Output: docs/copilot-e2e.json ; exit 0 when all scenarios pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "copilot-e2e.json"
BACKEND = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else "http://127.0.0.1:8000"

SCENARIOS = [
    ("quality_summary", "今天整体良率如何？", {"get_quality_summary"}),
    ("line_anomaly", "Line A 为什么异常？", {"compare_production_lines", "get_quality_summary"}),
    ("defect_trend", "哪种缺陷增长最快？", {"get_defect_distribution", "get_defect_trend"}),
    ("product_trace", "为什么产品被剔除了？请给出完整追溯链", {"get_inspection_detail", "get_product_history"}),
    ("model_drift", "当前模型是否存在漂移？", {"get_model_metrics", "get_drift_status", "get_review_metrics"}),
    ("review_backlog", "当前人工复核积压多少？", {"get_review_backlog"}),
    ("plc_fault", "最近 PLC 故障汇总", {"get_plc_fault_summary"}),
]


def main() -> int:
    results = []
    ok_all = True
    with httpx.Client(timeout=90) as c:
        for name, question, expected_tools in SCENARIOS:
            r = c.post(f"{BACKEND}/api/v1/copilot/query", json={"message": question})
            d = r.json()
            tools = set(d["tools_used"])
            grounded = "[insufficient evidence]" not in d["message"]
            read_only = d["safety"]["read_only"] is True and d["safety"]["write_actions_performed"] == []
            passed = bool(expected_tools & tools) and bool(d["evidence"]) and grounded and read_only and bool(d["message"])
            ok_all = ok_all and passed
            results.append(
                {
                    "scenario": name,
                    "question": question,
                    "tools_used": sorted(tools),
                    "evidence_count": len(d["evidence"]),
                    "grounded": grounded,
                    "read_only": read_only,
                    "confidence": d["confidence"],
                    "latency_ms": d["latency"]["total_latency_ms"],
                    "passed": passed,
                }
            )
            print(f"[{name}] {'PASS' if passed else 'FAIL'} tools={sorted(tools)} evidence={len(d['evidence'])} "
                  f"grounded={grounded} latency={d['latency']['total_latency_ms']}ms")

    summary = {"scenarios_total": len(SCENARIOS), "passed": sum(1 for r in results if r["passed"]), "all_passed": ok_all}
    OUT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
