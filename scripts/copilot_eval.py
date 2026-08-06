"""Phase 9 Copilot evaluation runner (9R).

Repeatable offline: with the default FakeLlmProvider (IVQC_LLM_PROVIDER
not set to openai) every query is deterministic. The runner exercises the
real HTTP pipeline (backend -> CopilotService -> tools -> PostgreSQL ->
grounding) against the fixed dataset in copilot-eval/cases.json and
measures:

  tool selection accuracy        expected tools subset of tools_used
  numeric grounding accuracy     no unsupported number survives grounding
  required fact coverage         required fact keys present in evidence
  unsupported claim rate         target 0 (critical acceptance metric)
  forbidden claim rate           causal/unverifiable phrases must not appear
  tool error recovery            tool errors must still yield a message
  avg tool calls / P50/P95 latency / token usage / estimated cost

Usage:  bash scripts/run_clean.sh python scripts/copilot_eval.py [--url http://127.0.0.1:8000]
Exit code 0 = all targets met.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "copilot-eval" / "cases.json"
OUT = ROOT / "docs" / "copilot-eval.json"
BACKEND = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else "http://127.0.0.1:8000"


def _find_key(obj, key: str) -> bool:
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_find_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_find_key(v, key) for v in obj)
    return False


def main() -> int:
    data = json.loads(CASES.read_text(encoding="utf-8"))
    cases = data["cases"]
    results = []
    totals = {"tool_calls": [], "latency": [], "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    metrics = {
        "tool_selection_pass": 0,
        "grounding_pass": 0,
        "fact_coverage_pass": 0,
        "unsupported_claims": 0,
        "forbidden_claims": 0,
        "tool_error_recovered": 0,
        "tool_error_total": 0,
    }

    with httpx.Client(timeout=90) as c:
        for case in cases:
            started = time.perf_counter()
            try:
                r = c.post(f"{BACKEND}/api/v1/copilot/query", json={"message": case["question"]})
                r.raise_for_status()
                out = r.json()
            except Exception as exc:  # noqa: BLE001
                results.append({"id": case["id"], "status": "ERROR", "detail": str(exc)})
                metrics["unsupported_claims"] += 1
                continue
            elapsed = (time.perf_counter() - started) * 1000.0

            tools_used = set(out["tools_used"])
            expected = set(case["expected_tools"])
            tool_ok = expected.issubset(tools_used) and not (expected == set() and tools_used)
            message = out["message"]
            limitations = out["limitations"]

            # grounding: no unsupported number may survive
            grounding_ok = "[insufficient evidence]" not in message and not any(
                "无证据支持" in l or "insufficient evidence" in l for l in limitations
            )
            # required facts: facts present in evidence, OR the tool honestly
            # errored (e.g. entity does not exist in this deployment) and the
            # pipeline recovered with a message (9S)
            has_tool_error = any("error" in e for e in out["evidence"])
            recovered = bool(message) and not message.startswith("分析服务暂时不可用")
            facts_ok = all(_find_key(out["evidence"], k) for k in case["required_facts"]) or (has_tool_error and recovered)

            if tool_ok:
                metrics["tool_selection_pass"] += 1
            if grounding_ok:
                metrics["grounding_pass"] += 1
            else:
                metrics["unsupported_claims"] += 1
            if facts_ok:
                metrics["fact_coverage_pass"] += 1
            forbidden_hit = [f for f in case["forbidden_claims"] if f in message]
            if forbidden_hit:
                metrics["forbidden_claims"] += 1
            if has_tool_error:
                metrics["tool_error_total"] += 1
                if recovered:
                    metrics["tool_error_recovered"] += 1

            totals["tool_calls"].append(out["latency"]["tool_call_count"])
            totals["latency"].append(out["latency"]["total_latency_ms"])
            totals["input_tokens"] += out["latency"]["input_tokens"]
            totals["output_tokens"] += out["latency"]["output_tokens"]
            totals["cost"] += out["latency"]["estimated_cost_usd"]

            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "tools_used": sorted(tools_used),
                    "expected_tools": sorted(expected),
                    "tool_ok": tool_ok,
                    "grounding_ok": grounding_ok,
                    "facts_ok": facts_ok,
                    "forbidden_hit": forbidden_hit,
                    "tool_error": has_tool_error,
                    "tool_call_count": out["latency"]["tool_call_count"],
                    "latency_ms": round(out["latency"]["total_latency_ms"], 2),
                }
            )

    n = len(cases)
    lat_sorted = sorted(totals["latency"])
    def pct(p):
        return round(lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * p))], 2) if lat_sorted else 0.0

    summary = {
        "cases_total": n,
        "cases_run": len(results),
        "tool_selection_accuracy": round(metrics["tool_selection_pass"] / n, 4),
        "numeric_grounding_accuracy": round(metrics["grounding_pass"] / n, 4),
        "required_fact_coverage": round(metrics["fact_coverage_pass"] / n, 4),
        "unsupported_critical_numeric_claim_rate": round(metrics["unsupported_claims"] / n, 4),
        "forbidden_claim_rate": round(metrics["forbidden_claims"] / n, 4),
        "tool_error_recovery_rate": round(metrics["tool_error_recovered"] / max(1, metrics["tool_error_total"]), 4),
        "avg_tool_calls": round(statistics.fmean(totals["tool_calls"]), 2) if totals["tool_calls"] else 0,
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "estimated_cost_usd": round(totals["cost"], 6),
        "provider": "offline-fake (deterministic)",
    }

    # acceptance targets (9R): unsupported critical numeric claim rate MUST be 0
    targets_met = (
        summary["unsupported_critical_numeric_claim_rate"] == 0.0
        and summary["numeric_grounding_accuracy"] == 1.0
        and summary["forbidden_claim_rate"] == 0.0
        and metrics["tool_error_recovered"] == metrics["tool_error_total"]
    )
    summary["targets_met"] = targets_met

    OUT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fails = [r for r in results if not (r.get("tool_ok") and r.get("grounding_ok") and r.get("facts_ok") and not r["forbidden_hit"])]
    if fails:
        print("--- non-passing cases ---")
        for f in fails[:12]:
            print(f["id"], f["category"], "tool_ok:", f["tool_ok"], "grounding:", f["grounding_ok"], "facts:", f["facts_ok"], "forbidden:", f["forbidden_hit"])
    print("TARGETS_MET:", targets_met)
    return 0 if targets_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
