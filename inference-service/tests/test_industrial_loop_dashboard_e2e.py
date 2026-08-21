"""Industrial closed-loop tests: dashboard + end-to-end factory simulation (Phases 6-7)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_loop.factory_simulator import FactorySimulator

REQUIRED_REPORT_KEYS = (
    "total_count", "pass_count", "reject_count", "hold_count",
    "plc_actions", "mes_orders", "latency", "reviews", "ground_truth_check", "lineage",
)


@pytest.fixture(scope="module")
def sim_report():
    simulator = FactorySimulator(products=60, seed=7)
    report = simulator.run()
    return simulator, report


class TestFactorySimulationE2E:
    def test_counts_conserve(self, sim_report):
        _, report = sim_report
        assert report["total_count"] == 60
        assert report["pass_count"] + report["reject_count"] + report["hold_count"] == 60

    def test_required_report_fields(self, sim_report):
        _, report = sim_report
        for key in REQUIRED_REPORT_KEYS:
            assert key in report
        plc = report["plc_actions"]
        assert plc["total"] == 60
        assert plc["reject_signals"] == report["reject_count"]
        assert plc["stop_signals"] >= 1 or report["hold_count"] == 0
        assert report["mes_orders"]["total"] == report["reject_count"]
        assert report["latency"]["loop_avg_ms"] is not None

    def test_every_reject_has_mes_order_and_review(self, sim_report):
        simulator, _ = sim_report
        events = simulator.store.events(limit=100)
        rejects = [e for e in events if e["decision"] == "REJECT"]
        holds = [e for e in events if e["decision"] == "HOLD"]
        assert rejects, "seeded run should contain rejects"
        for row in rejects:
            assert row["mes_status"] in {"OPEN", "PROCESSING", "CLOSED"}
            assert row["operator_status"] != "NOT_REQUIRED"
        for row in holds:
            assert row["reason_code"] == "AI_SYSTEM_FAILURE"
            assert row["plc_status"] == "ACK_STOP_SIGNAL"

    def test_ground_truth_accounting_wired(self, sim_report):
        _, report = sim_report
        truth = report["ground_truth_check"]
        assert truth["true_defects"] == truth["detected"] + truth["missed"]
        assert truth["detection_rate"] is not None

    def test_deterministic_for_fixed_seed(self):
        first = FactorySimulator(products=25, seed=11).run()
        second = FactorySimulator(products=25, seed=11).run()
        assert (first["pass_count"], first["reject_count"], first["hold_count"]) == (
            second["pass_count"], second["reject_count"], second["hold_count"],
        )
        assert first["plc_actions"]["reject_signals"] == second["plc_actions"]["reject_signals"]

    def test_report_is_json_serializable_and_writable(self, sim_report, tmp_path):
        simulator, report = sim_report
        path = simulator.write_report(report, tmp_path / "factory_simulation_report.json")
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["total_count"] == report["total_count"]
        assert payload["lineage"]["threshold_modified"] is False
        assert payload["lineage"]["d3_model_modified"] is False


class TestDashboard:
    def test_summary_events_trend_anomalies(self, sim_report):
        simulator, _ = sim_report
        client = TestClient(simulator.app)
        summary = client.get("/api/summary").json()
        assert summary["total"] == 60
        assert summary["pass"] + summary["reject"] + summary["hold"] == 60
        assert summary["plc"]["state"] in {"READY", "RUNNING", "STOP"}
        assert summary["mes"]["total"] == summary["reject"]

        events = client.get("/api/events", params={"limit": 10}).json()
        assert 0 < len(events) <= 10
        assert {"decision", "product_id", "camera_id"} <= set(events[0])

        trend = client.get("/api/trend").json()
        assert sum(b["pass"] + b["reject"] + b["hold"] for b in trend) == 60

        anomalies = client.get("/api/anomalies/recent").json()
        assert anomalies, "expected anomaly rows with heatmap previews"
        assert all(a["decision"] in {"REJECT", "HOLD"} for a in anomalies)
        assert any(a["heatmap_preview"] is not None for a in anomalies)

    def test_work_orders_reviews_plc_state_endpoints(self, sim_report):
        simulator, _ = sim_report
        client = TestClient(simulator.app)
        orders = client.get("/api/work-orders").json()
        assert len(orders) == simulator.mes.counts()["total"]
        assert {"work_order_id", "batch_id", "defect_type", "image_id", "severity", "status"} <= set(orders[0])

        reviews = client.get("/api/reviews").json()
        assert reviews["counts"]["total"] >= 1

        plc_state = client.get("/api/plc/state").json()
        assert {"state", "counters", "commands_executed"} <= set(plc_state)

    def test_index_page_served(self, sim_report):
        simulator, _ = sim_report
        client = TestClient(simulator.app)
        page = client.get("/").text
        assert "Closed-Loop Dashboard" in page
        assert "/api/summary" in page  # the SPA polls the same API
