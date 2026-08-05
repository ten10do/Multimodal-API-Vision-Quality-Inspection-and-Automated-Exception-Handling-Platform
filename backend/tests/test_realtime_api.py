from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.metrics import metrics  # noqa: E402


@pytest.fixture(autouse=True)
async def _reset_metrics():
    await metrics.reset()
    yield
    await metrics.reset()


@pytest.mark.asyncio
async def test_metrics_snapshot_fields(client):
    await metrics.record_completed("PASS", 120.5, 11.0)
    await metrics.record_completed("REVIEW", 80.0, 10.0)
    await metrics.record_completed("FAIL", 200.0, 12.0)
    await metrics.record_failed()
    resp = await client.get("/api/v1/realtime/status")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "captured_total", "queued_current", "processing_current", "completed_total",
        "failed_total", "pass_total", "review_total", "fail_total",
        "queue_peak_depth", "current_throughput", "average_processing_latency_ms",
        "p50_latency_ms", "p95_latency_ms", "ws_client_count",
    ):
        assert key in body, f"missing metric {key}"
    # backend-side invariants
    assert body["completed_total"] == 3
    assert body["failed_total"] == 1
    assert body["pass_total"] == 1 and body["review_total"] == 1 and body["fail_total"] == 1
    assert body["pass_total"] + body["review_total"] + body["fail_total"] == body["completed_total"]
    assert body["average_processing_latency_ms"] == pytest.approx(133.5, abs=0.1)


@pytest.mark.asyncio
async def test_metric_invariants_via_telemetry(client):
    """Conservation after drain: captured == completed + failed, and
    pass + review + fail == completed. Terminal counters come from the
    backend; pipeline-side captured/queued/processing come from telemetry."""
    for _ in range(9):
        await metrics.record_completed("PASS", 100.0, 10.0)
    for _ in range(12):
        await metrics.record_completed("REVIEW", 100.0, 10.0)
    for _ in range(6):
        await metrics.record_completed("FAIL", 100.0, 10.0)
    for _ in range(3):
        await metrics.record_failed()

    await client.post("/api/v1/realtime/telemetry", json={
        "captured_total": 30, "queued_current": 0, "processing_current": 0,
        "simulator_running": False, "worker_count": 2, "queue_size": 20, "queue_peak_depth": 9,
    })
    body = (await client.get("/api/v1/realtime/status")).json()
    assert body["captured_total"] == 30
    assert body["queued_current"] == 0 and body["processing_current"] == 0
    assert body["completed_total"] == 27 and body["failed_total"] == 3
    assert body["captured_total"] == body["completed_total"] + body["failed_total"]
    assert body["pass_total"] + body["review_total"] + body["fail_total"] == body["completed_total"]


@pytest.mark.asyncio
async def test_running_conservation_via_telemetry(client):
    """Running invariant: captured == queued + processing + completed + failed."""
    for _ in range(28):
        await metrics.record_completed("PASS", 100.0, 10.0)
    for _ in range(5):
        await metrics.record_failed()
    await client.post("/api/v1/realtime/telemetry", json={
        "captured_total": 40, "queued_current": 5, "processing_current": 2,
        "simulator_running": True, "worker_count": 2, "queue_size": 20, "queue_peak_depth": 8,
    })
    body = (await client.get("/api/v1/realtime/status")).json()
    assert body["captured_total"] == 5 + 2 + 28 + 5
    assert body["simulator_running"] is True


@pytest.mark.asyncio
async def test_telemetry_update(client):
    resp = await client.post("/api/v1/realtime/telemetry", json={
        "captured_total": 42, "queued_current": 7, "processing_current": 2,
        "simulator_running": True, "worker_count": 2, "queue_size": 20, "queue_peak_depth": 9,
    })
    assert resp.status_code == 200
    body = (await client.get("/api/v1/realtime/status")).json()
    assert body["captured_total"] == 42
    assert body["queued_current"] == 7
    assert body["worker_count"] == 2
    assert body["queue_peak_depth"] == 9
