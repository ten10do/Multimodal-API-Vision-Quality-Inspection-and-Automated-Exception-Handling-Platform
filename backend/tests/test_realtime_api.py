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
        "total_captured", "total_processed", "total_failed", "pass_count", "review_count",
        "fail_count", "queue_depth", "current_throughput", "average_processing_latency_ms",
        "p50_latency_ms", "p95_latency_ms", "ws_client_count",
    ):
        assert key in body, f"missing metric {key}"
    assert body["total_processed"] == 3
    assert body["total_failed"] == 1
    assert body["pass_count"] == 1 and body["review_count"] == 1 and body["fail_count"] == 1
    assert body["average_processing_latency_ms"] == pytest.approx(133.5, abs=0.1)


@pytest.mark.asyncio
async def test_telemetry_update(client):
    resp = await client.post("/api/v1/realtime/telemetry", json={
        "total_captured": 42, "queue_depth": 7, "processing_count": 2,
        "simulator_running": True, "worker_count": 2, "queue_size": 20, "queue_peak_depth": 9,
    })
    assert resp.status_code == 200
    body = (await client.get("/api/v1/realtime/status")).json()
    assert body["total_captured"] == 42
    assert body["queue_depth"] == 7
    assert body["worker_count"] == 2
    assert body["queue_peak_depth"] == 9
