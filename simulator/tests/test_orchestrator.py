from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from simulator.camera_simulator import CameraSimulator  # noqa: E402
from simulator.config import OrchestratorConfig, SimulatorConfig  # noqa: E402
from simulator.orchestrator import InspectionOrchestrator  # noqa: E402

IMG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x00\x00\xff\xd9"


def capture(product: str = "P-000001", key: str = "cap-1") -> dict:
    return {
        "product_id": product,
        "capture_id": key,
        "timestamp": "2026-08-04T00:00:00Z",
        "image_bytes": IMG,
        "filename": "img.jpg",
        "production_line": "line-a",
        "station": "qc-01",
        "batch_id": "batch-1",
    }


def make_capture(product: str = "P-000001", key: str = "cap-1"):
    from simulator.camera_simulator import Capture

    return Capture(**capture(product, key))


def make_src(tmp_path, count: int = 3):
    for i in range(count):
        (tmp_path / f"img_{i}.jpg").write_bytes(IMG)
    return tmp_path


async def run_orchestrator(transport, cfg: OrchestratorConfig, src, max_images=3):
    client = httpx.AsyncClient(transport=transport, timeout=10)
    orch = InspectionOrchestrator(cfg, client=client)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), orch.queue)
    await orch.run(sim, max_images=max_images)
    return orch


def ok_handler(body=None):
    body = body or {
        "inspection_id": "insp-x",
        "product_id": "P",
        "status": "completed",
        "quality_result": "REVIEW",
        "severity": "medium",
        "defects": [],
        "inference_latency_ms": 11.0,
        "model_version": "phase1-baseline",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(201, json=body)

    return handler


@pytest.mark.asyncio
async def test_orchestrator_success_flow(tmp_path):
    src = make_src(tmp_path, 3)
    cfg = OrchestratorConfig(backend_url="http://test", workers=1, retry_max=0, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(ok_handler()), cfg, src, max_images=3)
    m = orch.metrics
    assert m.total_captured == 3
    assert m.total_processed == 3
    assert m.total_failed == 0
    assert m.review_count == 3
    assert len(m.e2e_latencies) == 3
    assert len(m.inference_latencies) == 3


@pytest.mark.asyncio
async def test_orchestrator_retry_then_success(tmp_path):
    """Two transient 500s then success: retry_max=2 recovers, processed counted once."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(500, json={"error": {"code": "boom"}})
        return httpx.Response(201, json={
            "inspection_id": "insp-x", "product_id": "P", "status": "completed",
            "quality_result": "PASS", "severity": "low", "defects": [], "inference_latency_ms": 9.0,
        })

    src = make_src(tmp_path, 1)
    cfg = OrchestratorConfig(backend_url="http://test", workers=1, retry_max=2, retry_base_ms=10,
                             telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), cfg, src, max_images=1)
    assert calls["n"] == 3  # 2 failed attempts + 1 success
    assert orch.metrics.total_processed == 1
    assert orch.metrics.total_failed == 0


@pytest.mark.asyncio
async def test_orchestrator_no_infinite_retry(tmp_path):
    """Persistent 500: bounded retries, capture marked failed, system continues."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        calls["n"] += 1
        return httpx.Response(500, json={"error": {"code": "boom"}})

    src = make_src(tmp_path, 3)
    cfg = OrchestratorConfig(backend_url="http://test", workers=1, retry_max=2, retry_base_ms=10,
                             telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), cfg, src, max_images=3)
    assert orch.metrics.total_failed == 3
    assert orch.metrics.total_processed == 0
    assert calls["n"] == 3 * 3  # 3 captures x (1 + 2 retries)


@pytest.mark.asyncio
async def test_orchestrator_timeout_retry_idempotent(tmp_path):
    """Timeout on first attempt then success: idempotency key means the retried
    POST returns the SAME inspection; no duplicate counting."""
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        return httpx.Response(200, json={
            "inspection_id": "insp-same", "product_id": "P", "status": "completed",
            "quality_result": "FAIL", "severity": "high", "defects": [], "inference_latency_ms": 10.0,
        })

    src = make_src(tmp_path, 1)
    cfg = OrchestratorConfig(backend_url="http://test", workers=1, retry_max=2, retry_base_ms=10,
                             telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), cfg, src, max_images=1)
    assert calls["n"] == 2
    assert orch.metrics.total_processed == 1  # one logical inspection despite 2 HTTP calls
    assert orch.metrics.fail_count == 1


@pytest.mark.asyncio
async def test_failed_persisted_inspection_not_counted_as_success(tmp_path):
    """Retry after a persisted FAILED record (idempotent replay returns 200 +
    status FAILED) must be treated as a terminal failure, never a success."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/realtime/telemetry"):
            return httpx.Response(200, json={"status": "ok"})
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(504, json={"error": {"code": "inference_failed"}})
        return httpx.Response(200, json={
            "inspection_id": "insp-failed", "product_id": "P", "status": "FAILED",
            "quality_result": None, "severity": None, "defects": [], "error_message": "inference down",
        })

    src = make_src(tmp_path, 1)
    cfg = OrchestratorConfig(backend_url="http://test", workers=1, retry_max=2, retry_base_ms=10,
                             telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(handler), cfg, src, max_images=1)
    assert calls["n"] == 2  # 504 then idempotent replay, no further retry
    assert orch.metrics.total_failed == 1
    assert orch.metrics.total_processed == 0


@pytest.mark.asyncio
async def test_queue_bounded_and_no_drop(tmp_path):
    src = make_src(tmp_path, 5)
    cfg = OrchestratorConfig(backend_url="http://test", workers=0, retry_max=0, telemetry_interval_seconds=999)
    orch = InspectionOrchestrator(cfg)
    sim = CameraSimulator(SimulatorConfig(source_directory=str(src), interval_ms=1, loop=False), orch.queue)
    sim.start()
    await asyncio.sleep(0.3)
    assert orch.queue.qsize() <= cfg.queue_size, "queue must never exceed maxsize"
    assert sim.captured_count <= cfg.queue_size  # producer blocked at capacity
    await sim.stop()


@pytest.mark.asyncio
async def test_simulator_exhaustion_stops_pipeline(tmp_path):
    """loop=False + source exhausted -> pipeline terminates with all captures accounted."""
    src = make_src(tmp_path, 2)
    cfg = OrchestratorConfig(backend_url="http://test", workers=1, retry_max=0, telemetry_interval_seconds=999)
    orch = await run_orchestrator(httpx.MockTransport(ok_handler()), cfg, src, max_images=None)
    m = orch.metrics
    assert m.total_captured == 2
    assert m.total_processed == 2
    assert m.total_failed == 0
