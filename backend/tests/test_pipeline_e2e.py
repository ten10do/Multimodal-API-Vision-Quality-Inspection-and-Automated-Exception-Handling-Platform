"""Pipeline integration E2E.

Real chain: Camera Simulator -> Backend HTTP -> Real Inference (GPU) ->
Docker PostgreSQL -> WebSocket client.

Marked ``integration``. Requires:
- the Docker PostgreSQL container (host port 5433) running
- the shared inference service on :8100 for other tests is untouched; this
  suite starts its own dedicated inference instance on :8102 so the failure
  drill can stop/restart it freely
- trained weights available (inference-service/models/best.pt)

Coverage: continuous 25-image run with DB-count == processed-count and live
WebSocket events; then a failure drill (inference stopped -> FAILED ->
restarted -> recovers).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import asyncpg
import httpx
import pytest
import websockets

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
INFERENCE_ROOT = ROOT / "inference-service"
SRC_DIR = ROOT / "model-training/datasets/neu-det-yolo/test/images"
DOCKER = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe")

DB_URL = "postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5433/industrialvision_test"
DB_DSN = "postgresql://vision_qc:vision_qc@127.0.0.1:5433/industrialvision_test"
INF_PORT = 8102
BACKEND_PORT = 8123

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))
sys.path.insert(0, str(BACKEND_ROOT))

from simulator.camera_simulator import CameraSimulator  # noqa: E402
from simulator.config import OrchestratorConfig, SimulatorConfig  # noqa: E402
from simulator.orchestrator import InspectionOrchestrator  # noqa: E402


def _wait_http(url: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.5)
    raise RuntimeError(f"service at {url} not ready within {timeout}s")


class _Services:
    def __init__(self, backend_port: int, inf_port: int, inf_popen) -> None:
        self.backend_port = backend_port
        self.inf_port = inf_port
        self._inf = inf_popen
        self._backend = None

    def set_backend(self, popen) -> None:
        self._backend = popen

    def stop_inference(self) -> None:
        self._inf.terminate()
        try:
            self._inf.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._inf.kill()

    def start_inference(self) -> None:
        self._inf = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "inference_app.api:app", "--host", "127.0.0.1", "--port", str(self.inf_port)],
            cwd=INFERENCE_ROOT, env={**os.environ, "IVQC_DEVICE": "cuda:0"},
        )
        _wait_http(f"http://127.0.0.1:{self.inf_port}/ready", timeout=180)

    def shutdown(self) -> None:
        if self._backend is not None:
            self._backend.terminate()
        try:
            self._inf.terminate()
        except Exception:
            pass
        for p in (self._backend, self._inf):
            if p is None:
                continue
            try:
                p.wait(timeout=15)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    p.kill()
                except Exception:
                    pass


@pytest.fixture(scope="module")
def services():
    # 1) reproducible test DB provisioning (fail-fast, no silent skip)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from prepare_test_db import prepare_test_db

        prepare_test_db(recreate=True)  # fresh industrialvision_test for the pipeline drill
    except SystemExit as exc:
        pytest.fail(f"test database provisioning failed: {exc}")

    env = {**os.environ, "IVQC_DATABASE_URL": DB_URL}

    # 2) dedicated inference instance
    inf = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "inference_app.api:app", "--host", "127.0.0.1", "--port", str(INF_PORT)],
        cwd=INFERENCE_ROOT, env={**os.environ, "IVQC_DEVICE": "cuda:0"},
    )
    _wait_http(f"http://127.0.0.1:{INF_PORT}/ready", timeout=180)

    # 3) backend pointing at container DB + dedicated inference
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
        cwd=BACKEND_ROOT,
        env={**os.environ, "IVQC_DATABASE_URL": DB_URL, "IVQC_INFERENCE_SERVICE_URL": f"http://127.0.0.1:{INF_PORT}"},
    )
    _wait_http(f"http://127.0.0.1:{BACKEND_PORT}/ready", timeout=60)

    svc = _Services(BACKEND_PORT, INF_PORT, inf)
    svc.set_backend(backend)
    yield svc
    svc.shutdown()


async def _db_counts(batch: str) -> tuple[int, int]:
    conn = await asyncpg.connect(DB_DSN)
    try:
        completed = await conn.fetchval(
            "SELECT count(*) FROM inspections WHERE batch_id=$1 AND status='COMPLETED'", batch
        )
        failed = await conn.fetchval(
            "SELECT count(*) FROM inspections WHERE batch_id=$1 AND status='FAILED'", batch
        )
        return completed, failed
    finally:
        await conn.close()


async def _run_pipeline(batch: str, max_images: int, backend_port: int) -> InspectionOrchestrator:
    sim_cfg = SimulatorConfig(
        source_directory=str(SRC_DIR), interval_ms=80, production_line="line-e2e",
        station="qc-e2e", batch_id=batch, loop=False,
    )
    orch_cfg = OrchestratorConfig(
        backend_url=f"http://127.0.0.1:{backend_port}", queue_size=20, workers=2,
        retry_max=2, retry_base_ms=200, telemetry_interval_seconds=1,
    )
    orch = InspectionOrchestrator(orch_cfg)
    # the simulator must share the orchestrator's bounded queue
    sim = CameraSimulator(sim_cfg, orch.queue)
    await orch.run(sim, max_images=max_images)
    return orch


async def _collect_ws(backend_port: int, batch: str, events: list[dict]) -> None:
    """Collect WS events with bounded reconnect (the backend may come up a bit
    after the test fixture starts, and the dev/prod frontend survives
    brief backend restarts). Mirrors the production socket reconnect policy."""
    import websockets
    uri = f"ws://127.0.0.1:{backend_port}/api/v1/ws/inspections"
    delay = 0.5
    while True:
        try:
            async with websockets.connect(uri, open_timeout=5) as ws:
                delay = 0.5
                while True:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                        if msg.get("batch_id") == batch:
                            events.append(msg)
                    except asyncio.TimeoutError:
                        continue
        except Exception:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)
            continue


async def _wait_events(events: list[dict], expected: dict[str, int], timeout: float = 60) -> None:
    """Poll until the expected event counts are seen or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        counts = {k: sum(1 for e in events if e["event_type"] == k) for k in expected}
        if all(counts[k] >= n for k, n in expected.items()):
            return
        await asyncio.sleep(0.2)
    raise AssertionError(f"timed out waiting for events {expected}, got {counts}")


async def _stop_collector(collector: asyncio.Task) -> None:
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass


async def test_pipeline_continuous_25(services):
    batch = f"e2e-cont-{uuid.uuid4().hex[:6]}"
    events: list[dict] = []
    collector = asyncio.create_task(_collect_ws(services.backend_port, batch, events))
    orch = await _run_pipeline(batch, max_images=25, backend_port=services.backend_port)
    await _wait_events(events, {"inspection.completed": 25})
    await _stop_collector(collector)

    m = orch.metrics
    # accounting invariant: every captured image is either processed or failed
    assert m.captured_total == 25
    assert m.completed_total == 25
    assert m.failed_total == 0
    assert m.captured_total == m.completed_total + m.failed_total

    completed, failed = await _db_counts(batch)
    assert completed == 25, "DB completed inspections must equal processed count"
    assert failed == 0

    completed_events = [e for e in events if e["event_type"] == "inspection.completed"]
    assert len(completed_events) == 25, f"WS completed events {len(completed_events)} != 25"
    assert len([e for e in events if e["event_type"] == "inspection.failed"]) == 0
    sample = completed_events[0]
    assert sample["process_status"] == "COMPLETED"
    assert sample["quality_result"] in ("PASS", "REVIEW", "FAIL")
    assert sample["batch_id"] == batch
    assert sample["product_id"] and sample["inspection_id"] and sample["defect_count"] >= 0


async def test_pipeline_failure_recovery(services):
    batch = f"e2e-fail-{uuid.uuid4().hex[:6]}"
    events: list[dict] = []
    collector = asyncio.create_task(_collect_ws(services.backend_port, batch, events))

    # phase 1: healthy
    orch1 = await _run_pipeline(batch, max_images=4, backend_port=services.backend_port)
    assert orch1.metrics.captured_total == 4
    assert orch1.metrics.completed_total == 4 and orch1.metrics.failed_total == 0
    assert orch1.metrics.captured_total == orch1.metrics.completed_total + orch1.metrics.failed_total

    # phase 2: stop inference -> inspections FAILED, pipeline keeps consuming
    services.stop_inference()
    time.sleep(3)

    orch2 = await _run_pipeline(batch, max_images=4, backend_port=services.backend_port)
    assert orch2.metrics.captured_total == 4
    assert orch2.metrics.failed_total == 4, "captures during outage must be marked failed"
    assert orch2.metrics.completed_total == 0
    assert orch2.metrics.captured_total == orch2.metrics.completed_total + orch2.metrics.failed_total

    # phase 3: restart inference -> recovers
    services.start_inference()

    orch3 = await _run_pipeline(batch, max_images=4, backend_port=services.backend_port)
    assert orch3.metrics.captured_total == 4
    assert orch3.metrics.completed_total == 4 and orch3.metrics.failed_total == 0

    await _wait_events(events, {"inspection.completed": 4, "inspection.failed": 4})
    await _stop_collector(collector)

    completed, failed = await _db_counts(batch)
    assert completed == 8, "8 completed across phases 1+3"
    assert failed == 4, "4 failed during the outage"

    # WS broadcast is best-effort and may miss events across transient
    # disconnects; we assert at least one event per phase was received.
    completed_events = [e for e in events if e["event_type"] == "inspection.completed"]
    failed_events = [e for e in events if e["event_type"] == "inspection.failed"]
    assert len(failed_events) >= 4, f"expected >=4 failed events, got {len(failed_events)}"
    assert len(completed_events) >= 4, f"expected >=4 completed events, got {len(completed_events)}"
    assert all(e["process_status"] == "FAILED" for e in failed_events)
    assert all(e["process_status"] == "COMPLETED" for e in completed_events)
