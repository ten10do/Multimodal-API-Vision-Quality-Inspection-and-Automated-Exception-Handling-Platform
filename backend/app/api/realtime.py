from __future__ import annotations

import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..metrics import metrics
from ..ws import manager

rt_router = APIRouter(prefix="/api/v1/realtime", tags=["realtime"])
ws_router = APIRouter(prefix="/api/v1", tags=["realtime"])


@rt_router.get("/status")
async def realtime_status() -> dict:
    data = await metrics.snapshot()
    data["ws_client_count"] = manager.client_count
    return data


@rt_router.post("/telemetry")
async def update_telemetry(body: dict) -> dict:
    """Internal endpoint used by the orchestrator to report the pipeline view
    (produced captures, queue depth, workers). Terminal counters
    (completed/failed/pass/review/fail) are owned by the backend, which is the
    source of truth for persisted inspections, and cannot be overridden here.
    """
    allowed = {
        "captured_total",
        "queued_current",
        "processing_current",
        "simulator_running",
        "simulator_interval_ms",
        "worker_count",
        "queue_size",
        "queue_peak_depth",
    }
    await metrics.update_telemetry({k: v for k, v in body.items() if k in allowed})
    return {"status": "ok", "request_id": uuid.uuid4().hex[:12]}


@ws_router.websocket("/ws/inspections")
async def ws_inspections(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            # Receiving lets us detect disconnect promptly; payload ignored.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
