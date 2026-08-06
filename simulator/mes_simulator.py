"""Mock MES API (Phase 7).

A standalone HTTP server exposing the minimal MES surface:
  GET  /v1/products/{product_id}
  GET  /v1/batches/{batch_id}
  POST /v1/inspection-results          (idempotent by inspection_id + type)
  POST /v1/final-quality-results       (idempotent by inspection_id + type)

Fault injection: POST /v1/admin/fault?type=500|timeout (per endpoint).

Run:  python -m simulator.mes_simulator  (port 8502)
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel


class InspectionResultIn(BaseModel):
    inspection_id: str
    product_id: str
    batch_id: str | None = None
    line: str = ""
    station: str = ""
    ai_result: str | None = None
    model_version: str | None = None
    rule_version: int | None = None
    industrial_state: str | None = None
    timestamp: str = ""


class FinalQualityResultIn(BaseModel):
    inspection_id: str
    product_id: str
    batch_id: str | None = None
    final_result: str
    reviewed_by: str | None = None
    industrial_state: str | None = None
    timestamp: str = ""


class MesSimulatorCore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inspection_results: dict[str, dict] = {}  # key: inspection_id + ":inspection"
        self.final_results: dict[str, dict] = {}      # key: inspection_id + ":final"
        self.products: dict[str, dict] = {}
        self.batches: dict[str, dict] = {}
        self.fault: dict[str, str] = {}  # endpoint -> "500" | "timeout"

    def post_inspection(self, body: InspectionResultIn) -> tuple[dict, bool]:
        key = f"{body.inspection_id}:inspection"
        with self.lock:
            if key in self.inspection_results:
                return {"duplicate": True, "inspection_id": body.inspection_id}, False
            self.inspection_results[key] = body.model_dump()
            self.products.setdefault(body.product_id, {"product_id": body.product_id, "line": body.line, "station": body.station})
            if body.batch_id:
                self.batches.setdefault(body.batch_id, {"batch_id": body.batch_id, "count": 0})
            return {"duplicate": False, "inspection_id": body.inspection_id}, True

    def post_final(self, body: FinalQualityResultIn) -> tuple[dict, bool]:
        key = f"{body.inspection_id}:final"
        with self.lock:
            if key in self.final_results:
                return {"duplicate": True, "inspection_id": body.inspection_id}, False
            self.final_results[key] = body.model_dump()
            return {"duplicate": False, "inspection_id": body.inspection_id}, True


core = MesSimulatorCore()
app = FastAPI(title="MES Simulator", version="1.0.0")


async def _maybe_fault(endpoint: str) -> None:
    f = core.fault.get(endpoint)
    if f == "timeout":
        await asyncio.sleep(30)


@app.get("/v1/products/{product_id}")
async def get_product(product_id: str) -> dict:
    p = core.products.get(product_id, {"product_id": product_id})
    return p


@app.get("/v1/batches/{batch_id}")
async def get_batch(batch_id: str) -> dict:
    b = core.batches.get(batch_id, {"batch_id": batch_id, "count": 0})
    return b


@app.post("/v1/inspection-results")
async def post_inspection_result(body: InspectionResultIn) -> dict:
    if core.fault.get("inspection") == "500":
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="simulated_mes_500")
    await _maybe_fault("inspection")
    data, executed = core.post_inspection(body)
    return {"status": "ok", **data}


@app.post("/v1/final-quality-results")
async def post_final_quality_result(body: FinalQualityResultIn) -> dict:
    if core.fault.get("final") == "500":
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="simulated_mes_500")
    await _maybe_fault("final")
    data, executed = core.post_final(body)
    return {"status": "ok", **data}


@app.post("/v1/admin/fault")
async def admin_fault(endpoint: str = Query(...), mode: str = Query(...)) -> dict:
    core.fault[endpoint] = mode
    return {"endpoint": endpoint, "mode": mode}


@app.post("/v1/admin/reset")
async def admin_reset() -> dict:
    with core.lock:
        core.inspection_results.clear()
        core.final_results.clear()
        core.fault.clear()
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8502)


if __name__ == "__main__":
    main()
