"""Independent PLC Simulator (Phase 7).

A standalone HTTP server that mimics a programmable logic controller. It does
NOT couple to the backend ORM. It keeps its own machine state
(READY / RUNNING / HOLD / FAULT) and executes RELEASE / REJECT / HOLD / RESET
commands. Commands are idempotent: replaying the same command_id does not
execute a second physical action.

Fault injection via query params on POST /v1/command:
  ?mode=offline   -> connection refused (server still up, endpoint behaves)
  ?mode=timeout   -> hangs (test uses short client timeout)
  ?mode=nack      -> returns NACK
The backend tests drive these modes by pointing the adapter at per-case URLs.

Run:  python -m simulator.plc_simulator  (port 8501)
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel


class CommandIn(BaseModel):
    command_id: str
    product_id: str
    inspection_id: str
    command_type: str  # RELEASE / REJECT / HOLD / RESET
    reason_code: str
    timestamp: str = ""


@dataclass
class PlcAction:
    command_id: str
    inspection_id: str
    command_type: str
    reason_code: str
    at: float


class PlcSimulatorCore:
    """Pure state machine; the FastAPI layer is a thin transport."""

    def __init__(self) -> None:
        self.state = "READY"  # READY / RUNNING / HOLD / FAULT
        self.executed: dict[str, PlcAction] = {}  # idempotency: command_id -> action
        self.lock = threading.Lock()

    def execute(self, cmd: CommandIn) -> tuple[str, dict, bool]:
        """Returns (ack_code, response, executed_now)."""
        with self.lock:
            if cmd.command_id in self.executed:
                action = self.executed[cmd.command_id]
                return "ACK", {"duplicate": True, "command_id": cmd.command_id, "state": self.state, "first_at": action.at}, False
            if self.state == "FAULT":
                return "NACK", {"error": "plc_fault", "state": "FAULT"}, False
            # execute the physical/logical action once
            self.state = "RUNNING" if cmd.command_type == "RESET" else self.state
            if cmd.command_type == "HOLD":
                self.state = "HOLD"
            self.executed[cmd.command_id] = PlcAction(
                cmd.command_id, cmd.inspection_id, cmd.command_type, cmd.reason_code, time.time()
            )
            return "ACK", {"duplicate": False, "command_id": cmd.command_id, "state": self.state}, True


core = PlcSimulatorCore()
app = FastAPI(title="PLC Simulator", version="1.0.0")


@app.get("/v1/state")
async def get_state() -> dict:
    return {"state": core.state, "executed_count": len(core.executed)}


@app.post("/v1/command")
async def post_command(cmd: CommandIn, mode: str | None = Query(default=None)) -> dict:
    if mode == "timeout":
        await asyncio.sleep(30)  # the adapter's client timeout triggers first
    if mode == "nack":
        return {"ack": "NACK", "command_id": cmd.command_id, "error": "simulated_nack"}
    ack, response, executed = core.execute(cmd)
    return {"ack": ack, "command_id": cmd.command_id, "state": response.get("state"), "duplicate": response.get("duplicate")}


@app.post("/v1/admin/reset")
async def admin_reset() -> dict:
    with core.lock:
        core.state = "READY"
        core.executed.clear()
    return {"state": "READY"}


@app.post("/v1/admin/fault")
async def admin_fault() -> dict:
    with core.lock:
        core.state = "FAULT"
    return {"state": "FAULT"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8501)


if __name__ == "__main__":
    main()
