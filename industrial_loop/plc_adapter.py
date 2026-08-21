"""Phase 3 — PLC industrial-control simulation (OPC UA, no real PLC).

Semantics required by the closed-loop spec:

    decision      signal           line state
    PASS   ->     (none)           stays RUNNING
    REJECT ->     reject_signal    stays RUNNING (product is ejected)
    HOLD   ->     stop_signal      STOP

``InMemoryPlc`` is a deterministic simulated PLC used by the default test
suite and the factory simulation. ``OpcUaPlcServer``/``OpcUaPlcAdapter``
provide the real OPC UA address space + client over ``asyncua`` for the
marked integration test and standalone runs (`python -m
industrial_loop.plc_adapter`). Every applied command is appended to a JSONL
trace log (runtime dir, gitignored).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import RUNTIME_ROOT
from .events import Decision, PlcStatus, utc_now_iso

PLC_TRACE_PATH = RUNTIME_ROOT / "plc_trace.jsonl"


class PLCState(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    STOP = "STOP"


class PlcSignal(str, Enum):
    NONE = "none"
    REJECT_SIGNAL = "reject_signal"
    STOP_SIGNAL = "stop_signal"


def decision_to_signal(decision: Decision) -> PlcSignal:
    return {
        Decision.PASS: PlcSignal.NONE,
        Decision.REJECT: PlcSignal.REJECT_SIGNAL,
        Decision.HOLD: PlcSignal.STOP_SIGNAL,
    }[decision]


@dataclass(frozen=True)
class PlcCommand:
    command_id: str
    event_id: str
    product_id: str
    decision: Decision
    timestamp: str = ""

    @property
    def signal(self) -> PlcSignal:
        return decision_to_signal(self.decision)


@dataclass(frozen=True)
class PlcCommandResult:
    ack: bool
    state_before: PLCState
    state_after: PLCState
    signal: PlcSignal
    duplicated: bool = False
    detail: str = "ok"


class PlcTraceLog:
    """Append-only JSONL trace of every PLC command (crash-safe enough)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or PLC_TRACE_PATH
        self._lock = threading.Lock()
        self._memory: list[dict] = []
        if self.path == PLC_TRACE_PATH:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        row = {"timestamp": utc_now_iso(), **record}
        with self._lock:
            self._memory.append(row)
            if self.path.parent.exists():
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        with self._lock:
            return list(self._memory)


class InMemoryPlc:
    """Deterministic simulated PLC: state machine + idempotent commands."""

    def __init__(
        self,
        *,
        trace_log: PlcTraceLog | None = None,
        fail_command_ids: set[str] | None = None,
    ) -> None:
        self.state = PLCState.READY
        self.trace_log = trace_log or PlcTraceLog()
        self._fail = set(fail_command_ids or ())
        self.executed: dict[str, PlcCommandResult] = {}
        self.counters = {"pass_kept_running": 0, "reject_signals": 0, "stop_signals": 0}
        self._lock = threading.Lock()

    def start(self) -> PLCState:
        with self._lock:
            self.state = PLCState.RUNNING
            return self.state

    def reset(self) -> PLCState:
        """Supervisor reset after a STOP (line restarts production)."""
        with self._lock:
            if self.state is PLCState.STOP:
                self.state = PLCState.RUNNING
            return self.state

    def apply(self, command: PlcCommand) -> PlcCommandResult:
        with self._lock:
            previous = self.executed.get(command.command_id)
            if previous is not None:
                return PlcCommandResult(
                    ack=previous.ack,
                    state_before=previous.state_before,
                    state_after=previous.state_after,
                    signal=command.signal,
                    duplicated=True,
                    detail="duplicate_command_id",
                )
            before = self.state
            if command.command_id in self._fail:
                result = PlcCommandResult(
                    ack=False,
                    state_before=before,
                    state_after=before,
                    signal=command.signal,
                    detail="simulated_nack",
                )
            else:
                after = before
                if command.signal is PlcSignal.STOP_SIGNAL:
                    after = PLCState.STOP
                elif before is PLCState.READY:
                    # first product on a fresh line implicitly starts the line
                    after = PLCState.RUNNING
                if command.decision is Decision.PASS:
                    self.counters["pass_kept_running"] += 1
                elif command.decision is Decision.REJECT:
                    self.counters["reject_signals"] += 1
                else:
                    self.counters["stop_signals"] += 1
                self.state = after
                result = PlcCommandResult(
                    ack=True, state_before=before, state_after=after, signal=command.signal
                )
            self.executed[command.command_id] = result
        self.trace_log.append(
            {
                "command_id": command.command_id,
                "event_id": command.event_id,
                "product_id": command.product_id,
                "decision": command.decision.value,
                "signal": command.signal.value,
                "state_before": result.state_before.value,
                "state_after": result.state_after.value,
                "ack": result.ack,
                "duplicated": result.duplicated,
                "detail": result.detail,
            }
        )
        return result


# --- OPC UA simulator (asyncua) ----------------------------------------------


def plc_status_for(result: PlcCommandResult) -> PlcStatus:
    """Map a command result onto the event's ``plc_status`` field."""
    if not result.ack:
        return PlcStatus.NACK
    return {
        PlcSignal.NONE: PlcStatus.ACK_RUNNING,
        PlcSignal.REJECT_SIGNAL: PlcStatus.ACK_REJECT_SIGNAL,
        PlcSignal.STOP_SIGNAL: PlcStatus.ACK_STOP_SIGNAL,
    }[result.signal]


class OpcUaPlcServer:
    """Minimal OPC UA address space mirroring ``InMemoryPlc`` semantics.

    Nodes under ``ns=<idx>;Objects->Plc``:
        State          (String)  READY / RUNNING / STOP
        RejectSignal   (Boolean)
        StopSignal     (Boolean)
        LastCommandId  (String)
        ExecutedCount  (Int32)
    """

    def __init__(self, endpoint: str = "opc.tcp://127.0.0.1:4840/freeopcua/server/") -> None:
        self.endpoint = endpoint
        self.core = InMemoryPlc(trace_log=PlcTraceLog(path=Path(os.devnull)))
        self._server = None

    async def start(self):
        from asyncua import Server, ua

        server = Server()
        await server.init()
        server.set_endpoint(self.endpoint)
        server.set_server_name("IVQC Industrial Loop PLC Simulator")
        idx = await server.register_namespace("urn:ivqc:industrial-loop")
        objects = server.nodes.objects
        plc = await objects.add_object(idx, "Plc")
        self._state = await plc.add_variable(idx, "State", PLCState.READY.value)
        self._reject = await plc.add_variable(idx, "RejectSignal", False)
        self._stop = await plc.add_variable(idx, "StopSignal", False)
        self._last = await plc.add_variable(idx, "LastCommandId", "")
        self._count = await plc.add_variable(idx, "ExecutedCount", 0)
        for node in (self._state, self._reject, self._stop, self._last, self._count):
            await node.set_writable(True)

        async def execute(parent, command_json) -> list:  # noqa: ANN001
            payload = command_json.Value if isinstance(command_json, ua.Variant) else command_json
            data = json.loads(payload)
            cmd = PlcCommand(
                command_id=data["command_id"],
                event_id=data.get("event_id", ""),
                product_id=data.get("product_id", ""),
                decision=Decision(data["decision"]),
            )
            result = self.core.apply(cmd)
            await self._state.set_value(self.core.state.value)
            await self._reject.set_value(cmd.signal is PlcSignal.REJECT_SIGNAL)
            await self._stop.set_value(cmd.signal is PlcSignal.STOP_SIGNAL)
            await self._last.set_value(cmd.command_id)
            await self._count.set_value(len(self.core.executed))
            return [ua.Variant(json.dumps({"ack": result.ack, "state": result.state_after.value}), ua.VariantType.String)]

        await plc.add_method(idx, "execute", execute, [ua.VariantType.String], [ua.VariantType.String])
        self._server = server
        await server.start()
        return self

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop()
            self._server = None


class OpcUaPlcAdapter:
    """Client-side adapter writing AI decisions into the OPC UA PLC."""

    def __init__(self, endpoint: str = "opc.tcp://127.0.0.1:4840/freeopcua/server/") -> None:
        self.endpoint = endpoint

    async def apply(self, command: PlcCommand) -> PlcCommandResult:
        from asyncua import Client, ua

        async with Client(url=self.endpoint) as client:
            idx = await client.get_namespace_index("urn:ivqc:industrial-loop")
            plc = await client.nodes.objects.get_child(f"{idx}:Plc")
            state_before = PLCState(await (await plc.get_child(f"{idx}:State")).read_value())
            method = await plc.get_child(f"{idx}:execute")
            payload = json.dumps(
                {
                    "command_id": command.command_id,
                    "event_id": command.event_id,
                    "product_id": command.product_id,
                    "decision": command.decision.value,
                }
            )
            result = await plc.call_method(method, ua.Variant(payload, ua.VariantType.String))
            # asyncua 2.x returns a single output value directly (plain str);
            # older versions may wrap outputs in a list/Variant.
            raw = result
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            if isinstance(raw, ua.Variant):
                raw = raw.Value
            outcome = json.loads(str(raw))
            state_after = PLCState(outcome["state"])
            ack_status = {
                PlcSignal.NONE: PlcStatus.ACK_RUNNING,
                PlcSignal.REJECT_SIGNAL: PlcStatus.ACK_REJECT_SIGNAL,
                PlcSignal.STOP_SIGNAL: PlcStatus.ACK_STOP_SIGNAL,
            }[command.signal]
            return PlcCommandResult(
                ack=bool(outcome["ack"]),
                state_before=state_before,
                state_after=state_after,
                signal=command.signal,
                detail=ack_status.value,
            )


def main() -> None:  # pragma: no cover - manual standalone simulator
    import asyncio

    async def _run() -> None:
        server = OpcUaPlcServer()
        await server.start()
        print(f"OPC UA PLC simulator on {server.endpoint} (Ctrl+C to stop)")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await server.stop()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
