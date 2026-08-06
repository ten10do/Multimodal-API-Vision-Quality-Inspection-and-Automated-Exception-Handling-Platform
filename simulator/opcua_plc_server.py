"""Minimal OPC UA PLC simulator (Phase 7).

Exposes a /Plc object with:
  State        (String: READY/RUNNING/HOLD/FAULT)
  LastCommand  (String: JSON payload)
  ExecutedCount (Int32: how many physical actions ran)

The `execute` method (link_method) is the command entry point; it is
idempotent by command_id (replaying the same id emits no second action).

Run:  python -m simulator.opcua_plc_server  (port 8503, opc.tcp://127.0.0.1:8503)
"""

from __future__ import annotations

import asyncio
import json
import threading

from asyncua import Server, ua

_NS = "urn:ivqc:plc"


class OpcUaPlcCore:
    def __init__(self) -> None:
        self.state = "READY"
        self.executed: set[str] = set()
        self.count = 0
        self.lock = threading.Lock()

    def execute(self, command_json: str) -> str:
        try:
            payload = json.loads(command_json)
        except Exception:
            return "NACK:bad_payload"
        cid = payload.get("command_id", "")
        with self.lock:
            if cid in self.executed:
                return "ACK:duplicate"
            if self.state == "FAULT":
                return "NACK:plc_fault"
            cmd = payload.get("command_type", "")
            if cmd == "HOLD":
                self.state = "HOLD"
            self.executed.add(cid)
            self.count += 1
            return f"ACK:{self.count}"


async def main() -> None:
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:8503")
    server.set_server_name("IVQC OPC UA PLC Simulator")
    idx = await server.register_namespace(_NS)
    objects = server.nodes.objects
    plc = await objects.add_object(idx, "Plc")
    core = OpcUaPlcCore()
    state_var = await plc.add_variable(idx, "State", "READY")
    await state_var.set_writable(True)
    last_var = await plc.add_variable(idx, "LastCommand", "")
    await last_var.set_writable(True)
    count_var = await plc.add_variable(idx, "ExecutedCount", 0)
    await count_var.set_writable(True)

    async def execute(parent, command_json) -> list:
        # asyncua 2.x passes input arguments as Variant values
        payload = command_json.Value if isinstance(command_json, ua.Variant) else command_json
        result = await asyncio.to_thread(core.execute, payload)
        await state_var.set_value(core.state)
        await last_var.set_value(payload)
        await count_var.set_value(core.count)
        # asyncua 2.x: the callback return value becomes OutputArguments,
        # which must be a LIST of Variants
        return [ua.Variant(result, ua.VariantType.String)]

    # asyncua 2.x: node.add_method(nodeid, browsename, callback,
    # [input VariantTypes], [output VariantTypes])
    await plc.add_method(idx, "execute", execute, [ua.VariantType.String], [ua.VariantType.String])
    print("OPC UA PLC simulator on opc.tcp://127.0.0.1:8503")
    async with server:
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
