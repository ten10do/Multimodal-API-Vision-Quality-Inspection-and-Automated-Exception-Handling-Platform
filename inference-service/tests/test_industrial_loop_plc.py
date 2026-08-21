"""Industrial closed-loop tests: PLC adapter (Phase 3)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_loop.events import Decision, PlcStatus
from industrial_loop.plc_adapter import (
    InMemoryPlc,
    PLCState,
    PlcCommand,
    PlcSignal,
    PlcTraceLog,
    decision_to_signal,
    plc_status_for,
)


def _cmd(decision: Decision, command_id: str = "cmd-1") -> PlcCommand:
    return PlcCommand(command_id=command_id, event_id="evt-1", product_id="P1", decision=decision)


class TestDecisionToSignalMapping:
    def test_spec_mapping(self):
        assert decision_to_signal(Decision.PASS) is PlcSignal.NONE
        assert decision_to_signal(Decision.REJECT) is PlcSignal.REJECT_SIGNAL
        assert decision_to_signal(Decision.HOLD) is PlcSignal.STOP_SIGNAL


class TestInMemoryPlc:
    def test_pass_keeps_line_running(self, tmp_path):
        plc = InMemoryPlc(trace_log=PlcTraceLog(path=tmp_path / "t1.jsonl"))
        plc.start()
        result = plc.apply(_cmd(Decision.PASS))
        assert result.ack and result.signal is PlcSignal.NONE
        assert result.state_before is PLCState.RUNNING and result.state_after is PLCState.RUNNING
        assert plc_status_for(result) is PlcStatus.ACK_RUNNING

    def test_reject_sends_reject_signal_and_line_continues(self, tmp_path):
        plc = InMemoryPlc(trace_log=PlcTraceLog(path=tmp_path / "t2.jsonl"))
        plc.start()
        result = plc.apply(_cmd(Decision.REJECT))
        assert result.ack and result.signal is PlcSignal.REJECT_SIGNAL
        assert plc.state is PLCState.RUNNING
        assert plc_status_for(result) is PlcStatus.ACK_REJECT_SIGNAL
        assert plc.counters["reject_signals"] == 1

    def test_hold_sends_stop_signal_and_stops_line(self, tmp_path):
        plc = InMemoryPlc(trace_log=PlcTraceLog(path=tmp_path / "t3.jsonl"))
        plc.start()
        result = plc.apply(_cmd(Decision.HOLD))
        assert result.ack and result.signal is PlcSignal.STOP_SIGNAL
        assert plc.state is PLCState.STOP
        assert plc_status_for(result) is PlcStatus.ACK_STOP_SIGNAL
        assert plc.reset() is PLCState.RUNNING  # supervisor restart

    def test_idempotent_command_replay(self, tmp_path):
        log = PlcTraceLog(path=tmp_path / "t4.jsonl")
        plc = InMemoryPlc(trace_log=log)
        plc.start()
        first = plc.apply(_cmd(Decision.HOLD, command_id="same"))
        plc.reset()
        replay = plc.apply(_cmd(Decision.HOLD, command_id="same"))
        assert replay.duplicated and replay.state_after == first.state_after
        assert len(log.read_all()) == 1  # replay not re-executed / re-logged

    def test_nack_injection_maps_to_plc_nack(self, tmp_path):
        plc = InMemoryPlc(
            trace_log=PlcTraceLog(path=tmp_path / "t5.jsonl"),
            fail_command_ids={"bad"},
        )
        plc.start()
        result = plc.apply(_cmd(Decision.HOLD, command_id="bad"))
        assert not result.ack
        assert plc.state is PLCState.RUNNING  # NACK leaves the line untouched
        assert plc_status_for(result) is PlcStatus.NACK

    def test_trace_log_records_jsonl_rows(self, tmp_path):
        log_path = tmp_path / "trace.jsonl"
        plc = InMemoryPlc(trace_log=PlcTraceLog(path=log_path))
        plc.start()
        plc.apply(_cmd(Decision.REJECT, command_id="c1"))
        plc.apply(_cmd(Decision.HOLD, command_id="c2"))
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert [r["signal"] for r in rows] == ["reject_signal", "stop_signal"]
        assert rows[0]["decision"] == "REJECT" and rows[0]["ack"] is True
        assert {"timestamp", "command_id", "state_before", "state_after"} <= set(rows[0])


@pytest.mark.opcua
class TestOpcUaRoundTrip:
    """Live asyncua server + adapter loopback (marker-excluded in default runs)."""

    async def test_decision_to_plc_action_over_opc_ua(self):
        from industrial_loop.plc_adapter import OpcUaPlcAdapter, OpcUaPlcServer

        endpoint = f"opc.tcp://127.0.0.1:{random.randint(20000, 40000)}/freeopcua/server/"
        server = OpcUaPlcServer(endpoint)
        await server.start()
        try:
            server.core.start()
            adapter = OpcUaPlcAdapter(endpoint)
            hold = await adapter.apply(_cmd(Decision.HOLD, command_id="opc-1"))
            assert hold.ack and hold.state_after is PLCState.STOP
            server.core.reset()
            reject = await adapter.apply(_cmd(Decision.REJECT, command_id="opc-2"))
            assert reject.ack and reject.signal is PlcSignal.REJECT_SIGNAL
            assert server.core.state is PLCState.RUNNING
            assert len(server.core.trace_log.read_all()) == 2
        finally:
            await server.stop()
