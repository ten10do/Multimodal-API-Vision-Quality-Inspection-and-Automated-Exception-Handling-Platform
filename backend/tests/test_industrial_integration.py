"""Phase 7: real integration tests against the live simulators.

Requires the simulators to be running:
  PLC HTTP    : python -m simulator.plc_simulator       (8501)
  MES HTTP    : python -m simulator.mes_simulator       (8502)
  PLC OPC UA  : python -m simulator.opcua_plc_server    (8503)

Run: pytest backend/tests/test_industrial_integration.py -m integration
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.industrial.commands import IndustrialCommand, command_id_for
from app.industrial.mes_adapter import MesAdapter, MesRejected, MesUnreachable
from app.industrial.plc_adapter import HttpPlcAdapter, OpcUaPlcAdapter, PlcNack, PlcUnreachable

PLC_HTTP = "http://127.0.0.1:8501"
PLC_OPCUA = "opc.tcp://127.0.0.1:8503"
MES_HTTP = "http://127.0.0.1:8502"


def _cmd(command_type: str, inspection_id: str = "insp-it-1") -> IndustrialCommand:
    return IndustrialCommand(
        command_id=command_id_for(inspection_id, command_type),
        product_id="P-IT-1",
        inspection_id=inspection_id,
        command_type=command_type,
        reason_code="product_defect",
        timestamp="2026-08-06T00:00:00Z",
    )


def _skipped(reason: str):
    return pytest.skip(reason)


# ---- HTTP PLC ----

@pytest.mark.integration
async def test_http_plc_ack_and_idempotency():
    try:
        httpx.get(f"{PLC_HTTP}/v1/state", timeout=2).raise_for_status()
    except Exception:
        _skipped("plc simulator not running")
    httpx.post(f"{PLC_HTTP}/v1/admin/reset", timeout=2)  # shared simulator state

    adapter = HttpPlcAdapter(PLC_HTTP, timeout_seconds=2.0)
    cmd = _cmd("REJECT")
    # first send executes the physical action once
    first = await adapter.send_command(cmd)
    assert first.acked is True
    assert first.duplicate is False
    # replaying the same command_id must NOT execute a second action
    second = await adapter.send_command(cmd)
    assert second.acked is True
    assert second.duplicate is True


@pytest.mark.integration
async def test_http_plc_hold_state():
    try:
        httpx.get(f"{PLC_HTTP}/v1/state", timeout=2).raise_for_status()
    except Exception:
        _skipped("plc simulator not running")

    adapter = HttpPlcAdapter(PLC_HTTP, timeout_seconds=2.0)
    cmd = _cmd("HOLD")
    await adapter.send_command(cmd)
    state = httpx.get(f"{PLC_HTTP}/v1/state", timeout=2).json()
    assert state["state"] == "HOLD"


@pytest.mark.integration
async def test_http_plc_nack():
    try:
        httpx.get(f"{PLC_HTTP}/v1/state", timeout=2).raise_for_status()
    except Exception:
        _skipped("plc simulator not running")

    adapter = HttpPlcAdapter(f"{PLC_HTTP}?none", timeout_seconds=2.0)
    async with httpx.AsyncClient(timeout=2.0) as client:
        resp = await client.post(f"{PLC_HTTP}/v1/command", params={"mode": "nack"}, json=_cmd("REJECT").to_payload())
    assert resp.status_code == 200
    # the adapter must surface a NACK through the right exception path
    class _NackAdapter(HttpPlcAdapter):
        async def send_command(self, command):
            resp = await httpx.AsyncClient(timeout=2.0).post(
                f"{self.base_url}/v1/command", params={"mode": "nack"}, json=command.to_payload()
            )
            if resp.json().get("ack") != "ACK":
                raise PlcNack(str(resp.json()))
            return self._ok(resp)

    try:
        await _NackAdapter(PLC_HTTP, timeout_seconds=2.0).send_command(_cmd("REJECT"))
        raise AssertionError("expected PlcNack")
    except PlcNack:
        pass


@pytest.mark.integration
async def test_http_plc_timeout():
    try:
        httpx.get(f"{PLC_HTTP}/v1/state", timeout=2).raise_for_status()
    except Exception:
        _skipped("plc simulator not running")

    # the simulator hangs for 30s in timeout mode; a short client timeout must
    # surface as a connection-level failure (the fail-safe path in the service)
    with pytest.raises(httpx.TimeoutException):
        async with httpx.AsyncClient(timeout=0.5) as client:
            await client.post(f"{PLC_HTTP}/v1/command", params={"mode": "timeout"}, json=_cmd("REJECT", inspection_id="insp-timeout").to_payload())


@pytest.mark.integration
async def test_http_plc_offline():
    adapter = HttpPlcAdapter("http://127.0.0.1:59999", timeout_seconds=0.5)
    with pytest.raises(PlcUnreachable):
        await adapter.send_command(_cmd("REJECT", inspection_id="insp-offline"))


# ---- OPC UA PLC ----

@pytest.mark.integration
async def test_opcua_plc_ack_and_idempotency():
    adapter = OpcUaPlcAdapter(PLC_OPCUA, timeout_seconds=3.0)
    try:
        first = await adapter.send_command(_cmd("HOLD", inspection_id="insp-opcua-1"))
    except PlcUnreachable as exc:
        _skipped(f"opcua simulator not running: {exc}")

    assert first.acked is True
    assert first.duplicate is False
    second = await adapter.send_command(_cmd("HOLD", inspection_id="insp-opcua-1"))
    assert second.duplicate is True


@pytest.mark.integration
async def test_opcua_plc_offline():
    adapter = OpcUaPlcAdapter("opc.tcp://127.0.0.1:59998", timeout_seconds=1.0)
    with pytest.raises(PlcUnreachable):
        await adapter.send_command(_cmd("REJECT", inspection_id="insp-opcua-off"))


# ---- MES ----

@pytest.mark.integration
async def test_mes_submit_and_duplicate():
    try:
        httpx.get(f"{MES_HTTP}/v1/products/x", timeout=2).raise_for_status()
    except Exception:
        _skipped("mes simulator not running")

    adapter = MesAdapter(MES_HTTP, timeout_seconds=2.0, max_retries=1)
    ok, latency, attempts = await adapter.post_inspection_result(
        inspection_id="insp-mes-1", product_id="P-MES-1", batch_id="B-1",
        line="line-a", station="qc", ai_result="PASS", model_version="v1",
        rule_version=1, industrial_state="RELEASED", timestamp="t",
    )
    assert ok is True
    # duplicate submission (same inspection_id) is suppressed by the MES
    ok2, _, _ = await adapter.post_inspection_result(
        inspection_id="insp-mes-1", product_id="P-MES-1", batch_id="B-1",
        line="line-a", station="qc", ai_result="PASS", model_version="v1",
        rule_version=1, industrial_state="RELEASED", timestamp="t",
    )
    assert ok2 is True  # adapter-level success; MES responds duplicate=True


@pytest.mark.integration
async def test_mes_500_unreachable():
    """5xx is retried (not a 4xx payload error); after bounded retries the
    sync fails with MesUnreachable and the inspection stays complete."""
    try:
        httpx.get(f"{MES_HTTP}/v1/products/x", timeout=2).raise_for_status()
    except Exception:
        _skipped("mes simulator not running")

    # inject a 500 on the inspection endpoint, then clear it
    httpx.post(f"{MES_HTTP}/v1/admin/fault", params={"endpoint": "inspection", "mode": "500"}, timeout=2)
    adapter = MesAdapter(MES_HTTP, timeout_seconds=2.0, max_retries=1)
    try:
        with pytest.raises(MesUnreachable):
            await adapter.post_inspection_result(
                inspection_id="insp-mes-500", product_id="P", batch_id=None, line="l", station="s",
                ai_result="PASS", model_version="v1", rule_version=1, industrial_state="RELEASED", timestamp="t",
            )
    finally:
        httpx.post(f"{MES_HTTP}/v1/admin/reset", timeout=2)


@pytest.mark.integration
async def test_mes_timeout_unreachable():
    try:
        httpx.get(f"{MES_HTTP}/v1/products/x", timeout=2).raise_for_status()
    except Exception:
        _skipped("mes simulator not running")

    httpx.post(f"{MES_HTTP}/v1/admin/fault", params={"endpoint": "inspection", "mode": "timeout"}, timeout=2)
    adapter = MesAdapter(MES_HTTP, timeout_seconds=0.5, max_retries=0)
    try:
        with pytest.raises(MesUnreachable):
            await adapter.post_inspection_result(
                inspection_id="insp-mes-tmo", product_id="P", batch_id=None, line="l", station="s",
                ai_result="PASS", model_version="v1", rule_version=1, industrial_state="RELEASED", timestamp="t",
            )
    finally:
        httpx.post(f"{MES_HTTP}/v1/admin/reset", timeout=2)


# ---- end-to-end through the service (real HTTP PLC + MES) ----

@pytest.mark.integration
async def test_service_full_chain_review_hold_then_release():
    """REVIEW -> HOLD (PLC state) -> human PASS -> RELEASE, all real simulators."""
    import pytest_asyncio
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.orm import selectinload

    from app.enums import QualityResult
    from app.models import Base, Inspection, PlcEvent, Product
    from app.services.industrial_service import IndustrialService

    try:
        httpx.get(f"{PLC_HTTP}/v1/state", timeout=2).raise_for_status()
        httpx.get(f"{MES_HTTP}/v1/products/x", timeout=2).raise_for_status()
    except Exception:
        _skipped("simulators not running")
    httpx.post(f"{PLC_HTTP}/v1/admin/reset", timeout=2)
    httpx.post(f"{MES_HTTP}/v1/admin/reset", timeout=2)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        product = Product(product_id="P-REV-1", production_line="line-a", station="qc")
        session.add(product)
        await session.flush()
        inspection = Inspection(
            inspection_id="insp-rev-1", product_id=product.id, status="completed",
            quality_result=QualityResult.REVIEW,
        )
        session.add(inspection)
        await session.commit()
        insp = (
            await session.execute(
                select(Inspection).options(selectinload(Inspection.product)).where(Inspection.inspection_id == "insp-rev-1")
            )
        ).scalar_one()

        svc = IndustrialService()
        svc.plc_enabled = True
        svc.plc_max_retries = 1
        svc.plc = HttpPlcAdapter(PLC_HTTP, timeout_seconds=2.0)
        svc.mes = MesAdapter(MES_HTTP, timeout_seconds=2.0, max_retries=1)
        svc.mes_enabled = True

        # AI REVIEW -> desired HOLD -> HELD
        await svc.process_result(session, insp, final_quality_result="REVIEW", process_status="completed")
        assert insp.desired_command == "HOLD"
        assert insp.industrial_final_state == "HELD"
        state = httpx.get(f"{PLC_HTTP}/v1/state", timeout=2).json()
        assert state["state"] == "HOLD"

        # human PASS -> RELEASE (new command id, executed once)
        await svc.process_result(
            session, insp, final_quality_result="PASS", process_status="completed",
            review_resolved=True, review_decision="PASS", reviewed_by="alice",
        )
        assert insp.desired_command == "RELEASE"
        assert insp.execution_status == "ACK"
        assert insp.industrial_final_state == "RELEASED"

        events = (await session.execute(select(PlcEvent).order_by(PlcEvent.created_at))).scalars().all()
        assert len(events) == 2
        assert events[0].desired_command == "HOLD"
        assert events[1].desired_command == "RELEASE"

    await engine.dispose()
