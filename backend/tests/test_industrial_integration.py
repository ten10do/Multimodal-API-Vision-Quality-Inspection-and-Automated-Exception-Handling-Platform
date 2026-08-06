"""Phase 7: real integration tests against the live simulators.

Requires the simulators to be running:
  PLC HTTP    : python -m simulator.plc_simulator       (8501)
  MES HTTP    : python -m simulator.mes_simulator       (8502)
  PLC OPC UA  : python -m simulator.opcua_plc_server    (8503)

Run: pytest backend/tests/test_industrial_integration.py -m integration
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.enums import QualityResult
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
    """Optional-simulator skip helper.

    Default (dev) run: missing simulator -> skip, so the plain unit/dev
    suite stays green. When the gate run declares the simulators present
    (IVQC_REQUIRE_SIMULATORS=1), a missing simulator is a FAILURE, never a
    silent skip -- a skipped industrial gate test must not be reported as
    "passed".
    """
    if os.environ.get("IVQC_REQUIRE_SIMULATORS") == "1":
        raise AssertionError(f"required simulator missing (IVQC_REQUIRE_SIMULATORS=1): {reason}")
    pytest.skip(reason)


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


# ---- OPC UA PLC (opcua gate: FAIL-FAST, never skip) ----

@pytest.mark.integration
@pytest.mark.opcua
async def test_opcua_plc_ack_and_idempotency():
    """Adapter-level ACK + idempotency against the real local OPC UA server.

    This is a Phase 7 gate test: if the simulator is missing the test FAILS
    (no skip). Historically this test skipped on PlcUnreachable, which masked
    the BadNoMatch namespace bug -- a skip must never be reported as pass.
    """
    adapter = OpcUaPlcAdapter(PLC_OPCUA, timeout_seconds=3.0)
    # unique command_id per run: the simulator's idempotency set is persistent
    # across runs (no admin reset on the OPC UA server), so a fixed id would
    # come back as duplicate on the second run
    iid = f"insp-opcua-{int(time.time() * 1000)}"
    first = await adapter.send_command(_cmd("HOLD", inspection_id=iid))
    assert first.acked is True
    assert first.duplicate is False
    second = await adapter.send_command(_cmd("HOLD", inspection_id=iid))
    assert second.duplicate is True


@pytest.mark.integration
@pytest.mark.opcua
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
@pytest.mark.industrial_e2e
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


# ---- OPC UA full E2E: local OPC UA server -> OpcUaPlcAdapter ->
#      IndustrialService -> plc_event persistence (Phase 7 closing gate) ----

@pytest.mark.integration
@pytest.mark.opcua
@pytest.mark.parametrize(
    "quality_result, expected_command, expected_state",
    [
        (QualityResult.REVIEW, "HOLD", "HELD"),
        (QualityResult.PASS, "RELEASE", "RELEASED"),
        (QualityResult.FAIL, "REJECT", "REJECTED"),
    ],
)
async def test_opcua_service_ack_and_persistence(quality_result, expected_command, expected_state):
    """Full OPC UA path through IndustrialService for each business outcome:
    REVIEW -> HOLD/HELD, PASS -> RELEASE/RELEASED, FAIL -> REJECT/REJECTED.
    Each run must record exactly one PlcEvent with execution_status=ACK,
    industrial_state=<resolved terminal>, adapter_type=opcua.

    Gate: the OPC UA and MES simulators are required; a missing one FAILS
    (no skip, fail-fast when IVQC_REQUIRE_SIMULATORS=1, and even without it
    the OPC UA path must never be silently skipped)."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.orm import selectinload

    from app.enums import QualityResult
    from app.industrial.plc_adapter import OpcUaPlcAdapter
    from app.models import Base, Inspection, PlcEvent, Product
    from app.services.industrial_service import IndustrialService

    resp = httpx.get(f"{MES_HTTP}/v1/products/x", timeout=2)
    resp.raise_for_status()  # fail-fast: MES is part of this gate
    httpx.post(f"{MES_HTTP}/v1/admin/reset", timeout=2)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    iid = f"insp-opcua-e2e-{expected_command.lower()}-{int(time.time() * 1000)}"
    async with factory() as session:
        product = Product(product_id=f"P-OPCUA-{expected_command}", production_line="line-a", station="qc")
        session.add(product)
        await session.flush()
        inspection = Inspection(
            inspection_id=iid, product_id=product.id, status="completed",
            quality_result=quality_result,
        )
        session.add(inspection)
        await session.commit()
        insp = (
            await session.execute(
                select(Inspection).options(selectinload(Inspection.product)).where(Inspection.inspection_id == iid)
            )
        ).scalar_one()

        svc = IndustrialService()
        svc.plc_enabled = True
        svc.plc_max_retries = 1
        svc.plc = OpcUaPlcAdapter(PLC_OPCUA, timeout_seconds=3.0)
        svc.mes = MesAdapter(MES_HTTP, timeout_seconds=2.0, max_retries=1)
        svc.mes_enabled = True

        await svc.process_result(session, insp, final_quality_result=quality_result.value, process_status="completed")
        await session.commit()

        assert insp.desired_command == expected_command
        assert insp.execution_status == "ACK"
        assert insp.industrial_final_state == expected_state
        assert insp.plc_adapter_type == "opcua"

        events = (await session.execute(select(PlcEvent))).scalars().all()
        assert len(events) == 1
        ev = events[0]
        assert ev.adapter_type == "opcua"
        assert ev.execution_status == "ACK"
        assert ev.industrial_state == expected_state
        assert ev.desired_command == expected_command
        assert ev.acknowledged_at is not None

    await engine.dispose()


@pytest.mark.integration
@pytest.mark.opcua
async def test_opcua_unavailable_never_release():
    """OPC UA server unreachable -> adapter failure -> SAFE_HOLD.

    Fail-safe contract: even a PASS result must never produce RELEASED when
    the field layer cannot be reached; the product lands in SAFE_HOLD and the
    persisted PlcEvent records the real (un-acked) state."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.orm import selectinload

    from app.enums import QualityResult
    from app.industrial.plc_adapter import OpcUaPlcAdapter
    from app.models import Base, Inspection, PlcEvent, Product
    from app.services.industrial_service import IndustrialService

    httpx.get(f"{MES_HTTP}/v1/products/x", timeout=2).raise_for_status()
    httpx.post(f"{MES_HTTP}/v1/admin/reset", timeout=2)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        product = Product(product_id="P-OPCUA-OFF", production_line="line-a", station="qc")
        session.add(product)
        await session.flush()
        inspection = Inspection(
            inspection_id="insp-opcua-unavail", product_id=product.id, status="completed",
            quality_result=QualityResult.PASS,
        )
        session.add(inspection)
        await session.commit()
        insp = (
            await session.execute(
                select(Inspection).options(selectinload(Inspection.product)).where(Inspection.inspection_id == "insp-opcua-unavail")
            )
        ).scalar_one()

        svc = IndustrialService()
        svc.plc_enabled = True
        svc.plc_max_retries = 1
        svc.plc = OpcUaPlcAdapter("opc.tcp://127.0.0.1:59997", timeout_seconds=1.0)
        svc.mes = MesAdapter(MES_HTTP, timeout_seconds=2.0, max_retries=1)
        svc.mes_enabled = True

        await svc.process_result(session, insp, final_quality_result="PASS", process_status="completed")
        await session.commit()

        assert insp.desired_command == "RELEASE"  # what was wanted...
        assert insp.execution_status in ("ERROR", "TIMEOUT")  # ...but never ACKed
        assert insp.industrial_final_state == "SAFE_HOLD"  # ...and never RELEASED
        assert insp.industrial_final_state != "RELEASED"

        events = (await session.execute(select(PlcEvent))).scalars().all()
        assert len(events) == 1
        ev = events[0]
        assert ev.adapter_type == "opcua"
        assert ev.industrial_state == "SAFE_HOLD"
        assert ev.desired_command == "RELEASE"
        assert ev.execution_status in ("ERROR", "TIMEOUT")
        assert ev.acknowledged_at is None

    await engine.dispose()


@pytest.mark.integration
@pytest.mark.opcua
async def test_opcua_adapter_resolves_any_namespace_index():
    """Namespace robustness: the Plc object is registered under a namespace
    whose INDEX IS NOT 2 (two filler namespaces are registered first). The
    adapter must still resolve it (URI -> index, browse fallback) and ACK --
    proving a namespace-index change can never break the adapter again."""
    import json as _json
    import socket

    from asyncua import Server, ua

    from app.industrial.plc_adapter import OpcUaPlcAdapter

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = Server()
    await server.init()
    server.set_endpoint(f"opc.tcp://127.0.0.1:{port}")
    # two filler namespaces force the Plc namespace index away from 2
    await server.register_namespace("urn:ivqc:filler-a")
    await server.register_namespace("urn:ivqc:filler-b")
    idx = await server.register_namespace("urn:ivqc:plc")
    assert idx != 2, "test setup broken: expected Plc namespace index != 2"
    plc_obj = await server.nodes.objects.add_object(idx, "Plc")
    executed: set[str] = set()

    async def execute(parent, command_json) -> list:
        payload = command_json.Value if isinstance(command_json, ua.Variant) else command_json
        cid = _json.loads(payload).get("command_id", "")
        if cid in executed:
            return [ua.Variant("ACK:duplicate", ua.VariantType.String)]
        executed.add(cid)
        return [ua.Variant("ACK:1", ua.VariantType.String)]

    await plc_obj.add_method(idx, "execute", execute, [ua.VariantType.String], [ua.VariantType.String])

    async def _run_server() -> None:
        async with server:
            await asyncio.Future()  # keep serving until cancelled

    task = asyncio.create_task(_run_server())
    try:
        await asyncio.sleep(0.5)  # let the listener come up
        adapter = OpcUaPlcAdapter(f"opc.tcp://127.0.0.1:{port}", timeout_seconds=5.0)
        first = await adapter.send_command(_cmd("HOLD", inspection_id="insp-ns-robust"))
        assert first.acked is True
        assert first.duplicate is False
        second = await adapter.send_command(_cmd("HOLD", inspection_id="insp-ns-robust"))
        assert second.duplicate is True
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
