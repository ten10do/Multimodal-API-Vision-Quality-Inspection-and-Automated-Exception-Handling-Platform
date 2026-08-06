"""Phase 7: industrial semantics (10/11/12).

The three-layer model is tested directly:
  desired_command  = what the system wants the field layer to do
  execution_status = whether the command was really sent / ACKed
  industrial_state = the product's actual field state

plc_enabled=False MUST record NOT_INTEGRATED and NEVER fabricate
SAFE_HOLD / HELD / RELEASED / REJECTED. The adapter must never be called.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.industrial.commands import IndustrialCommand
from app.industrial.plc_adapter import PlcCommandResult, PlcNack, PlcUnreachable
from app.enums import QualityResult
from app.models import Base, Inspection, PlcEvent, Product
from app.services.industrial_service import IndustrialService


class FakePlc:
    """Adapter double: records every send_command call."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[IndustrialCommand] = []
        self.result: PlcCommandResult = PlcCommandResult(acked=True, response={"ack": "ACK"}, latency_ms=3.0)
        self.exc: Exception | None = None
        self.duplicate = False

    async def send_command(self, command: IndustrialCommand) -> PlcCommandResult:
        self.calls.append(command)
        if self.exc is not None:
            raise self.exc
        return PlcCommandResult(
            acked=True,
            response={"ack": "ACK"},
            latency_ms=3.0,
            duplicate=self.duplicate,
        )


class FakeMes:
    """Adapter double: records payloads; can be told to fail."""

    def __init__(self) -> None:
        self.inspection_payloads: list[dict] = []
        self.final_payloads: list[dict] = []
        self.fail = False

    async def post_inspection_result(self, **kw) -> tuple[bool, float, int]:
        if self.fail:
            raise RuntimeError("mes down")
        self.inspection_payloads.append(kw)
        return True, 1.0, 1

    async def post_final_quality_result(self, **kw) -> tuple[bool, float, int]:
        if self.fail:
            raise RuntimeError("mes down")
        self.final_payloads.append(kw)
        return True, 1.0, 1


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session, factory
    await engine.dispose()


async def _make_inspection(db, quality: str | None = "PASS", status: str = "completed") -> Inspection:
    session, factory = db
    product = Product(product_id="P-1", production_line="line-a", station="qc-01")
    session.add(product)
    await session.flush()
    qr = QualityResult(quality) if quality else None
    inspection = Inspection(
        inspection_id="insp-1",
        product_id=product.id,
        status=status,
        quality_result=qr,
    )
    session.add(inspection)
    await session.commit()
    fresh = (
        await session.execute(
            select(Inspection)
            .options(selectinload(Inspection.product))
            .where(Inspection.inspection_id == "insp-1")
        )
    ).scalar_one()
    return fresh


def _service(plc: FakePlc, mes: FakeMes, *, plc_enabled: bool) -> IndustrialService:
    svc = IndustrialService()
    svc.plc_enabled = plc_enabled
    svc.plc_max_retries = 2
    svc.plc = plc
    svc.mes = mes
    svc.mes_enabled = True
    return svc


# ---- 10. plc_enabled=False: NOT_INTEGRATED, adapter never called ----

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quality,process_status,expected_desired,expected_reason",
    [
        ("PASS", "completed", "RELEASE", "quality_pass"),
        ("FAIL", "completed", "REJECT", "product_defect"),
        ("REVIEW", "completed", "HOLD", "review_pending"),
        (None, "failed", "HOLD", "system_failed"),
    ],
)
async def test_plc_disabled_not_integrated(db, quality, process_status, expected_desired, expected_reason):
    session, _ = db
    inspection = await _make_inspection(db, quality=quality, status=process_status)
    plc = FakePlc()
    mes = FakeMes()
    svc = _service(plc, mes, plc_enabled=False)

    await svc.process_result(
        session, inspection,
        final_quality_result=quality,
        process_status=process_status,
    )
    await session.commit()

    # adapter was NEVER invoked
    assert plc.calls == [], "the PLC adapter must not be called when integration is disabled"
    # three-layer semantics
    assert inspection.desired_command == expected_desired
    assert inspection.execution_status == "NOT_INTEGRATED"
    assert inspection.industrial_state == "NOT_INTEGRATED"
    assert inspection.industrial_final_state == "NOT_INTEGRATED"
    assert inspection.plc_status == "NOT_INTEGRATED"
    # forbidden fabricated states
    assert inspection.industrial_final_state not in ("RELEASED", "REJECTED", "HELD", "SAFE_HOLD")

    events = (await session.execute(select(PlcEvent))).scalars().all()
    assert len(events) == 1
    ev = events[0]
    assert ev.status == "NOT_INTEGRATED"
    assert ev.execution_status == "NOT_INTEGRATED"
    assert ev.industrial_state == "NOT_INTEGRATED"
    assert ev.reason_code == "plc_integration_disabled"
    assert ev.desired_command == expected_desired
    assert ev.adapter_type == "none"
    # the business reason is preserved in the request payload
    assert ev.request_payload["reason_code"] == expected_reason


@pytest.mark.asyncio
async def test_plc_disabled_no_fake_ack_no_acknowledged_at(db):
    session, _ = db
    inspection = await _make_inspection(db, quality="FAIL", status="completed")
    svc = _service(FakePlc(), FakeMes(), plc_enabled=False)

    await svc.process_result(session, inspection, final_quality_result="FAIL", process_status="completed")
    await session.commit()

    ev = (await session.execute(select(PlcEvent))).scalar_one()
    assert ev.status == "NOT_INTEGRATED"
    assert ev.acknowledged_at is None
    assert ev.latency_ms is None


# ---- 11. plc_enabled=True: real command flow ----

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quality,process_status,review_resolved,review_decision,expected_cmd,expected_final,expected_reason",
    [
        ("PASS", "completed", False, None, "RELEASE", "RELEASED", "quality_pass"),
        ("FAIL", "completed", False, None, "REJECT", "REJECTED", "product_defect"),
        ("REVIEW", "completed", False, None, "HOLD", "HELD", "review_pending"),
        (None, "failed", False, None, "HOLD", "SAFE_HOLD", "system_failed"),
        ("PASS", "completed", True, "PASS", "RELEASE", "RELEASED", "review_pass"),
        ("FAIL", "completed", True, "FAIL", "REJECT", "REJECTED", "review_fail"),
    ],
)
async def test_plc_enabled_ack(db, quality, process_status, review_resolved, review_decision,
                               expected_cmd, expected_final, expected_reason):
    session, _ = db
    inspection = await _make_inspection(db, quality=quality, status=process_status)
    plc = FakePlc()
    mes = FakeMes()
    svc = _service(plc, mes, plc_enabled=True)

    await svc.process_result(
        session, inspection,
        final_quality_result=quality,
        process_status=process_status,
        review_resolved=review_resolved,
        review_decision=review_decision,
        reviewed_by="alice",
    )
    await session.commit()

    assert len(plc.calls) == 1
    assert plc.calls[0].command_type == expected_cmd
    assert plc.calls[0].reason_code == expected_reason
    assert inspection.execution_status == "ACK"
    assert inspection.industrial_final_state == expected_final
    assert inspection.plc_latency_ms == 3.0

    ev = (await session.execute(select(PlcEvent))).scalar_one()
    assert ev.execution_status == "ACK"
    assert ev.industrial_state == expected_final
    assert ev.acknowledged_at is not None
    assert ev.adapter_type == "fake"
    assert ev.reason_code == expected_reason


@pytest.mark.asyncio
async def test_plc_timeout_safe_hold_never_release(db):
    session, _ = db
    inspection = await _make_inspection(db, quality="PASS", status="completed")
    plc = FakePlc()
    plc.exc = PlcUnreachable("plc timeout after 2s")
    mes = FakeMes()
    svc = _service(plc, mes, plc_enabled=True)

    await svc.process_result(session, inspection, final_quality_result="PASS", process_status="completed")
    await session.commit()

    # PASS wanted RELEASE, but timeout must never default to RELEASE
    assert inspection.desired_command == "RELEASE"
    assert inspection.execution_status == "TIMEOUT"
    assert inspection.industrial_final_state == "SAFE_HOLD"
    assert plc.calls != []


@pytest.mark.asyncio
async def test_plc_offline_safe_hold(db):
    session, _ = db
    inspection = await _make_inspection(db, quality="PASS", status="completed")
    plc = FakePlc()
    plc.exc = PlcUnreachable("plc unreachable: connection refused")
    svc = _service(plc, FakeMes(), plc_enabled=True)

    await svc.process_result(session, inspection, final_quality_result="PASS", process_status="completed")
    await session.commit()

    assert inspection.execution_status == "ERROR"
    assert inspection.industrial_final_state == "SAFE_HOLD"


@pytest.mark.asyncio
async def test_plc_nack_command_failed(db):
    session, _ = db
    inspection = await _make_inspection(db, quality="FAIL", status="completed")
    plc = FakePlc()
    plc.exc = PlcNack("fault: servo error")
    svc = _service(plc, FakeMes(), plc_enabled=True)

    await svc.process_result(session, inspection, final_quality_result="FAIL", process_status="completed")
    await session.commit()

    assert inspection.execution_status == "NACK"
    assert inspection.industrial_final_state == "COMMAND_FAILED"


@pytest.mark.asyncio
async def test_plc_duplicate_suppressed(db):
    session, _ = db
    inspection = await _make_inspection(db, quality="PASS", status="completed")
    plc = FakePlc()
    plc.duplicate = True
    svc = _service(plc, FakeMes(), plc_enabled=True)

    await svc.process_result(session, inspection, final_quality_result="PASS", process_status="completed")
    await session.commit()

    assert inspection.execution_status == "DUPLICATE_SUPPRESSED"
    assert inspection.industrial_final_state == "RELEASED"
    ev = (await session.execute(select(PlcEvent))).scalar_one()
    assert ev.status == "DUPLICATE_SUPPRESSED"


# ---- 12. MES and PLC are decoupled ----

@pytest.mark.asyncio
async def test_plc_release_ok_mes_fail_keeps_released(db):
    session, _ = db
    inspection = await _make_inspection(db, quality="PASS", status="completed")
    plc = FakePlc()
    mes = FakeMes()
    mes.fail = True
    svc = _service(plc, mes, plc_enabled=True)

    await svc.process_result(session, inspection, final_quality_result="PASS", process_status="completed")
    await session.commit()

    # MES failure must not rewrite a real PLC state
    assert inspection.industrial_final_state == "RELEASED"
    assert inspection.mes_sync_status == "FAILED"


@pytest.mark.asyncio
async def test_mes_sees_real_state_safe_hold_not_released(db):
    session, _ = db
    inspection = await _make_inspection(db, quality="PASS", status="completed")
    plc = FakePlc()
    plc.exc = PlcUnreachable("plc unreachable")
    mes = FakeMes()
    svc = _service(plc, mes, plc_enabled=True)

    await svc.process_result(session, inspection, final_quality_result="PASS", process_status="completed")
    await session.commit()

    # MES succeeded but must have received SAFE_HOLD, never RELEASED
    assert inspection.industrial_final_state == "SAFE_HOLD"
    assert inspection.mes_sync_status == "SYNCED"
    assert mes.inspection_payloads, "mes should have received the inspection result"
    assert mes.inspection_payloads[-1]["industrial_state"] == "SAFE_HOLD"
    assert mes.final_payloads[-1]["industrial_state"] == "SAFE_HOLD"


@pytest.mark.asyncio
async def test_mes_disabled_records_pending(db):
    session, _ = db
    inspection = await _make_inspection(db, quality="PASS", status="completed")
    svc = _service(FakePlc(), FakeMes(), plc_enabled=True)
    svc.mes_enabled = False

    await svc.process_result(session, inspection, final_quality_result="PASS", process_status="completed")
    await session.commit()

    assert inspection.industrial_final_state == "RELEASED"
    assert inspection.mes_sync_status == "PENDING"
