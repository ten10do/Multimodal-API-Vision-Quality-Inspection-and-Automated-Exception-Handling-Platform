"""Phase 6 pre-gate: REAL concurrent claim on Docker PostgreSQL.

Two independent DB sessions claim the same PENDING review task at the same
time. Exactly one succeeds; the other must observe the committed state and
fail with already_claimed.

Requires the Docker PostgreSQL container on :5433 with the schema migrated
(alembic upgrade head). Marked ``integration``.

Unlike the in-memory SQLite unit test (single shared session), this test uses
two separate connections to the same Postgres database, which is the actual
concurrency boundary of production.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

DB_URL = "postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5433/industrialvision_test"

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.enums import HumanDecision, InspectionStatus, QualityResult, ReviewTaskStatus  # noqa: E402
from app.models import Inspection, Product, ReviewDecision, ReviewTask  # noqa: E402
from app.services.review_service import ReviewConflictError, ReviewService  # noqa: E402


@pytest_asyncio.fixture
async def engine():
    """Function-scoped engine: avoids Windows ProactorEventLoop teardown
    races with pooled asyncpg connections."""
    eng = create_async_engine(DB_URL, pool_size=3, max_overflow=3)
    yield eng
    await asyncio.sleep(0.05)
    await eng.dispose()


async def _seed_task(engine) -> str:
    """Insert an inspection (REVIEW) + PENDING review task; return task id."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        product = Product(product_id=f"pg-{uuid.uuid4().hex[:8]}", production_line="line-pg", station="qc-01")
        s.add(product)
        await s.flush()
        inspection = Inspection(
            inspection_id=f"insp-{uuid.uuid4().hex[:12]}",
            product_id=product.id,
            status=InspectionStatus.COMPLETED,
            quality_result=QualityResult.REVIEW,
            model_version="v1",
            rule_version=1,
        )
        s.add(inspection)
        await s.flush()
        task = ReviewTask(
            review_task_id=f"rt-{uuid.uuid4().hex[:12]}",
            inspection_id=inspection.id,
            status=ReviewTaskStatus.PENDING,
            priority=200,
            version=1,
            ai_quality_result="REVIEW",
            ai_defects_snapshot=[{"class_name": "crazing", "confidence": 0.42}],
            ai_model_version="v1",
            ai_rule_version=1,
            ai_severity="medium",
            product_id=product.product_id,
            production_line="line-pg",
            station="qc-01",
        )
        s.add(task)
        await s.commit()
        return task.review_task_id


@pytest.mark.asyncio
async def test_real_concurrent_claim_pg(engine):
    """Two independent sessions race on the same task; exactly one wins."""
    task_id = await _seed_task(engine)
    service = ReviewService()
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def claim(reviewer: str):
        async with factory() as session:
            return await service.claim(session, task_id, reviewer)

    results = await asyncio.gather(
        claim("alice-pg"),
        claim("bob-pg"),
        return_exceptions=True,
    )

    ok = [r for r in results if isinstance(r, ReviewTask)]
    conflicts = [r for r in results if isinstance(r, ReviewConflictError)]
    assert len(ok) == 1, f"exactly one claim must win, got {len(ok)}"
    assert len(conflicts) == 1, f"exactly one conflict expected, got {len(conflicts)}"
    winner = ok[0]
    loser = conflicts[0]
    assert winner.status == ReviewTaskStatus.IN_REVIEW
    assert winner.assigned_to in ("alice-pg", "bob-pg")
    assert loser.code == "already_claimed"
    assert winner.version == 2

    # the loser session sees the committed winner (no stale state)
    async with factory() as s:
        task = (await s.execute(select(ReviewTask).where(ReviewTask.review_task_id == task_id))).scalar_one()
        assert task.status == ReviewTaskStatus.IN_REVIEW
        assert task.assigned_to == winner.assigned_to


@pytest.mark.asyncio
async def test_concurrent_resolve_only_owner_pg(engine):
    """After a claim, only the owner can resolve; a second owner cannot
    double-resolve (optimistic version guard)."""
    task_id = await _seed_task(engine)
    service = ReviewService()
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as s:
        await service.claim(s, task_id, "alice-pg")

    async with factory() as s:
        r = await service.resolve(s, task_id, "alice-pg", HumanDecision.CONFIRM_DEFECT, "crazing", "visible")
        assert r.status == ReviewTaskStatus.RESOLVED

    # double resolve -> conflict, original decision preserved
    with pytest.raises(ReviewConflictError) as exc:
        async with factory() as s:
            await service.resolve(s, task_id, "alice-pg", HumanDecision.PASS, None, None)
    assert exc.value.code == "already_resolved"

    async with factory() as s:
        task_row = (await s.execute(select(ReviewTask).where(ReviewTask.review_task_id == task_id))).scalar_one()
        decisions = (
            await s.execute(select(ReviewDecision).where(ReviewDecision.review_task_id == task_row.id))
        ).scalars().all()
        assert len(decisions) == 1
        assert decisions[0].human_decision == HumanDecision.CONFIRM_DEFECT
        assert decisions[0].final_quality_result == QualityResult.FAIL
