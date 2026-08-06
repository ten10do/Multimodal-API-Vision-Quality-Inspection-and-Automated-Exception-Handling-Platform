"""Deterministic demo seed (10F).

Idempotent: safe to re-run. Seeds a stable, named demo scenario set so the
demo does NOT depend on whatever happens to be in a scratch database:

  - known defect     : product DEMO-KNOWN -> FAIL (scratches) -> PLC REJECTED
  - unknown anomaly  : product DEMO-ANOM  -> REVIEW (anomaly_score) -> PENDING review
  - human review     : DEMO-REV -> REVIEW -> human CONFIRM_DEFECT (audited)
  - PLC hold/release : DEMO-HOLD -> REVIEW -> HOLD ACK -> human PASS -> RELEASE
  - monitoring       : 7 days of inspections with latency + anomaly scores
                       (gives Model Ops metrics / drift / Copilot real data)

HONESTY: these are integration fixtures, NOT production data. The
anomaly_score / industrial states are authored directly to demonstrate the
UI + pipeline states; they are never claimed to come from real steel-domain
PatchCore inference (see docs/09-phase6-report.md).

Usage:  IVQC_DATABASE_URL=postgresql+asyncpg://vision_qc:vision_qc@127.0.0.1:5433/industrialvision_test \
        bash scripts/run_clean.sh python scripts/demo_seed.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker  # noqa: E402

from app.database import engine  # noqa: E402
from app.enums import HumanDecision, QualityResult, ReviewTaskStatus  # noqa: E402
from app.models import (  # noqa: E402
    Defect,
    Inspection,
    PlcEvent,
    Product,
    ReviewDecision,
    ReviewTask,
)

BATCH = "demo-seed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _existing(session, model, **kwargs):
    return (await session.execute(select(model).filter_by(**kwargs))).scalar_one_or_none()


async def _product(session, pid: str, line: str, station: str) -> Product:
    p = await _existing(session, Product, product_id=pid)
    if p is None:
        p = Product(product_id=pid, production_line=line, station=station)
        session.add(p)
        await session.flush()
    return p


async def _inspection(session, iid: str, product: Product, *, quality: QualityResult, created: datetime,
                      batch: str | None = BATCH, latency: float = 120.0, anomaly: float | None = None,
                      model_version: str = "phase1-baseline", deployment: str = "2026.08.1",
                      final: QualityResult | None = None) -> Inspection:
    insp = await _existing(session, Inspection, inspection_id=iid)
    if insp is None:
        insp = Inspection(
            inspection_id=iid, product_id=product.id, batch_id=batch, status="completed",
            quality_result=quality, final_quality_result=final, model_name="yolov8s",
            model_version=model_version, deployment_version=deployment, rule_version=1,
            inference_latency_ms=latency, anomaly_score=anomaly, created_at=created,
        )
        session.add(insp)
        await session.flush()
    return insp


async def _defect(session, insp: Inspection, class_name: str, confidence: float) -> None:
    existing = await _existing(session, Defect, inspection_id=insp.id, class_name=class_name)
    if existing is None:
        session.add(
            Defect(
                inspection_id=insp.id, class_id=1, class_name=class_name, confidence=confidence,
                bbox_xyxy=[10, 10, 60, 60], bbox_normalized=[0.05, 0.05, 0.3, 0.3],
                defect_area_px=500.0, defect_area_ratio=0.02, matched_rule="r1",
                created_at=_now(),
            )
        )


async def _plc_event(session, iid: str, product_id: str, *, command: str, state: str, status: str,
                     adapter: str = "http", created: datetime | None = None) -> None:
    e = await _existing(session, PlcEvent, inspection_id=iid, command=command)
    if e is None:
        session.add(
            PlcEvent(
                command_id=f"demo-{iid}-{command}", product_id=product_id, inspection_id=iid,
                command=command, desired_command=command, execution_status=status,
                industrial_state=state, adapter_type=adapter, request_payload={},
                response={"ack": "ACK"}, status=status, retry_count=0, latency_ms=10.0,
                reason_code="product_defect", created_at=created or _now(),
            )
        )


async def _review_task(session, insp: Inspection, *, status: ReviewTaskStatus, decision: HumanDecision | None = None,
                       label: str | None = None, reason: str | None = None, reviewer: str = "demo-qc",
                       anomaly_score: float | None = None) -> None:
    t = await _existing(session, ReviewTask, inspection_id=insp.id)
    if t is None:
        t = ReviewTask(
            review_task_id=f"demo-{insp.inspection_id}", inspection_id=insp.id, status=status,
            ai_quality_result="REVIEW",
            ai_defects_snapshot=[{"class_name": "crazing", "confidence": 0.42}],
            ai_model_version="phase1-baseline", ai_rule_version=1, product_id="DEMO-ANOM",
            production_line="line-a", station="qc-01", batch_id=BATCH, anomaly_score=anomaly_score,
            created_at=_now(),
        )
        session.add(t)
        await session.flush()
    if decision is not None:
        from app.models import ReviewDecision as RD

        existing_dec = (await session.execute(select(RD).where(RD.review_task_id == t.id))).scalar_one_or_none()
        if existing_dec is None:
            session.add(
                ReviewDecision(
                    review_task_id=t.id, inspection_id=insp.id, reviewer=reviewer,
                    ai_quality_result="REVIEW", final_quality_result="PASS" if decision == HumanDecision.PASS else "FAIL",
                    human_decision=decision, human_label=label, reason=reason, created_at=_now(),
                )
            )


async def main() -> int:
    created = 0
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        now = _now()

        # 1. known defect -> FAIL -> REJECT
        p = await _product(session, "DEMO-KNOWN", "line-a", "qc-01")
        insp = await _inspection(session, "demo-known-defect", p, quality=QualityResult.FAIL,
                                 created=now - timedelta(hours=2), final=QualityResult.FAIL,
                                 latency=132.0, anomaly=0.87)
        await _defect(session, insp, "scratches", 0.87)
        insp.desired_command = "REJECT"; insp.execution_status = "ACK"
        insp.industrial_final_state = "REJECTED"; insp.plc_adapter_type = "http"
        insp.mes_sync_status = "SYNCED"
        await _plc_event(session, "demo-known-defect", "DEMO-KNOWN", command="REJECT",
                         state="REJECTED", status="ACK", created=now - timedelta(hours=2))
        created += 1

        # 2. unknown anomaly -> REVIEW (PENDING human)
        p2 = await _product(session, "DEMO-ANOM", "line-b", "qc-03")
        insp2 = await _inspection(session, "demo-unknown-anomaly", p2, quality=QualityResult.REVIEW,
                                  created=now - timedelta(hours=1), latency=145.0, anomaly=0.95)
        await _review_task(session, insp2, status=ReviewTaskStatus.PENDING, anomaly_score=0.95)
        insp2.desired_command = "HOLD"; insp2.execution_status = "ACK"
        insp2.industrial_final_state = "HELD"; insp2.plc_adapter_type = "http"
        insp2.mes_sync_status = "SYNCED"
        await _plc_event(session, "demo-unknown-anomaly", "DEMO-ANOM", command="HOLD",
                         state="HELD", status="ACK", created=now - timedelta(hours=1))
        created += 1

        # 3. human review resolved (CONFIRM_DEFECT)
        p3 = await _product(session, "DEMO-REV", "line-a", "qc-02")
        insp3 = await _inspection(session, "demo-review-resolved", p3, quality=QualityResult.REVIEW,
                                  created=now - timedelta(days=1), latency=150.0, anomaly=0.9)
        await _review_task(session, insp3, status=ReviewTaskStatus.RESOLVED,
                           decision=HumanDecision.CONFIRM_DEFECT, label="crazing",
                           reason="demo: confirmed defect")
        created += 1

        # 4. PLC hold -> human PASS -> release (two events, one inspection)
        p4 = await _product(session, "DEMO-HOLD", "line-a", "qc-01")
        insp4 = await _inspection(session, "demo-hold-release", p4, quality=QualityResult.REVIEW,
                                  created=now - timedelta(hours=6), final=QualityResult.PASS)
        await _review_task(session, insp4, status=ReviewTaskStatus.RESOLVED,
                           decision=HumanDecision.PASS, reason="demo: human pass")
        insp4.desired_command = "RELEASE"; insp4.execution_status = "ACK"
        insp4.industrial_final_state = "RELEASED"; insp4.plc_adapter_type = "http"
        insp4.mes_sync_status = "SYNCED"
        await _plc_event(session, "demo-hold-release", "DEMO-HOLD", command="HOLD",
                         state="HELD", status="ACK", created=now - timedelta(hours=6))
        await _plc_event(session, "demo-hold-release", "DEMO-HOLD", command="RELEASE",
                         state="RELEASED", status="ACK", created=now - timedelta(hours=6))
        created += 1

        # 5. monitoring spread: last 7 days (for Model Ops / drift / Copilot)
        for day in range(7, 0, -1):
            base = now - timedelta(days=day)
            for idx, (line, station) in enumerate((("line-a", "qc-01"), ("line-b", "qc-03"))):
                iid = f"demo-mon-{day}-{idx}"
                insp = await _inspection(
                    session, iid, await _product(session, f"DEMO-MON-{day}-{idx}", line, station),
                    quality=QualityResult.PASS if (day + idx) % 3 else QualityResult.REVIEW,
                    created=base + timedelta(hours=idx * 6),
                    latency=110.0 + (day + idx) * 7.0, anomaly=0.5 + (idx * 0.2) % 0.4,
                )
                if insp.quality_result == QualityResult.FAIL:
                    await _defect(session, insp, "patches", 0.6)
                created += 1

        await session.commit()
    print(f"demo seed ok: {created} inspections (idempotent; DEMO-* fixture data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
