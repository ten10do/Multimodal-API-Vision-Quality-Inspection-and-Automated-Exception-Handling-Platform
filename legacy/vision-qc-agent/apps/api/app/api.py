import asyncio
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from redis.asyncio import Redis
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import add_audit
from app.config import Settings, get_settings
from app.db import get_session
from app.enums import (
    ActionStatus,
    ActionType,
    Disposition,
    InspectionStatus,
    RiskLevel,
)
from app.models import AuditLog, HumanFeedback, Inspection, WorkflowAction
from app.schemas import (
    ApprovalRequest,
    AuditLogRead,
    DashboardStats,
    FeedbackCreate,
    FeedbackRead,
    HealthResponse,
    InspectionList,
    InspectionListItem,
    InspectionRead,
    ReadyResponse,
    WorkflowActionRead,
)
from app.security import validate_and_store_image
from app.tools import SimulatedToolExecutor
from app.workflow import process_inspection, validate_transition

router = APIRouter()


async def _inspection_read(session: AsyncSession, inspection: Inspection) -> InspectionRead:
    actions = list(
        await session.scalars(
            select(WorkflowAction)
            .where(WorkflowAction.inspection_id == inspection.id)
            .order_by(WorkflowAction.created_at)
        )
    )
    audits = list(
        await session.scalars(
            select(AuditLog)
            .where(AuditLog.inspection_id == inspection.id)
            .order_by(AuditLog.created_at)
        )
    )
    feedback = list(
        await session.scalars(
            select(HumanFeedback)
            .where(HumanFeedback.inspection_id == inspection.id)
            .order_by(HumanFeedback.created_at)
        )
    )
    base = {
        column: getattr(inspection, column)
        for column in (
            "id",
            "product_code",
            "batch_code",
            "original_filename",
            "content_type",
            "status",
            "risk_level",
            "disposition",
            "vision_result",
            "analysis_result",
            "error_code",
            "error_message",
            "created_at",
            "updated_at",
        )
    }
    return InspectionRead(
        **base,
        actions=[WorkflowActionRead.model_validate(item) for item in actions],
        audit_logs=[AuditLogRead.model_validate(item) for item in audits],
        feedback=[FeedbackRead.model_validate(item) for item in feedback],
    )


@router.get("/health", response_model=HealthResponse)
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(status="ok", provider_mode=settings.ai_mode)


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadyResponse:
    try:
        await session.execute(text("SELECT 1"))
        redis_status = "not_required"
        if not settings.celery_task_always_eager:
            redis_client = Redis.from_url(settings.redis_url)
            try:
                await asyncio.wait_for(redis_client.ping(), timeout=2)
                redis_status = "ok"
            finally:
                await redis_client.aclose()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "not_ready",
                "message": "Database or queue dependency is unavailable",
            },
        ) from exc
    return ReadyResponse(status="ready", database="ok", redis=redis_status)


@router.post("/inspections", response_model=InspectionRead, status_code=status.HTTP_201_CREATED)
async def create_inspection(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    image: Annotated[UploadFile, File()],
    product_code: Annotated[str, Form(min_length=1, max_length=100)],
    batch_code: Annotated[str, Form(min_length=1, max_length=100)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8, max_length=128)],
) -> InspectionRead:
    normalized_product_code = product_code.strip()
    normalized_batch_code = batch_code.strip()
    if not normalized_product_code or not normalized_batch_code:
        raise HTTPException(
            status_code=422,
            detail={"code": "blank_identifier", "message": "产品编码和批次不能为空"},
        )
    existing = await session.scalar(
        select(Inspection).where(Inspection.idempotency_key == idempotency_key)
    )
    if existing:
        if (
            existing.product_code != normalized_product_code
            or existing.batch_code != normalized_batch_code
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "idempotency_conflict",
                    "message": "同一幂等键不能用于不同的质检请求",
                },
            )
        return await _inspection_read(session, existing)

    _, stored_name, sha256 = await validate_and_store_image(image, settings)
    inspection = Inspection(
        idempotency_key=idempotency_key,
        product_code=normalized_product_code,
        batch_code=normalized_batch_code,
        original_filename=Path(image.filename or "image").name,
        stored_filename=stored_name,
        content_type=image.content_type or "application/octet-stream",
        file_sha256=sha256,
        status=InspectionStatus.QUEUED,
    )
    session.add(inspection)
    await session.flush()
    add_audit(
        session,
        inspection.id,
        event_type="inspection_created",
        actor_type="user",
        actor_id="operator",
        new_state={"status": InspectionStatus.QUEUED.value},
        detail={"product_code": inspection.product_code, "batch_code": inspection.batch_code},
    )
    await session.commit()

    if settings.celery_task_always_eager:
        await process_inspection(inspection.id)
    else:
        from app.worker import process_inspection_task

        process_inspection_task.delay(str(inspection.id))
    await session.refresh(inspection)
    return await _inspection_read(session, inspection)


@router.get("/inspections", response_model=InspectionList)
async def list_inspections(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: int = 50,
) -> InspectionList:
    limit = max(1, min(limit, 100))
    items = list(
        await session.scalars(
            select(Inspection).order_by(Inspection.created_at.desc()).limit(limit)
        )
    )
    total = await session.scalar(select(func.count()).select_from(Inspection)) or 0
    return InspectionList(
        items=[InspectionListItem.model_validate(item) for item in items],
        total=total,
    )


@router.get("/inspections/{inspection_id}", response_model=InspectionRead)
async def get_inspection(
    inspection_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> InspectionRead:
    inspection = await session.get(Inspection, inspection_id)
    if not inspection:
        raise HTTPException(status_code=404, detail={"code": "not_found"})
    return await _inspection_read(session, inspection)


@router.post("/inspections/{inspection_id}/approval", response_model=InspectionRead)
async def decide_approval(
    inspection_id: UUID,
    request: ApprovalRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> InspectionRead:
    inspection = await session.get(Inspection, inspection_id, with_for_update=True)
    if not inspection:
        raise HTTPException(status_code=404, detail={"code": "not_found"})
    if inspection.status != InspectionStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail={"code": "invalid_state", "message": "当前任务不在等待审批状态"},
        )
    pending = await session.scalar(
        select(WorkflowAction).where(
            WorkflowAction.inspection_id == inspection.id,
            WorkflowAction.action_type == ActionType.REQUEST_LINE_STOP,
        )
    )
    if request.decision == "approve":
        if pending:
            pending.status = ActionStatus.SUCCEEDED
        await SimulatedToolExecutor().execute(
            session,
            inspection.id,
            ActionType.EXECUTE_LINE_STOP,
            {"reviewer": request.reviewer, "comment": request.comment},
        )
        validate_transition(inspection.status, InspectionStatus.COMPLETED)
        inspection.status = InspectionStatus.COMPLETED
        event = "line_stop_approved"
    else:
        if pending:
            pending.status = ActionStatus.REJECTED
        validate_transition(inspection.status, InspectionStatus.MANUAL_REVIEW)
        inspection.status = InspectionStatus.MANUAL_REVIEW
        inspection.disposition = Disposition.MANUAL_REVIEW
        event = "line_stop_rejected"
    add_audit(
        session,
        inspection.id,
        event_type=event,
        actor_type="human",
        actor_id=request.reviewer,
        new_state={"status": inspection.status.value},
        detail={"comment": request.comment},
    )
    await session.commit()
    await session.refresh(inspection)
    return await _inspection_read(session, inspection)


@router.post("/inspections/{inspection_id}/feedback", response_model=InspectionRead)
async def create_feedback(
    inspection_id: UUID,
    request: FeedbackCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> InspectionRead:
    inspection = await session.get(Inspection, inspection_id, with_for_update=True)
    if not inspection:
        raise HTTPException(status_code=404, detail={"code": "not_found"})
    previous = {
        "risk_level": inspection.risk_level.value if inspection.risk_level else None,
        "disposition": inspection.disposition.value if inspection.disposition else None,
    }
    feedback = HumanFeedback(
        inspection_id=inspection.id,
        reviewer=request.reviewer,
        comment=request.comment,
        corrected_risk=request.corrected_risk,
        corrected_disposition=request.corrected_disposition,
    )
    session.add(feedback)
    if request.corrected_risk:
        inspection.risk_level = request.corrected_risk
    if request.corrected_disposition:
        inspection.disposition = request.corrected_disposition
    add_audit(
        session,
        inspection.id,
        event_type="human_feedback_recorded",
        actor_type="human",
        actor_id=request.reviewer,
        previous_state=previous,
        new_state={
            "risk_level": inspection.risk_level.value if inspection.risk_level else None,
            "disposition": inspection.disposition.value if inspection.disposition else None,
        },
        detail={"comment": request.comment},
    )
    await session.commit()
    await session.refresh(inspection)
    return await _inspection_read(session, inspection)


@router.get("/dashboard/stats", response_model=DashboardStats)
async def dashboard_stats(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DashboardStats:
    inspections = list(await session.scalars(select(Inspection)))
    total = len(inspections)
    defective = sum(item.risk_level not in {None, RiskLevel.LOW} for item in inspections)
    return DashboardStats(
        total=total,
        completed=sum(item.status == InspectionStatus.COMPLETED for item in inspections),
        awaiting_approval=sum(
            item.status == InspectionStatus.AWAITING_APPROVAL for item in inspections
        ),
        manual_review=sum(item.status == InspectionStatus.MANUAL_REVIEW for item in inspections),
        defect_rate=round(defective / total, 4) if total else 0,
        by_risk={
            risk.value: sum(item.risk_level == risk for item in inspections) for risk in RiskLevel
        },
    )
