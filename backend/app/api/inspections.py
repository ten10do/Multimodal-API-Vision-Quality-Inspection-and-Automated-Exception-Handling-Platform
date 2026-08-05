from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..events import InspectionEvent
from ..metrics import metrics
from ..models import Inspection
from ..schemas import InspectionDetail
from ..services.contract_validation import InvalidImageError
from ..services.inspection_service import CreateInspectionInput, InspectionService, InspectionServiceError
from ..storage import get_storage
from ..ws import schedule_broadcast
from .serializers import to_inspection_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inspections"])

_EAGER = (selectinload(Inspection.defects), selectinload(Inspection.product))


def get_inspection_service() -> InspectionService:
    return InspectionService()


@router.post("/inspections", response_model=InspectionDetail, status_code=201)
async def create_inspection(
    file: UploadFile = File(...),
    product_id: str = Form(...),
    batch_id: str | None = Form(default=None),
    production_line: str = Form(default="line-a"),
    station: str = Form(default="qc-01"),
    idempotency_key: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
    service: InspectionService = Depends(get_inspection_service),
) -> InspectionDetail:
    started = time.perf_counter()
    data = await file.read()
    try:
        inspection, created = await service.create(
            session,
            CreateInspectionInput(
                image_bytes=data,
                filename=file.filename or "image.jpg",
                product_id=product_id,
                batch_id=batch_id,
                production_line=production_line,
                station=station,
                idempotency_key=idempotency_key,
            ),
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=_err("invalid_image", str(exc))) from exc
    except InspectionServiceError as exc:
        # The inspection row was already persisted (FAILED) before raising.
        # Persistence-first: metrics + WS event are best-effort after commit.
        await metrics.record_failed()
        if exc.inspection_id:
            failed = await _load_detail(session, exc.inspection_id)
            if failed is not None:
                event = _build_event(failed, event_type="inspection.failed", error=exc.message)
                schedule_broadcast(event.to_broadcast())
        raise HTTPException(status_code=exc.http_status, detail=_err(exc.code, exc.message)) from exc

    eager = await _load_detail(session, inspection.inspection_id)
    assert eager is not None
    detail = to_inspection_detail(eager)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not created:
        # Idempotent replay: do not count as a new processing event.
        return Response(status_code=200, content=detail.model_dump_json(), media_type="application/json")

    if eager.quality_result is not None:
        await metrics.record_completed(eager.quality_result.value, elapsed_ms, eager.inference_latency_ms)
    event = _build_event(eager, event_type="inspection.completed")
    schedule_broadcast(event.to_broadcast())

    # 5B: REVIEW inspections automatically enter the human review queue
    # (idempotent; system FAILED never creates a task). `eager` carries the
    # defects snapshot loaded for the response.
    from ..services.review_service import ReviewService

    await ReviewService().create_task_for_inspection(session, eager)
    return detail


@router.get("/inspections", response_model=list[InspectionDetail])
async def list_inspections(
    product_id: str | None = None,
    inspection_id: str | None = None,
    batch_id: str | None = None,
    quality_result: str | None = None,
    status: str | None = None,
    defect_type: str | None = None,
    production_line: str | None = None,
    station: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[InspectionDetail]:
    """Search/traceability endpoint (4E). Filters are applied server-side."""
    from datetime import datetime

    from ..enums import InspectionStatus, QualityResult as QR
    from ..models import Defect, Product

    stmt = (
        select(Inspection)
        .join(Inspection.product)
        .options(*_EAGER)
        .order_by(Inspection.created_at.desc())
    )
    if product_id:
        stmt = stmt.where(Product.product_id == product_id)
    if inspection_id:
        stmt = stmt.where(Inspection.inspection_id == inspection_id)
    if batch_id:
        stmt = stmt.where(Inspection.batch_id == batch_id)
    if quality_result:
        stmt = stmt.where(Inspection.quality_result == QR(quality_result))
    if status:
        stmt = stmt.where(Inspection.status == InspectionStatus(status))
    if defect_type:
        stmt = stmt.where(Inspection.defects.any(Defect.class_name == defect_type))
    if production_line:
        stmt = stmt.where(Product.production_line == production_line)
    if station:
        stmt = stmt.where(Product.station == station)
    if date_from:
        stmt = stmt.where(Inspection.created_at >= datetime.fromisoformat(date_from))
    if date_to:
        stmt = stmt.where(Inspection.created_at <= datetime.fromisoformat(date_to))
    stmt = stmt.limit(limit).offset(offset)
    result = await session.execute(stmt)
    return [to_inspection_detail(i) for i in result.scalars()]


@router.get("/inspections/{inspection_id}", response_model=InspectionDetail)
async def get_inspection(inspection_id: str, session: AsyncSession = Depends(get_session)) -> InspectionDetail:
    inspection = await _load_detail(session, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "inspection not found"))
    return to_inspection_detail(inspection)


@router.get("/inspections/{inspection_id}/image")
async def get_inspection_image(inspection_id: str, session: AsyncSession = Depends(get_session)) -> FileResponse:
    """Serve the raw captured image for an inspection (4D). The dashboard
    draws Bounding Boxes from the Vision Contract; no server-side annotation."""
    inspection = await _load_detail(session, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "inspection not found"))
    path = get_storage().path_for(inspection_id)
    if path is None:
        raise HTTPException(status_code=404, detail=_err("image_missing", "image not stored for this inspection"))
    return FileResponse(path, media_type="image/jpeg")


@router.get("/inspections/{inspection_id}/anomaly-map")
async def get_anomaly_map(inspection_id: str, session: AsyncSession = Depends(get_session)) -> Response:
    """Phase 6 (6G): serve the PatchCore heatmap PNG for a REVIEW inspection."""
    inspection = await _load_detail(session, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "inspection not found"))
    data = get_storage().load_anomaly_map(inspection_id)
    if data is None:
        raise HTTPException(status_code=404, detail=_err("anomaly_map_missing", "anomaly map not stored"))
    return Response(content=data, media_type="image/png")


async def _load_detail(session: AsyncSession, inspection_id: str) -> Inspection | None:
    result = await session.execute(
        select(Inspection).where(Inspection.inspection_id == inspection_id).options(*_EAGER)
    )
    return result.scalar_one_or_none()


def _build_event(inspection: Inspection, *, event_type: str, error: str | None = None) -> InspectionEvent:
    return InspectionEvent(
        event_type=event_type,
        product_id=inspection.product.product_id,
        inspection_id=inspection.inspection_id,
        batch_id=inspection.batch_id,
        production_line=inspection.product.production_line,
        station=inspection.product.station,
        process_status=inspection.status.value.upper(),
        quality_result=inspection.quality_result,
        severity=inspection.severity,
        defect_count=len(inspection.defects),
        inference_latency_ms=inspection.inference_latency_ms,
        model_version=inspection.model_version,
        error_message=error,
    )


def _err(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": uuid.uuid4().hex[:12]}}
