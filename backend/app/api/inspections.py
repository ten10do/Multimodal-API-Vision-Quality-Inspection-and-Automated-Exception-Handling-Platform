from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..models import Inspection
from ..schemas import InspectionDetail
from ..services.contract_validation import InvalidImageError
from ..services.inspection_service import CreateInspectionInput, InspectionService, InspectionServiceError
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
        raise HTTPException(status_code=exc.http_status, detail=_err(exc.code, exc.message)) from exc

    eager = await _load_detail(session, inspection.inspection_id)
    assert eager is not None
    detail = to_inspection_detail(eager)
    if not created:
        return Response(status_code=200, content=detail.model_dump_json(), media_type="application/json")
    return detail


@router.get("/inspections/{inspection_id}", response_model=InspectionDetail)
async def get_inspection(inspection_id: str, session: AsyncSession = Depends(get_session)) -> InspectionDetail:
    inspection = await _load_detail(session, inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "inspection not found"))
    return to_inspection_detail(inspection)


async def _load_detail(session: AsyncSession, inspection_id: str) -> Inspection | None:
    result = await session.execute(
        select(Inspection).where(Inspection.inspection_id == inspection_id).options(*_EAGER)
    )
    return result.scalar_one_or_none()


def _err(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": uuid.uuid4().hex[:12]}}
