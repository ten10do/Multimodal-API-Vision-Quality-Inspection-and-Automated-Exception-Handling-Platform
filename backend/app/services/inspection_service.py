from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..enums import InspectionStatus, QualityResult
from ..inference.client import InferenceClient
from ..models import Defect, Inspection, Product, QualityRule
from ..quality.engine import DefectInput, QualityRuleEngine
from .contract_validation import validate_image_bytes

logger = logging.getLogger(__name__)


class InspectionServiceError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 500, inspection_id: str | None = None) -> None:
        self.code = code
        self.message = message
        self.http_status = http_status
        self.inspection_id = inspection_id
        super().__init__(message)


@dataclass
class CreateInspectionInput:
    image_bytes: bytes
    filename: str
    product_id: str
    batch_id: str | None = None
    production_line: str = "line-a"
    station: str = "qc-01"
    idempotency_key: str | None = None


class InspectionService:
    def __init__(self, inference_client: InferenceClient | None = None) -> None:
        self.inference = inference_client or InferenceClient()

    async def create(self, session: AsyncSession, data: CreateInspectionInput) -> tuple[Inspection, bool]:
        validate_image_bytes(data.image_bytes)

        if data.idempotency_key:
            existing = await self._find_by_idempotency(session, data.idempotency_key)
            if existing is not None:
                logger.info("idempotent hit key=%s inspection=%s", data.idempotency_key, existing.inspection_id)
                return existing, False

        inspection_id = f"insp-{uuid.uuid4().hex[:12]}"
        product = await self._get_or_create_product(session, data)

        inspection = Inspection(
            inspection_id=inspection_id,
            product_id=product.id,
            idempotency_key=data.idempotency_key,
            batch_id=data.batch_id,
            status=InspectionStatus.PENDING,
        )
        session.add(inspection)
        try:
            await session.flush()
        except IntegrityError as exc:
            await session.rollback()
            raise InspectionServiceError("duplicate_request", "duplicate inspection request", 409) from exc

        request_id = f"req-{uuid.uuid4().hex[:12]}"
        try:
            contract = await self.inference.infer(data.image_bytes, data.filename, request_id=request_id)
        except Exception as exc:
            await self._mark_failed(session, inspection, str(exc))
            logger.warning("inference failed inspection=%s error=%s", inspection_id, exc)
            raise InspectionServiceError(
                "inference_failed", str(exc), http_status=_http_status_for(exc), inspection_id=inspection_id
            ) from exc

        rules = await self._load_rules(session)
        decision = QualityRuleEngine(rules).evaluate(
            [
                DefectInput(
                    class_id=d.class_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    defect_area_ratio=d.defect_area_ratio,
                )
                for d in contract.detections
            ]
        )

        inspection.status = InspectionStatus.COMPLETED
        inspection.quality_result = decision.quality_result
        inspection.severity = decision.severity
        inspection.model_name = contract.model_name
        inspection.model_version = contract.model_version
        inspection.rule_version = decision.rule_version
        inspection.inference_latency_ms = contract.inference_latency_ms
        inspection.inference_request_id = request_id
        inspection.image_path = data.filename

        for det in contract.detections:
            session.add(
                Defect(
                    inspection_id=inspection.id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=list(det.bbox_xyxy),
                    bbox_normalized=list(det.bbox_normalized),
                    defect_area_px=det.defect_area_px,
                    defect_area_ratio=det.defect_area_ratio,
                )
            )

        try:
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.error("db commit failed inspection=%s error=%s", inspection_id, exc)
            raise InspectionServiceError("db_write_failed", "failed to persist inspection", 500) from exc

        await session.refresh(inspection)
        return inspection, True

    async def _find_by_idempotency(self, session: AsyncSession, key: str) -> Inspection | None:
        result = await session.execute(select(Inspection).where(Inspection.idempotency_key == key))
        return result.scalar_one_or_none()

    async def _get_or_create_product(self, session: AsyncSession, data: CreateInspectionInput) -> Product:
        result = await session.execute(select(Product).where(Product.product_id == data.product_id))
        product = result.scalar_one_or_none()
        if product is None:
            product = Product(
                product_id=data.product_id,
                production_line=data.production_line,
                station=data.station,
            )
            session.add(product)
            try:
                await session.flush()
            except IntegrityError:
                await session.rollback()
                result = await session.execute(select(Product).where(Product.product_id == data.product_id))
                product = result.scalar_one()
        return product

    async def _load_rules(self, session: AsyncSession) -> list[QualityRule]:
        result = await session.execute(select(QualityRule).where(QualityRule.enabled.is_(True)))
        return list(result.scalars())

    async def _mark_failed(self, session: AsyncSession, inspection: Inspection, message: str) -> None:
        inspection.status = InspectionStatus.FAILED
        inspection.error_message = message[:500]
        try:
            await session.commit()
        except SQLAlchemyError:
            await session.rollback()
            logger.exception("failed to persist inspection failure state")


def _http_status_for(exc: Exception) -> int:
    from ..inference.client import InferenceConnectionError, InferenceContractError, InferenceHTTPError, InferenceTimeoutError

    if isinstance(exc, (InferenceTimeoutError, InferenceConnectionError)):
        return 504
    if isinstance(exc, (InferenceHTTPError, InferenceContractError)):
        return 502
    return 500
