from __future__ import annotations

from ..models import Inspection
from ..schemas import DefectOut, InspectionDetail, InspectionOut, ProductOut


def to_inspection_out(inspection: Inspection) -> InspectionOut:
    return InspectionOut(
        inspection_id=inspection.inspection_id,
        product_id=inspection.product.product_id,
        status=inspection.status,
        quality_result=inspection.quality_result,
        severity=inspection.severity,
        model_name=inspection.model_name,
        model_version=inspection.model_version,
        rule_version=inspection.rule_version,
        inference_latency_ms=inspection.inference_latency_ms,
        error_message=inspection.error_message,
        created_at=inspection.created_at,
        defects=[DefectOut.model_validate(d) for d in inspection.defects],
    )


def to_inspection_detail(inspection: Inspection) -> InspectionDetail:
    base = to_inspection_out(inspection)
    return InspectionDetail(
        **base.model_dump(),
        product=ProductOut.model_validate(inspection.product),
    )
