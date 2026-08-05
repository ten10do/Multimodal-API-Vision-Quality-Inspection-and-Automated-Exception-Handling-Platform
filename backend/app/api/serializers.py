from __future__ import annotations

from ..models import Inspection, ReviewDecision, ReviewTask
from ..schemas import (
    DefectOut,
    InspectionDetail,
    InspectionOut,
    ProductOut,
    ReviewDecisionOut,
    ReviewTaskOut,
)


def to_inspection_out(inspection: Inspection) -> InspectionOut:
    return InspectionOut(
        inspection_id=inspection.inspection_id,
        product_id=inspection.product.product_id,
        batch_id=inspection.batch_id,
        image_url=f"/api/v1/inspections/{inspection.inspection_id}/image",
        status=inspection.status,
        quality_result=inspection.quality_result,
        final_quality_result=inspection.final_quality_result,
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


def to_review_task_out(task: ReviewTask) -> ReviewTaskOut:
    inspection_detail = None
    if task.inspection is not None:
        inspection_detail = to_inspection_detail(task.inspection)
    decision_out = None
    if task.decision is not None:
        decision_out = ReviewDecisionOut.model_validate(task.decision)
    return ReviewTaskOut(
        review_task_id=task.review_task_id,
        inspection_id=task.inspection_id,
        inspection=inspection_detail,
        status=task.status.value,
        priority=task.priority,
        assigned_to=task.assigned_to,
        claimed_at=task.claimed_at,
        resolved_at=task.resolved_at,
        version=task.version,
        ai_quality_result=task.ai_quality_result,
        ai_defects_snapshot=task.ai_defects_snapshot,
        ai_model_version=task.ai_model_version,
        ai_rule_version=task.ai_rule_version,
        ai_severity=task.ai_severity,
        product_id=task.product_id,
        production_line=task.production_line,
        station=task.station,
        batch_id=task.batch_id,
        image_url=task.image_url,
        decision=decision_out,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )
