import time
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import add_audit
from app.config import get_settings
from app.db import SessionLocal
from app.enums import ActionType, Disposition, InspectionStatus, RiskLevel
from app.models import Inspection, ModelCall
from app.providers import get_reasoning_provider, get_vision_provider
from app.providers.base import ProviderError
from app.schemas import AnalysisRequest, AnalysisResult, InspectionContext, VisionInspectionResult
from app.tools import SimulatedToolExecutor

QUALITY_RULES = [
    "critical 缺陷必须隔离产品并申请停线",
    "high 缺陷必须剔除并创建工单",
    "medium 缺陷转人工复检",
    "只有 low 风险可以自动放行",
]
RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}
DISPOSITION_ORDER = {
    Disposition.RELEASE: 0,
    Disposition.MANUAL_REVIEW: 1,
    Disposition.REJECT: 2,
    Disposition.STOP_LINE: 3,
}
ALLOWED_TRANSITIONS: dict[InspectionStatus, set[InspectionStatus]] = {
    InspectionStatus.QUEUED: {InspectionStatus.VISION_ANALYZING},
    InspectionStatus.FAILED: {InspectionStatus.VISION_ANALYZING},
    InspectionStatus.VISION_ANALYZING: {
        InspectionStatus.REASONING,
        InspectionStatus.MANUAL_REVIEW,
        InspectionStatus.FAILED,
    },
    InspectionStatus.REASONING: {
        InspectionStatus.EXECUTING,
        InspectionStatus.MANUAL_REVIEW,
        InspectionStatus.FAILED,
    },
    InspectionStatus.EXECUTING: {
        InspectionStatus.COMPLETED,
        InspectionStatus.MANUAL_REVIEW,
        InspectionStatus.AWAITING_APPROVAL,
        InspectionStatus.FAILED,
    },
    InspectionStatus.AWAITING_APPROVAL: {
        InspectionStatus.COMPLETED,
        InspectionStatus.MANUAL_REVIEW,
    },
    InspectionStatus.MANUAL_REVIEW: {InspectionStatus.COMPLETED},
    InspectionStatus.COMPLETED: set(),
}


class InvalidStateTransition(ValueError):
    pass


def validate_transition(previous: InspectionStatus, target: InspectionStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[previous]:
        raise InvalidStateTransition(
            f"Illegal inspection transition: {previous.value} -> {target.value}"
        )


def enforce_quality_rules(
    vision: VisionInspectionResult, analysis: AnalysisResult
) -> AnalysisResult:
    minimum_risk = RiskLevel.LOW
    rule_reason = "处置与风险等级必须一致。"
    if vision.overall_confidence < 0.65:
        minimum_risk = RiskLevel.MEDIUM
        rule_reason = "视觉置信度低于 0.65，规则强制人工复检。"
    elif vision.defects:
        minimum_risk = max(
            (defect.severity for defect in vision.defects),
            key=RISK_ORDER.__getitem__,
        )
        rule_reason = "处置不得低于视觉缺陷的最高严重度。"
    final_risk = max((analysis.risk_level, minimum_risk), key=RISK_ORDER.__getitem__)
    risk_disposition = {
        RiskLevel.LOW: Disposition.RELEASE,
        RiskLevel.MEDIUM: Disposition.MANUAL_REVIEW,
        RiskLevel.HIGH: Disposition.REJECT,
        RiskLevel.CRITICAL: Disposition.STOP_LINE,
    }[final_risk]
    final_disposition = max(
        (analysis.disposition, risk_disposition), key=DISPOSITION_ORDER.__getitem__
    )
    requires_approval = (
        analysis.requires_human_approval or final_disposition == Disposition.STOP_LINE
    )
    if (
        final_risk == analysis.risk_level
        and final_disposition == analysis.disposition
        and requires_approval == analysis.requires_human_approval
    ):
        return analysis
    return analysis.model_copy(
        update={
            "risk_level": final_risk,
            "disposition": final_disposition,
            "requires_human_approval": requires_approval,
            "rationale": f"{analysis.rationale} {rule_reason}",
        }
    )


async def _transition(
    session: AsyncSession, inspection: Inspection, status: InspectionStatus
) -> None:
    previous = inspection.status
    validate_transition(previous, status)
    inspection.status = status
    add_audit(
        session,
        inspection.id,
        event_type="status_changed",
        previous_state={"status": previous.value},
        new_state={"status": status.value},
    )
    await session.commit()


async def _recent_cases(session: AsyncSession, inspection: Inspection) -> list[dict[str, object]]:
    result = await session.scalars(
        select(Inspection)
        .where(
            Inspection.product_code == inspection.product_code,
            Inspection.id != inspection.id,
            Inspection.analysis_result.is_not(None),
        )
        .order_by(Inspection.created_at.desc())
        .limit(5)
    )
    return [
        {
            "risk_level": item.risk_level.value if item.risk_level else None,
            "disposition": item.disposition.value if item.disposition else None,
            "analysis": item.analysis_result or {},
        }
        for item in result
    ]


async def process_inspection(inspection_id: UUID) -> None:
    settings = get_settings()
    async with SessionLocal() as session:
        inspection = await session.get(Inspection, inspection_id)
        if inspection is None or inspection.status not in {
            InspectionStatus.QUEUED,
            InspectionStatus.FAILED,
        }:
            return
        context = InspectionContext(
            product_code=inspection.product_code,
            batch_code=inspection.batch_code,
            image_mime_type=cast(
                Literal["image/jpeg", "image/png", "image/webp"],
                inspection.content_type,
            ),
            quality_rules=QUALITY_RULES,
        )
        stage = "bailian"
        started = time.perf_counter()
        try:
            await _transition(session, inspection, InspectionStatus.VISION_ANALYZING)
            image = (Path(settings.upload_dir) / inspection.stored_filename).read_bytes()
            vision_result = await get_vision_provider(settings).inspect(image, context)
            session.add(
                ModelCall(
                    inspection_id=inspection.id,
                    provider="mock" if settings.ai_mode == "mock" else "bailian",
                    model=settings.mock_vision_model
                    if settings.ai_mode == "mock"
                    else settings.bailian_model,
                    request_summary={
                        "image_sha256": inspection.file_sha256,
                        "image_bytes": len(image),
                        "context": context.model_dump(mode="json"),
                    },
                    response_payload=vision_result.model_dump(mode="json"),
                    success=True,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            inspection.vision_result = vision_result.model_dump(mode="json")
            await session.commit()

            await _transition(session, inspection, InspectionStatus.REASONING)
            stage = "deepseek"
            analysis_request = AnalysisRequest(
                vision_result=vision_result,
                context=context,
                historical_cases=await _recent_cases(session, inspection),
            )
            started = time.perf_counter()
            provider_analysis = await get_reasoning_provider(settings).analyze(analysis_request)
            session.add(
                ModelCall(
                    inspection_id=inspection.id,
                    provider="mock" if settings.ai_mode == "mock" else "deepseek",
                    model=settings.mock_reasoning_model
                    if settings.ai_mode == "mock"
                    else settings.deepseek_model,
                    request_summary={
                        "defect_count": len(vision_result.defects),
                        "history_count": len(analysis_request.historical_cases),
                    },
                    response_payload=provider_analysis.model_dump(mode="json"),
                    success=True,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            analysis = enforce_quality_rules(vision_result, provider_analysis)
            if analysis != provider_analysis:
                add_audit(
                    session,
                    inspection.id,
                    event_type="quality_rule_override",
                    previous_state=provider_analysis.model_dump(mode="json"),
                    new_state=analysis.model_dump(mode="json"),
                )
            inspection.analysis_result = analysis.model_dump(mode="json")
            inspection.risk_level = analysis.risk_level
            inspection.disposition = analysis.disposition
            await session.commit()

            await _transition(session, inspection, InspectionStatus.EXECUTING)
            await _execute_actions(session, inspection, analysis.disposition)
        except (ProviderError, OSError, ValueError) as exc:
            safe_message = (
                exc.safe_message
                if isinstance(exc, ProviderError)
                else "自动分析失败，已安全降级到人工复检"
            )
            validate_transition(inspection.status, InspectionStatus.MANUAL_REVIEW)
            inspection.status = InspectionStatus.MANUAL_REVIEW
            inspection.disposition = Disposition.MANUAL_REVIEW
            inspection.error_code = exc.code if isinstance(exc, ProviderError) else "workflow_error"
            inspection.error_message = safe_message
            session.add(
                ModelCall(
                    inspection_id=inspection.id,
                    provider="mock" if settings.ai_mode == "mock" else stage,
                    model=(
                        settings.mock_vision_model
                        if settings.ai_mode == "mock" and stage == "bailian"
                        else settings.mock_reasoning_model
                        if settings.ai_mode == "mock"
                        else settings.bailian_model
                        if stage == "bailian"
                        else settings.deepseek_model
                    ),
                    request_summary={"stage": stage},
                    response_payload=None,
                    success=False,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error_code=inspection.error_code,
                )
            )
            add_audit(
                session,
                inspection.id,
                event_type="safe_fallback",
                new_state={"status": InspectionStatus.MANUAL_REVIEW.value},
                detail={"code": inspection.error_code, "message": safe_message},
            )
            await SimulatedToolExecutor().execute(
                session,
                inspection.id,
                ActionType.MANUAL_REVIEW,
                {"reason": safe_message},
            )
            await session.commit()


async def _execute_actions(
    session: AsyncSession, inspection: Inspection, disposition: Disposition
) -> None:
    tools = SimulatedToolExecutor()
    common: dict[str, object] = {
        "product_code": inspection.product_code,
        "batch_code": inspection.batch_code,
    }
    if disposition == Disposition.RELEASE:
        await tools.execute(session, inspection.id, ActionType.RELEASE_PRODUCT, common)
        final_status = InspectionStatus.COMPLETED
    elif disposition == Disposition.MANUAL_REVIEW:
        await tools.execute(session, inspection.id, ActionType.MANUAL_REVIEW, common)
        await tools.execute(session, inspection.id, ActionType.CREATE_TICKET, common)
        final_status = InspectionStatus.MANUAL_REVIEW
    else:
        await tools.execute(session, inspection.id, ActionType.REJECT_PRODUCT, common)
        await tools.execute(session, inspection.id, ActionType.CREATE_TICKET, common)
        await tools.execute(session, inspection.id, ActionType.SEND_NOTIFICATION, common)
        if disposition == Disposition.STOP_LINE:
            await tools.execute(
                session,
                inspection.id,
                ActionType.REQUEST_LINE_STOP,
                common,
                pending_approval=True,
            )
            final_status = InspectionStatus.AWAITING_APPROVAL
        else:
            final_status = InspectionStatus.COMPLETED
    validate_transition(inspection.status, final_status)
    inspection.status = final_status
    add_audit(
        session,
        inspection.id,
        event_type="workflow_actions_executed",
        new_state={"status": final_status.value, "disposition": disposition.value},
    )
    await session.commit()
