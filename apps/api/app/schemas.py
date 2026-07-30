from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from app.enums import ActionStatus, ActionType, Disposition, InspectionStatus, RiskLevel

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundingBox(StrictSchema):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class Defect(StrictSchema):
    defect_type: NonEmptyText
    confidence: float = Field(ge=0, le=1)
    severity: RiskLevel
    description: NonEmptyText
    bounding_box: BoundingBox | None = None


class VisionInspectionResult(StrictSchema):
    is_defective: bool
    overall_confidence: float = Field(ge=0, le=1)
    defects: list[Defect] = Field(max_length=50)
    summary: NonEmptyText

    @model_validator(mode="after")
    def defects_match_flag(self) -> "VisionInspectionResult":
        if self.is_defective != bool(self.defects):
            raise ValueError("is_defective must match whether defects are present")
        return self


class InspectionContext(StrictSchema):
    product_code: NonEmptyText
    batch_code: NonEmptyText
    image_mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    quality_rules: list[NonEmptyText]


class AnalysisRequest(StrictSchema):
    vision_result: VisionInspectionResult
    context: InspectionContext
    historical_cases: list[dict[str, object]] = Field(default_factory=list, max_length=10)


class AnalysisResult(StrictSchema):
    risk_level: RiskLevel
    probable_causes: list[NonEmptyText] = Field(min_length=1, max_length=10)
    recommended_actions: list[NonEmptyText] = Field(min_length=1, max_length=10)
    disposition: Disposition
    requires_human_approval: bool
    rationale: NonEmptyText

    @field_validator("requires_human_approval")
    @classmethod
    def stop_requires_approval(cls, value: bool, info: object) -> bool:
        data = getattr(info, "data", {})
        if data.get("disposition") == Disposition.STOP_LINE and not value:
            raise ValueError("stop_line disposition requires human approval")
        return value


class WorkflowActionRead(StrictSchema):
    id: UUID
    action_type: ActionType
    status: ActionStatus
    result_payload: dict[str, object] | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class AuditLogRead(StrictSchema):
    id: UUID
    actor_type: str
    actor_id: str
    event_type: str
    previous_state: dict[str, object] | None
    new_state: dict[str, object] | None
    detail: dict[str, object]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class FeedbackRead(StrictSchema):
    id: UUID
    reviewer: str
    comment: str
    corrected_risk: RiskLevel | None
    corrected_disposition: Disposition | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class InspectionRead(StrictSchema):
    id: UUID
    product_code: str
    batch_code: str
    original_filename: str
    content_type: str
    status: InspectionStatus
    risk_level: RiskLevel | None
    disposition: Disposition | None
    vision_result: dict[str, object] | None
    analysis_result: dict[str, object] | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    actions: list[WorkflowActionRead] = Field(default_factory=list)
    audit_logs: list[AuditLogRead] = Field(default_factory=list)
    feedback: list[FeedbackRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class InspectionListItem(StrictSchema):
    id: UUID
    product_code: str
    batch_code: str
    status: InspectionStatus
    risk_level: RiskLevel | None
    disposition: Disposition | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class InspectionList(StrictSchema):
    items: list[InspectionListItem]
    total: int


class ApprovalRequest(StrictSchema):
    decision: Literal["approve", "reject"]
    reviewer: NonEmptyText
    comment: NonEmptyText


class FeedbackCreate(StrictSchema):
    reviewer: NonEmptyText
    comment: NonEmptyText
    corrected_risk: RiskLevel | None = None
    corrected_disposition: Disposition | None = None


class DashboardStats(StrictSchema):
    total: int
    completed: int
    awaiting_approval: int
    manual_review: int
    defect_rate: float
    by_risk: dict[str, int]


class HealthResponse(StrictSchema):
    status: Literal["ok"]
    provider_mode: Literal["mock", "real"]


class ReadyResponse(StrictSchema):
    status: Literal["ready"]
    database: Literal["ok"]
    redis: Literal["ok", "not_required"]
