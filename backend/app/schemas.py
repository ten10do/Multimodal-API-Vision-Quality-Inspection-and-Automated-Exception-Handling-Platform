from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import InspectionStatus, QualityResult, Severity


class RuleBase(BaseModel):
    defect_type: str = Field(min_length=1, description="class_name or '*' for any")
    min_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    max_area_ratio: float = Field(ge=0.0, le=1.0, default=1.0)
    action: QualityResult
    severity: Severity
    priority: int = Field(default=100, ge=0, description="lower is evaluated first")
    enabled: bool = True


class RuleCreate(RuleBase):
    rule_version: int = Field(default=1, ge=1)


class RuleUpdate(BaseModel):
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_area_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    action: QualityResult | None = None
    severity: Severity | None = None
    priority: int | None = Field(default=None, ge=0)
    enabled: bool | None = None


class RuleOut(RuleBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    rule_version: int
    created_at: datetime
    updated_at: datetime


class DefectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_normalized: list[float]
    defect_area_px: float
    defect_area_ratio: float
    severity: Severity | None
    matched_rule: str | None


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    product_id: str
    production_line: str
    station: str
    created_at: datetime


class InspectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    inspection_id: str
    product_id: str
    batch_id: str | None = None
    status: InspectionStatus
    quality_result: QualityResult | None
    severity: Severity | None
    model_name: str | None
    model_version: str | None
    rule_version: int | None
    inference_latency_ms: float | None
    error_message: str | None
    created_at: datetime
    defects: list[DefectOut] = Field(default_factory=list)


class InspectionDetail(InspectionOut):
    product: ProductOut


class ErrorBody(BaseModel):
    error: dict[str, Any]
