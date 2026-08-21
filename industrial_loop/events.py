"""Phase 1 — unified inspection event schema for the industrial closed loop.

Every product passing through the loop produces exactly one traceable
``InspectionEvent``. The event is created by the decision engine and then
enriched in place (immutable copies) as PLC / MES / human-review states
advance, so the full lifecycle of one product is reconstructable from the
event log alone (trace_id + id + timestamps).
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import EVENT_SCHEMA_VERSION


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_event_id() -> str:
    return f"evt-{uuid.uuid4().hex[:20]}"


def new_trace_id() -> str:
    return f"trc-{uuid.uuid4().hex[:20]}"


class Decision(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    HOLD = "HOLD"


class ReasonCode(str, Enum):
    NORMAL = "NORMAL"
    DEFECT_DETECTED = "DEFECT_DETECTED"
    AI_SYSTEM_FAILURE = "AI_SYSTEM_FAILURE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PLC_UNACKNOWLEDGED = "PLC_UNACKNOWLEDGED"


class OperatorStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    CONFIRMED_DEFECT = "CONFIRMED_DEFECT"
    FALSE_ALARM = "FALSE_ALARM"
    RECHECK_REQUESTED = "RECHECK_REQUESTED"


class PlcStatus(str, Enum):
    NOT_APPLIED = "NOT_APPLIED"
    ACK_RUNNING = "ACK_RUNNING"
    ACK_REJECT_SIGNAL = "ACK_REJECT_SIGNAL"
    ACK_STOP_SIGNAL = "ACK_STOP_SIGNAL"
    NACK = "NACK"
    UNREACHABLE = "UNREACHABLE"


class MesStatus(str, Enum):
    NOT_CREATED = "NOT_CREATED"
    OPEN = "OPEN"
    PROCESSING = "PROCESSING"
    CLOSED = "CLOSED"
    SYNC_FAILED = "SYNC_FAILED"


#: decision -> the only reason codes that may accompany it.
_DECISION_REASONS: dict[Decision, set[ReasonCode]] = {
    Decision.PASS: {ReasonCode.NORMAL},
    Decision.REJECT: {ReasonCode.DEFECT_DETECTED},
    Decision.HOLD: {
        ReasonCode.AI_SYSTEM_FAILURE,
        ReasonCode.LOW_CONFIDENCE,
        ReasonCode.PLC_UNACKNOWLEDGED,
    },
}


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a number or null")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return value


class InspectionEvent(BaseModel):
    """One traceable unit of production inspection (Phase 1 contract)."""

    schema_version: str = EVENT_SCHEMA_VERSION
    id: str = Field(default_factory=new_event_id)
    trace_id: str = Field(default_factory=new_trace_id)
    timestamp: str = Field(default_factory=utc_now_iso)
    product_id: str
    batch_id: str
    camera_id: str
    model_version: str
    artifact_version: str
    image_score: float | None = None
    pixel_score: float | None = None
    threshold: float | None = None
    confidence: dict | None = None
    decision: Decision
    reason_code: ReasonCode
    heatmap_reference: str | None = None
    operator_status: OperatorStatus = OperatorStatus.NOT_REQUIRED
    plc_status: PlcStatus = PlcStatus.NOT_APPLIED
    mes_status: MesStatus = MesStatus.NOT_CREATED
    error_detail: str | None = None
    latency_ms: float | None = None

    @field_validator("image_score", "pixel_score", "threshold", mode="before")
    @classmethod
    def _scores_finite(cls, value):  # noqa: ANN001
        return _finite_or_none(value)

    @model_validator(mode="after")
    def _validate_pairing(self) -> "InspectionEvent":
        if self.reason_code not in _DECISION_REASONS[self.decision]:
            raise ValueError(
                f"decision {self.decision.value} cannot carry reason {self.reason_code.value}"
            )
        if self.decision is Decision.HOLD:
            if self.image_score is not None and self.threshold is not None:
                pass  # low-confidence holds still carry both scores
            if self.error_detail is None and self.reason_code is ReasonCode.AI_SYSTEM_FAILURE:
                raise ValueError("AI_SYSTEM_FAILURE hold must record error_detail")
        else:
            for name in ("image_score", "threshold"):
                if getattr(self, name) is None:
                    raise ValueError(f"{name} is required for {self.decision.value}")
        if self.decision is Decision.REJECT and self.threshold is not None:
            if self.image_score < self.threshold:
                raise ValueError("REJECT requires image_score >= threshold")
        if self.decision is Decision.PASS and self.threshold is not None:
            if self.image_score >= self.threshold:
                raise ValueError("PASS requires image_score < threshold")
        return self

    def with_updates(self, **fields) -> "InspectionEvent":
        """Return an enriched copy; the original event stays immutable."""
        data = self.model_dump()
        data.update(fields)
        return InspectionEvent.model_validate(data)

    def short(self) -> dict:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "product_id": self.product_id,
            "batch_id": self.batch_id,
            "camera_id": self.camera_id,
            "decision": self.decision.value,
            "reason_code": self.reason_code.value,
            "image_score": self.image_score,
            "pixel_score": self.pixel_score,
            "operator_status": self.operator_status.value,
            "plc_status": self.plc_status.value,
            "mes_status": self.mes_status.value,
        }
