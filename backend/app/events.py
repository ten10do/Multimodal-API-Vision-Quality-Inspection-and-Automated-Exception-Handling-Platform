"""Business event DTO for the realtime WebSocket channel.

Events are DTOs, deliberately decoupled from database ORM models. They carry
the pipeline status (process_status) separately from the quality judgement
(quality_result) so FAILED (system processing failure) can never be confused
with FAIL (product judged unqualified after a completed inspection).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field
from vision_contract import utc_now_iso

from .enums import QualityResult, Severity

EventType = Literal["inspection.completed", "inspection.failed"]

# Pipeline status model (3C). CAPTURED / QUEUED / PROCESSING are tracked by
# the orchestrator locally; COMPLETED / FAILED are terminal states persisted
# by the backend and broadcast here.
PipelineStatus = Literal["CAPTURED", "QUEUED", "PROCESSING", "COMPLETED", "FAILED"]

ReviewEventType = Literal["review.created", "review.claimed", "review.resolved"]


class ReviewEvent(BaseModel):
    """Review lifecycle notification (5I). DB remains the source of truth;
    these events only notify subscribers."""

    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    event_type: ReviewEventType
    timestamp: str = Field(default_factory=utc_now_iso)
    review_task_id: str
    inspection_id: str
    product_id: str
    status: str  # PENDING / IN_REVIEW / RESOLVED
    priority: int | None = None
    assigned_to: str | None = None
    reviewer: str | None = None
    human_decision: str | None = None
    final_quality_result: str | None = None
    top_defect_class: str | None = None
    top_confidence: float | None = None
    severity: str | None = None
    model_version: str | None = None
    image_url: str | None = None
    # Phase 6 anomaly snapshot
    anomaly_score: float | None = None
    is_anomalous: bool | None = None
    anomaly_map_url: str | None = None

    def to_broadcast(self) -> dict:
        return self.model_dump()


class InspectionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    event_type: EventType
    timestamp: str = Field(default_factory=utc_now_iso)
    product_id: str
    inspection_id: str
    batch_id: str | None = None
    production_line: str
    station: str
    process_status: PipelineStatus
    quality_result: QualityResult | None = None
    severity: Severity | None = None
    defect_count: int = 0
    inference_latency_ms: float | None = None
    model_version: str | None = None
    error_message: str | None = None

    def to_broadcast(self) -> dict:
        data = self.model_dump()
        data["timestamp"] = self.timestamp
        data["quality_result"] = self.quality_result.value if self.quality_result else None
        data["severity"] = self.severity.value if self.severity else None
        return data


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
