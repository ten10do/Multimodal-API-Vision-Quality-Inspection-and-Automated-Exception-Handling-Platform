from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db import Base
from app.enums import ActionStatus, ActionType, Disposition, InspectionStatus, RiskLevel


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    product_code: Mapped[str] = mapped_column(String(100), index=True)
    batch_code: Mapped[str] = mapped_column(String(100), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(255), unique=True)
    content_type: Mapped[str] = mapped_column(String(100))
    file_sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[InspectionStatus] = mapped_column(
        Enum(InspectionStatus, native_enum=False), default=InspectionStatus.QUEUED, index=True
    )
    risk_level: Mapped[RiskLevel | None] = mapped_column(
        Enum(RiskLevel, native_enum=False), nullable=True, index=True
    )
    disposition: Mapped[Disposition | None] = mapped_column(
        Enum(Disposition, native_enum=False), nullable=True
    )
    vision_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    analysis_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ModelCall(Base):
    __tablename__ = "model_calls"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    inspection_id: Mapped[UUID] = mapped_column(ForeignKey("inspections.id"), index=True)
    provider: Mapped[str] = mapped_column(String(30))
    model: Mapped[str] = mapped_column(String(120))
    request_summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    response_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    success: Mapped[bool]
    latency_ms: Mapped[int]
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WorkflowAction(Base):
    __tablename__ = "workflow_actions"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_action_idempotency"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    inspection_id: Mapped[UUID] = mapped_column(ForeignKey("inspections.id"), index=True)
    action_type: Mapped[ActionType] = mapped_column(Enum(ActionType, native_enum=False))
    status: Mapped[ActionStatus] = mapped_column(Enum(ActionStatus, native_enum=False))
    idempotency_key: Mapped[str] = mapped_column(String(180))
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_inspection_created", "inspection_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    inspection_id: Mapped[UUID] = mapped_column(ForeignKey("inspections.id"), index=True)
    actor_type: Mapped[str] = mapped_column(String(30))
    actor_id: Mapped[str] = mapped_column(String(100))
    event_type: Mapped[str] = mapped_column(String(80))
    previous_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    new_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class HumanFeedback(Base):
    __tablename__ = "human_feedback"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    inspection_id: Mapped[UUID] = mapped_column(ForeignKey("inspections.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(100))
    comment: Mapped[str] = mapped_column(Text)
    corrected_risk: Mapped[RiskLevel | None] = mapped_column(
        Enum(RiskLevel, native_enum=False), nullable=True
    )
    corrected_disposition: Mapped[Disposition | None] = mapped_column(
        Enum(Disposition, native_enum=False), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
