from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .enums import BatchStatus, InspectionStatus, QualityResult, Severity


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Batch(TimestampMixin, Base):
    __tablename__ = "batches"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    batch_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    production_line: Mapped[str] = mapped_column(String(64), nullable=False)
    product_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[BatchStatus] = mapped_column(SAEnum(BatchStatus, native_enum=False), nullable=False, default=BatchStatus.OPEN)

    products: Mapped[list["Product"]] = relationship(back_populates="batch")


class Product(TimestampMixin, Base):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    product_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    batch_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("batches.id"), nullable=True)
    production_line: Mapped[str] = mapped_column(String(64), nullable=False)
    station: Mapped[str] = mapped_column(String(64), nullable=False)

    batch: Mapped[Batch | None] = relationship(back_populates="products")
    inspections: Mapped[list["Inspection"]] = relationship(back_populates="product")


class Inspection(TimestampMixin, Base):
    __tablename__ = "inspections"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    product_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("products.id"), index=True, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, index=True, nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    status: Mapped[InspectionStatus] = mapped_column(
        SAEnum(InspectionStatus, native_enum=False), nullable=False, default=InspectionStatus.PENDING
    )
    quality_result: Mapped[QualityResult | None] = mapped_column(SAEnum(QualityResult, native_enum=False), nullable=True)
    severity: Mapped[Severity | None] = mapped_column(SAEnum(Severity, native_enum=False), nullable=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rule_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    product: Mapped[Product] = relationship(back_populates="inspections")
    defects: Mapped[list["Defect"]] = relationship(back_populates="inspection", cascade="all, delete-orphan")


class Defect(TimestampMixin, Base):
    __tablename__ = "defects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    inspection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspections.id"), index=True, nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_xyxy: Mapped[list] = mapped_column(JSON, nullable=False)
    bbox_normalized: Mapped[list] = mapped_column(JSON, nullable=False)
    defect_area_px: Mapped[float] = mapped_column(Float, nullable=False)
    defect_area_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[Severity | None] = mapped_column(SAEnum(Severity, native_enum=False), nullable=True)
    matched_rule: Mapped[str | None] = mapped_column(String(64), nullable=True)

    inspection: Mapped[Inspection] = relationship(back_populates="defects")


class QualityRule(TimestampMixin, Base):
    __tablename__ = "quality_rules"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    defect_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="class_name or '*' for any")
    min_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_area_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    action: Mapped[QualityResult] = mapped_column(SAEnum(QualityResult, native_enum=False), nullable=False)
    severity: Mapped[Severity] = mapped_column(SAEnum(Severity, native_enum=False), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100, comment="lower is evaluated first")
    rule_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
