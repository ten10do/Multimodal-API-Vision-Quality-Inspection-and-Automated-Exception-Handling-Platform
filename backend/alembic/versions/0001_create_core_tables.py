"""create core tables (batches, products, inspections, defects, quality_rules)

Revision ID: 0001_create_core_tables
Revises:
Create Date: 2026-08-04

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_create_core_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "batches",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("batch_id", sa.String(length=64), nullable=False),
        sa.Column("production_line", sa.String(length=64), nullable=False),
        sa.Column("product_type", sa.String(length=64), nullable=False),
        sa.Column("target_qty", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("OPEN", "CLOSED", name="batchstatus", native_enum=False, create_constraint=True),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_batches_batch_id", "batches", ["batch_id"], unique=True)

    op.create_table(
        "products",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("product_id", sa.String(length=64), nullable=False),
        sa.Column("batch_id", sa.Uuid(), nullable=True),
        sa.Column("production_line", sa.String(length=64), nullable=False),
        sa.Column("station", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["batch_id"], ["batches.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_products_product_id", "products", ["product_id"], unique=True)

    op.create_table(
        "inspections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("inspection_id", sa.String(length=64), nullable=False),
        sa.Column("product_id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column(
            "status",
            sa.Enum("PENDING", "COMPLETED", "FAILED", name="inspectionstatus", native_enum=False, create_constraint=True),
            nullable=False,
        ),
        sa.Column(
            "quality_result",
            sa.Enum("PASS", "REVIEW", "FAIL", name="qualityresult", native_enum=False, create_constraint=True),
            nullable=True,
        ),
        sa.Column(
            "severity",
            sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="severity", native_enum=False, create_constraint=True),
            nullable=True,
        ),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("model_name", sa.String(length=64), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("rule_version", sa.Integer(), nullable=True),
        sa.Column("inference_latency_ms", sa.Float(), nullable=True),
        sa.Column("inference_request_id", sa.String(length=64), nullable=True),
        sa.Column("image_path", sa.String(length=512), nullable=True),
        sa.Column("error_message", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_inspections_inspection_id", "inspections", ["inspection_id"], unique=True)
    op.create_index("ix_inspections_idempotency_key", "inspections", ["idempotency_key"], unique=True)
    op.create_index("ix_inspections_product_id", "inspections", ["product_id"])

    op.create_table(
        "defects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("inspection_id", sa.Uuid(), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("class_name", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox_xyxy", sa.JSON(), nullable=False),
        sa.Column("bbox_normalized", sa.JSON(), nullable=False),
        sa.Column("defect_area_px", sa.Float(), nullable=False),
        sa.Column("defect_area_ratio", sa.Float(), nullable=False),
        sa.Column(
            "severity",
            sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="severity", native_enum=False, create_constraint=True),
            nullable=True,
        ),
        sa.Column("matched_rule", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["inspection_id"], ["inspections.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_defects_inspection_id", "defects", ["inspection_id"])

    op.create_table(
        "quality_rules",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("defect_type", sa.String(length=64), nullable=False),
        sa.Column("min_confidence", sa.Float(), nullable=False),
        sa.Column("max_area_ratio", sa.Float(), nullable=False),
        sa.Column(
            "action",
            sa.Enum("PASS", "REVIEW", "FAIL", name="qualityresult", native_enum=False, create_constraint=True),
            nullable=False,
        ),
        sa.Column(
            "severity",
            sa.Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="severity", native_enum=False, create_constraint=True),
            nullable=False,
        ),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("rule_version", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("quality_rules")
    op.drop_table("defects")
    op.drop_table("inspections")
    op.drop_table("products")
    op.drop_table("batches")
