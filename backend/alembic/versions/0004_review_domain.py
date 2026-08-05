"""phase5: human review domain (review_tasks / review_decisions / review_corrections)

Revision ID: 0004_review_domain
Revises: 0003_inspections_batch_id
Create Date: 2026-08-05

- inspections.final_quality_result: business-final quality, distinct from the
  immutable AI quality_result
- review_tasks: one active (non-RESOLVED) task per inspection enforced by a
  partial unique index; AI snapshot frozen at creation
- review_decisions: immutable human decision records
- review_corrections: audit revisions appended after RESOLVED (no silent
  overwrite)
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004_review_domain"
down_revision = "0003_inspections_batch_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inspections", sa.Column("final_quality_result", sa.String(length=8), nullable=True))

    op.create_table(
        "review_tasks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("review_task_id", sa.String(length=64), nullable=False),
        sa.Column("inspection_id", sa.Uuid(), sa.ForeignKey("inspections.id"), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("assigned_to", sa.String(length=64), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("ai_quality_result", sa.String(length=16), nullable=False),
        sa.Column("ai_defects_snapshot", sa.JSON(), nullable=False),
        sa.Column("ai_model_version", sa.String(length=64), nullable=True),
        sa.Column("ai_rule_version", sa.Integer(), nullable=True),
        sa.Column("ai_severity", sa.String(length=16), nullable=True),
        sa.Column("product_id", sa.String(length=64), nullable=False),
        sa.Column("production_line", sa.String(length=64), nullable=False),
        sa.Column("station", sa.String(length=64), nullable=False),
        sa.Column("batch_id", sa.String(length=64), nullable=True),
        sa.Column("image_url", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_review_tasks_review_task_id", "review_tasks", ["review_task_id"], unique=True)
    op.create_index("ix_review_tasks_inspection_id", "review_tasks", ["inspection_id"])
    op.create_index("ix_review_tasks_status", "review_tasks", ["status"])
    op.create_index("ix_review_tasks_product_id", "review_tasks", ["product_id"])
    op.create_index("ix_review_tasks_batch_id", "review_tasks", ["batch_id"])
    # 一个 inspection 最多一个 active review task（5B）
    op.create_index(
        "uq_review_task_active_inspection",
        "review_tasks",
        ["inspection_id"],
        unique=True,
        postgresql_where=sa.text("status != 'RESOLVED'"),
        sqlite_where=sa.text("status != 'RESOLVED'"),
    )

    op.create_table(
        "review_decisions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("review_task_id", sa.Uuid(), sa.ForeignKey("review_tasks.id"), nullable=False, unique=True),
        sa.Column("inspection_id", sa.Uuid(), sa.ForeignKey("inspections.id"), nullable=False),
        sa.Column("reviewer", sa.String(length=64), nullable=False),
        sa.Column("ai_quality_result", sa.String(length=16), nullable=False),
        sa.Column("ai_defects_snapshot", sa.JSON(), nullable=False),
        sa.Column("human_decision", sa.String(length=32), nullable=False),
        sa.Column("human_label", sa.String(length=256), nullable=True),
        sa.Column("final_quality_result", sa.String(length=8), nullable=False),
        sa.Column("reason", sa.String(length=1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_review_decisions_inspection_id", "review_decisions", ["inspection_id"])

    op.create_table(
        "review_corrections",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("review_decision_id", sa.Uuid(), sa.ForeignKey("review_decisions.id"), nullable=False),
        sa.Column("reviewer", sa.String(length=64), nullable=False),
        sa.Column("field_changed", sa.String(length=64), nullable=False),
        sa.Column("old_value", sa.JSON(), nullable=True),
        sa.Column("new_value", sa.JSON(), nullable=True),
        sa.Column("reason", sa.String(length=1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_review_corrections_decision_id", "review_corrections", ["review_decision_id"])


def downgrade() -> None:
    op.drop_table("review_corrections")
    op.drop_table("review_decisions")
    op.drop_index("uq_review_task_active_inspection", table_name="review_tasks")
    op.drop_table("review_tasks")
    op.drop_column("inspections", "final_quality_result")
