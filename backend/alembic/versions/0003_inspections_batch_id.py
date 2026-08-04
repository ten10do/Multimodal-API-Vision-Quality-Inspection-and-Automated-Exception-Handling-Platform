"""add batch_id to inspections for pipeline traceability

Revision ID: 0003_inspections_batch_id
Revises: 0002_quality_rules_unique
Create Date: 2026-08-04

The orchestrator sends batch_id with every inspection request; persisting it
on the inspection row allows batch-level traceability and the pipeline E2E
DB-count assertion.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_inspections_batch_id"
down_revision = "0002_quality_rules_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inspections", sa.Column("batch_id", sa.String(length=64), nullable=True))
    op.create_index("ix_inspections_batch_id", "inspections", ["batch_id"])


def downgrade() -> None:
    op.drop_index("ix_inspections_batch_id", table_name="inspections")
    op.drop_column("inspections", "batch_id")
