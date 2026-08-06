"""phase6: anomaly (PatchCore) columns on inspections + review_tasks

Revision ID: 0006_anomaly_columns
Revises: 0005_review_task_unique
Create Date: 2026-08-05

- inspections: anomaly_score / anomaly_threshold / is_anomalous /
  anomaly_map_path / anomaly_model_version / fusion_class
- review_tasks: anomaly snapshot for the review UI (score / threshold /
  is_anomalous / regions / map url)
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0006_anomaly_columns"
down_revision = "0005_review_task_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NOTE: anomaly_score already exists on inspections (0001_create_core_tables
    # carries it, matching the models); adding it here too would fail on a
    # fresh database. Only the columns 0001 does NOT have are added.
    op.add_column("inspections", sa.Column("anomaly_threshold", sa.Float(), nullable=True))
    op.add_column("inspections", sa.Column("is_anomalous", sa.Boolean(), nullable=True))
    op.add_column("inspections", sa.Column("anomaly_map_path", sa.String(length=512), nullable=True))
    op.add_column("inspections", sa.Column("anomaly_model_version", sa.String(length=64), nullable=True))
    op.add_column("inspections", sa.Column("anomaly_regions", sa.JSON(), nullable=True))
    op.add_column("inspections", sa.Column("fusion_class", sa.String(length=32), nullable=True))

    op.add_column("review_tasks", sa.Column("anomaly_score", sa.Float(), nullable=True))
    op.add_column("review_tasks", sa.Column("anomaly_threshold", sa.Float(), nullable=True))
    op.add_column("review_tasks", sa.Column("is_anomalous", sa.Boolean(), nullable=True))
    op.add_column("review_tasks", sa.Column("anomaly_regions", sa.JSON(), nullable=True))
    op.add_column("review_tasks", sa.Column("anomaly_map_url", sa.String(length=512), nullable=True))


def downgrade() -> None:
    for col in ("anomaly_map_url", "anomaly_regions", "is_anomalous", "anomaly_threshold", "anomaly_score"):
        op.drop_column("review_tasks", col)
    for col in ("fusion_class", "anomaly_model_version", "anomaly_map_path", "is_anomalous", "anomaly_threshold"):
        op.drop_column("inspections", col)
