"""phase8: model registry + deployment manifest

Revision ID: 0008_model_registry
Revises: 0007_industrial_integration
Create Date: 2026-08-06

- model_registry: model identity (name/version/type/artifact/sha256/
  dataset_version/training_run_id/status), PRODUCTION uniqueness per
  model_name via partial unique index, dataset registry for dataset
  versioning (manifest + sha256)
- inspections.deployment_version: which AI stack version judged this
  inspection (8D)
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0008_model_registry"
down_revision = "0007_industrial_integration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_registry",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_type", sa.String(length=32), nullable=False),  # yolo | patchcore
        sa.Column("artifact_uri", sa.String(length=512), nullable=True),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("dataset_version", sa.String(length=64), nullable=True),
        sa.Column("training_run_id", sa.String(length=128), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16), nullable=False,
            server_default="CANDIDATE",
        ),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("domain_validated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("notes", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_model_registry_name_version", "model_registry", ["model_name", "model_version"], unique=True)
    # at most one PRODUCTION model per model_name (8C)
    op.create_index(
        "uq_model_registry_active_production",
        "model_registry",
        ["model_name"],
        unique=True,
        postgresql_where=sa.text("status = 'PRODUCTION'"),
        sqlite_where=sa.text("status = 'PRODUCTION'"),
    )

    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("dataset_version", sa.String(length=64), nullable=False, unique=True),
        sa.Column("manifest_uri", sa.String(length=512), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("description", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.add_column("inspections", sa.Column("deployment_version", sa.String(length=64), nullable=True))
    op.add_column("inspections", sa.Column("model_registry_id", sa.Uuid(), nullable=True))


def downgrade() -> None:
    op.drop_column("inspections", "model_registry_id")
    op.drop_column("inspections", "deployment_version")
    op.drop_table("dataset_versions")
    op.drop_index("uq_model_registry_active_production", table_name="model_registry")
    op.drop_index("ix_model_registry_name_version", table_name="model_registry")
    op.drop_table("model_registry")
