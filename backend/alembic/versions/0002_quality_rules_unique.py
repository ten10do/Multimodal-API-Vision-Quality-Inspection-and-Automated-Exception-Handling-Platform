"""add unique constraint on quality_rules business key

Revision ID: 0002_quality_rules_unique
Revises: 0001_create_core_tables
Create Date: 2026-08-04

A business rule is uniquely identified by (defect_type, priority, rule_version):
the same defect_type cannot have two rules with the same priority in the same
version, which keeps the engine's priority ordering deterministic and prevents
duplicate seeding.
"""
from __future__ import annotations

from alembic import op

revision = "0002_quality_rules_unique"
down_revision = "0001_create_core_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_quality_rules_defect_type_priority_version",
        "quality_rules",
        ["defect_type", "priority", "rule_version"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_quality_rules_defect_type_priority_version", "quality_rules", type_="unique")
