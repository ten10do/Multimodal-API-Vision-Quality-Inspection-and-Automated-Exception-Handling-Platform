"""phase6 pre-gate: review_tasks per-inspection global uniqueness

Revision ID: 0005_review_task_unique
Revises: 0004_review_domain
Create Date: 2026-08-05

The system does not support re-review (a RESOLVED task has no reopen path),
so the partial unique index on active tasks is replaced by a global unique
constraint: at most ONE review task per inspection over its lifetime.
Corrections of a resolved decision keep using review_corrections.
"""
from __future__ import annotations

from alembic import op

revision = "0005_review_task_unique"
down_revision = "0004_review_domain"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("uq_review_task_active_inspection", table_name="review_tasks")
    op.create_index(
        "uq_review_task_inspection",
        "review_tasks",
        ["inspection_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_review_task_inspection", table_name="review_tasks")
    op.create_index(
        "uq_review_task_active_inspection",
        "review_tasks",
        ["inspection_id"],
        unique=True,
        postgresql_where=op.f("status") != "RESOLVED",
    )
