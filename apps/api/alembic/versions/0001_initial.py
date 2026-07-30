"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-07-30
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Metadata is the single schema source; this keeps SQLite and PostgreSQL aligned.
    from app import models  # noqa: F401
    from app.db import Base

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    from app import models  # noqa: F401
    from app.db import Base

    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
