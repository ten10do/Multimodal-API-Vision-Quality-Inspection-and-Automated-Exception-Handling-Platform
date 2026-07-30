from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLog


def add_audit(
    session: AsyncSession,
    inspection_id: UUID,
    *,
    event_type: str,
    actor_type: str = "system",
    actor_id: str = "workflow",
    previous_state: dict[str, Any] | None = None,
    new_state: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    session.add(
        AuditLog(
            inspection_id=inspection_id,
            actor_type=actor_type,
            actor_id=actor_id,
            event_type=event_type,
            previous_state=previous_state,
            new_state=new_state,
            detail=detail or {},
        )
    )
