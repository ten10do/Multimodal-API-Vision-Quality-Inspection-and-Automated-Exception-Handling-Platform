from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.enums import ActionStatus, ActionType
from app.models import WorkflowAction


class ToolAuthorizationError(RuntimeError):
    pass


class SimulatedToolExecutor:
    async def execute(
        self,
        session: AsyncSession,
        inspection_id: UUID,
        action_type: ActionType,
        payload: dict[str, object],
        *,
        pending_approval: bool = False,
    ) -> WorkflowAction:
        if action_type == ActionType.EXECUTE_LINE_STOP:
            approval = await session.scalar(
                select(WorkflowAction).where(
                    WorkflowAction.inspection_id == inspection_id,
                    WorkflowAction.action_type == ActionType.REQUEST_LINE_STOP,
                    WorkflowAction.status == ActionStatus.SUCCEEDED,
                )
            )
            if approval is None:
                raise ToolAuthorizationError("Line stop cannot execute before human approval")
        key = f"{inspection_id}:{action_type.value}"
        existing = await session.scalar(
            select(WorkflowAction).where(WorkflowAction.idempotency_key == key)
        )
        if existing is not None:
            return existing

        status = ActionStatus.PENDING_APPROVAL if pending_approval else ActionStatus.SUCCEEDED
        reference = f"SIM-{action_type.value.upper()}-{uuid4().hex[:8]}"
        action = WorkflowAction(
            inspection_id=inspection_id,
            action_type=action_type,
            status=status,
            idempotency_key=key,
            request_payload=payload,
            result_payload={
                "simulated": True,
                "reference": reference,
                "message": "动作已记录" if not pending_approval else "等待人工审批",
            },
        )
        session.add(action)
        await session.flush()
        return action
