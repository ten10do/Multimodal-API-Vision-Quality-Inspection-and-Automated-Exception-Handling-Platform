import json
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from app.config import get_settings
from app.db import SessionLocal
from app.enums import ActionType, InspectionStatus
from app.models import ModelCall, WorkflowAction
from app.providers.base import ProviderError
from app.tools import SimulatedToolExecutor, ToolAuthorizationError
from app.workflow import (
    InvalidStateTransition,
    process_inspection,
    validate_transition,
)
from tests.factories import upload_image


@pytest.mark.parametrize(
    ("previous", "target"),
    [
        (InspectionStatus.QUEUED, InspectionStatus.VISION_ANALYZING),
        (InspectionStatus.VISION_ANALYZING, InspectionStatus.REASONING),
        (InspectionStatus.REASONING, InspectionStatus.EXECUTING),
        (InspectionStatus.EXECUTING, InspectionStatus.COMPLETED),
        (InspectionStatus.EXECUTING, InspectionStatus.MANUAL_REVIEW),
        (InspectionStatus.EXECUTING, InspectionStatus.AWAITING_APPROVAL),
        (InspectionStatus.AWAITING_APPROVAL, InspectionStatus.COMPLETED),
    ],
)
def test_state_machine_allows_declared_transitions(
    previous: InspectionStatus, target: InspectionStatus
) -> None:
    validate_transition(previous, target)


@pytest.mark.parametrize(
    ("previous", "target"),
    [
        (InspectionStatus.QUEUED, InspectionStatus.COMPLETED),
        (InspectionStatus.COMPLETED, InspectionStatus.REASONING),
        (InspectionStatus.AWAITING_APPROVAL, InspectionStatus.EXECUTING),
        (InspectionStatus.MANUAL_REVIEW, InspectionStatus.VISION_ANALYZING),
    ],
)
def test_state_machine_rejects_illegal_transitions(
    previous: InspectionStatus, target: InspectionStatus
) -> None:
    with pytest.raises(InvalidStateTransition):
        validate_transition(previous, target)


async def test_pass_automatically_releases_product(client: AsyncClient) -> None:
    response = await upload_image(client, 0, key="workflow-pass")
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["disposition"] == "release"
    assert [action["action_type"] for action in payload["actions"]] == ["release_product"]


async def test_review_creates_ticket_and_manual_review(client: AsyncClient) -> None:
    response = await upload_image(client, 1, key="workflow-review")
    payload = response.json()
    assert payload["status"] == "manual_review"
    assert {action["action_type"] for action in payload["actions"]} == {
        "manual_review",
        "create_ticket",
    }


async def test_fail_rejects_product_and_creates_ticket(client: AsyncClient) -> None:
    response = await upload_image(client, 2, key="workflow-fail")
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["disposition"] == "reject"
    assert {action["action_type"] for action in payload["actions"]} == {
        "reject_product",
        "create_ticket",
        "send_notification",
    }


async def test_critical_requests_stop_without_executing_it(client: AsyncClient) -> None:
    response = await upload_image(client, 3, key="workflow-critical")
    payload = response.json()
    assert payload["status"] == "awaiting_approval"
    actions = {action["action_type"]: action for action in payload["actions"]}
    assert actions["request_line_stop"]["status"] == "pending_approval"
    assert "execute_line_stop" not in actions


async def test_line_stop_tool_rejects_unapproved_execution(client: AsyncClient) -> None:
    response = await upload_image(client, 3, key="unapproved-tool")
    inspection_id = UUID(response.json()["id"])
    async with SessionLocal() as session:
        with pytest.raises(ToolAuthorizationError):
            await SimulatedToolExecutor().execute(
                session,
                inspection_id,
                ActionType.EXECUTE_LINE_STOP,
                {"reviewer": "bypass-attempt"},
            )


async def test_tool_call_and_ticket_creation_are_idempotent(
    client: AsyncClient,
) -> None:
    response = await upload_image(client, 2, key="tool-idempotency")
    inspection_id = UUID(response.json()["id"])
    async with SessionLocal() as session:
        executor = SimulatedToolExecutor()
        first = await executor.execute(
            session, inspection_id, ActionType.CREATE_TICKET, {"attempt": 1}
        )
        second = await executor.execute(
            session, inspection_id, ActionType.CREATE_TICKET, {"attempt": 2}
        )
        assert first.id == second.id
        count = await session.scalar(
            select(func.count())
            .select_from(WorkflowAction)
            .where(
                WorkflowAction.inspection_id == inspection_id,
                WorkflowAction.action_type == ActionType.CREATE_TICKET,
            )
        )
        assert count == 1

    await process_inspection(inspection_id)
    detail = await client.get(f"/api/v1/inspections/{inspection_id}")
    assert sum(action["action_type"] == "create_ticket" for action in detail.json()["actions"]) == 1


async def test_repeated_schema_failure_safely_falls_back_to_review(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingVisionProvider:
        async def inspect(self, image: bytes, context: object) -> object:
            raise ProviderError(
                "provider_unavailable",
                "Vision provider unavailable; routed to manual review",
            )

    monkeypatch.setattr(
        "app.workflow.get_vision_provider", lambda settings: FailingVisionProvider()
    )
    response = await upload_image(client, 0, key="safe-fallback")
    payload = response.json()
    assert payload["status"] == "manual_review"
    assert payload["error_code"] == "provider_unavailable"
    assert {action["action_type"] for action in payload["actions"]} == {"manual_review"}
    assert any(log["event_type"] == "safe_fallback" for log in payload["audit_logs"])


async def test_model_call_audit_does_not_persist_api_keys(client: AsyncClient) -> None:
    secret = "sentinel-real-key-never-log"
    settings = get_settings()
    previous_bailian = settings.bailian_api_key
    previous_deepseek = settings.deepseek_api_key
    settings.bailian_api_key = secret
    settings.deepseek_api_key = secret
    try:
        await upload_image(client, 2, key="redacted-model-log")
        async with SessionLocal() as session:
            calls = list(await session.scalars(select(ModelCall)))
            serialized = json.dumps(
                [
                    {
                        "request": call.request_summary,
                        "response": call.response_payload,
                        "error": call.error_code,
                    }
                    for call in calls
                ]
            )
        assert secret not in serialized
        assert "Authorization" not in serialized
    finally:
        settings.bailian_api_key = previous_bailian
        settings.deepseek_api_key = previous_deepseek
