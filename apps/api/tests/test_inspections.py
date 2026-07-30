from httpx import AsyncClient

from tests.factories import png_for_bucket


async def upload(
    client: AsyncClient,
    content: bytes,
    *,
    key: str,
    batch_code: str = "B20260730",
) -> object:
    return await client.post(
        "/api/v1/inspections",
        headers={"Idempotency-Key": key},
        data={"product_code": "AX-240", "batch_code": batch_code},
        files={"image": ("part.png", content, "image/png")},
    )


async def test_full_mock_workflow_is_persisted_and_idempotent(client: AsyncClient) -> None:
    response = await upload(client, png_for_bucket(2), key="same-request-001")
    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["risk_level"] == "high"
    assert {action["action_type"] for action in payload["actions"]} == {
        "reject_product",
        "create_ticket",
        "send_notification",
    }
    assert any(log["event_type"] == "workflow_actions_executed" for log in payload["audit_logs"])

    repeated = await upload(client, png_for_bucket(2), key="same-request-001")
    assert repeated.status_code == 201
    assert repeated.json()["id"] == payload["id"]

    conflict = await upload(
        client,
        png_for_bucket(2),
        key="same-request-001",
        batch_code="DIFFERENT-BATCH",
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"


async def test_critical_stop_requires_human_approval(client: AsyncClient) -> None:
    created = await upload(client, png_for_bucket(3), key="critical-request-01")
    assert created.status_code == 201
    payload = created.json()
    assert payload["status"] == "awaiting_approval"
    assert payload["disposition"] == "stop_line"
    assert not any(action["action_type"] == "execute_line_stop" for action in payload["actions"])

    approved = await client.post(
        f"/api/v1/inspections/{payload['id']}/approval",
        json={
            "decision": "approve",
            "reviewer": "shift-lead",
            "comment": "现场确认存在批次性风险",
        },
    )
    assert approved.status_code == 200
    approved_payload = approved.json()
    assert approved_payload["status"] == "completed"
    assert any(
        action["action_type"] == "execute_line_stop" for action in approved_payload["actions"]
    )


async def test_invalid_image_content_is_rejected(client: AsyncClient) -> None:
    response = await upload(client, b"not-a-real-png", key="invalid-request-01")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


async def test_human_feedback_is_audited(client: AsyncClient) -> None:
    created = await upload(client, png_for_bucket(0), key="feedback-request-1")
    inspection_id = created.json()["id"]
    response = await client.post(
        f"/api/v1/inspections/{inspection_id}/feedback",
        json={
            "reviewer": "quality-engineer",
            "comment": "抽检确认无缺陷",
            "corrected_risk": "low",
            "corrected_disposition": "release",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["feedback"][0]["reviewer"] == "quality-engineer"
    assert any(log["event_type"] == "human_feedback_recorded" for log in payload["audit_logs"])
