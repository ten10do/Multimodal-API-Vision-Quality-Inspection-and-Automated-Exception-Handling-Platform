from httpx import AsyncClient

from tests.factories import upload_image


async def test_unified_error_response_contains_request_id(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/inspections/not-a-uuid",
        headers={"X-Request-ID": "request-id-from-test"},
    )
    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "request-id-from-test"
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Request validation failed",
            "request_id": "request-id-from-test",
        }
    }


async def test_health_and_readiness_aliases(client: AsyncClient) -> None:
    health = await client.get("/health")
    ready = await client.get("/ready")
    assert health.json() == {"status": "ok", "provider_mode": "mock"}
    assert ready.json() == {
        "status": "ready",
        "database": "ok",
        "redis": "not_required",
    }


async def test_approval_is_rejected_outside_pending_state(client: AsyncClient) -> None:
    created = await upload_image(client, 0, key="invalid-approval-state")
    response = await client.post(
        f"/api/v1/inspections/{created.json()['id']}/approval",
        json={
            "decision": "approve",
            "reviewer": "supervisor",
            "comment": "should not be accepted",
        },
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_state"


async def test_dashboard_statistics_match_persisted_workflows(
    client: AsyncClient,
) -> None:
    for bucket in range(4):
        await upload_image(client, bucket, key=f"dashboard-{bucket}")
    response = await client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200
    assert response.json() == {
        "total": 4,
        "completed": 2,
        "awaiting_approval": 1,
        "manual_review": 1,
        "defect_rate": 0.75,
        "by_risk": {"low": 1, "medium": 1, "high": 1, "critical": 1},
    }
