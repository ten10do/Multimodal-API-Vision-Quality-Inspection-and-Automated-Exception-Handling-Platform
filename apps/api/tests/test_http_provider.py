import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.providers.base import ProviderError
from app.providers.http import OpenAICompatibleClient, _extract_json
from app.schemas import AnalysisResult


def analysis_payload() -> dict[str, object]:
    return {
        "risk_level": "low",
        "probable_causes": ["none"],
        "recommended_actions": ["release"],
        "disposition": "release",
        "requires_human_approval": False,
        "rationale": "meets quality rules",
    }


def client(transport: httpx.AsyncBaseTransport, *, max_retries: int) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name="test-provider",
        api_key="test-secret-must-not-leak",
        base_url="https://provider.invalid/v1",
        model="test-model",
        timeout_seconds=0.01,
        max_retries=max_retries,
        transport=transport,
    )


async def test_provider_timeout_retries_then_returns_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def timeout_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timed out", request=request)

    sleep = AsyncMock()
    monkeypatch.setattr("app.providers.http.asyncio.sleep", sleep)
    provider_client = client(httpx.MockTransport(timeout_handler), max_retries=2)
    with pytest.raises(ProviderError) as error:
        await provider_client.complete([], AnalysisResult)
    assert error.value.code == "provider_unavailable"
    assert attempts == 3
    assert sleep.await_count == 2
    assert "test-secret-must-not-leak" not in error.value.safe_message


async def test_provider_retries_transient_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(analysis_payload())}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_cache_hit_tokens": 2,
                    "prompt_cache_miss_tokens": 8,
                },
            },
        )

    monkeypatch.setattr("app.providers.http.asyncio.sleep", AsyncMock())
    provider_client = client(httpx.MockTransport(handler), max_retries=1)
    result = await provider_client.complete([], AnalysisResult)
    assert result.disposition.value == "release"
    assert attempts == 2
    assert provider_client.last_call_metadata.http_status == 200
    assert provider_client.last_call_metadata.schema_valid is True
    assert provider_client.last_call_metadata.total_tokens == 15
    assert provider_client.last_call_metadata.cached_prompt_tokens == 2


def test_illegal_json_wrapper_and_trailing_comma_are_repaired() -> None:
    content = "Model output follows:\n```json\n" + json.dumps(analysis_payload())[:-1] + ",}\n```"
    assert _extract_json(content)["risk_level"] == "low"


async def test_second_schema_validation_failure_becomes_safe_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"risk_level":"low"}'}}]},
        )

    monkeypatch.setattr("app.providers.http.asyncio.sleep", AsyncMock())
    provider_client = client(httpx.MockTransport(handler), max_retries=1)
    with pytest.raises(ProviderError) as error:
        await provider_client.complete([], AnalysisResult)
    assert attempts == 2
    assert error.value.code == "provider_unavailable"
    assert provider_client.last_call_metadata.http_status == 200
    assert provider_client.last_call_metadata.schema_valid is False
    assert provider_client.last_call_metadata.error_type == "schema_validation_error"
