import asyncio
import base64
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.providers.base import ProviderError
from app.schemas import (
    AnalysisRequest,
    AnalysisResult,
    InspectionContext,
    VisionInspectionResult,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ProviderCallMetadata:
    http_status: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    cache_miss_prompt_tokens: int | None = None
    schema_valid: bool = False
    error_type: str | None = None


def _optional_token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _call_metadata(
    response: httpx.Response | None,
    body: object,
    *,
    schema_valid: bool,
    error_type: str | None,
) -> ProviderCallMetadata:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    return ProviderCallMetadata(
        http_status=response.status_code if response is not None else None,
        prompt_tokens=_optional_token_count(usage.get("prompt_tokens")),
        completion_tokens=_optional_token_count(usage.get("completion_tokens")),
        total_tokens=_optional_token_count(usage.get("total_tokens")),
        cached_prompt_tokens=_optional_token_count(
            usage.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens"))
        ),
        cache_miss_prompt_tokens=_optional_token_count(usage.get("prompt_cache_miss_tokens")),
        schema_valid=schema_valid,
        error_type=error_type,
    )


def _classify_provider_error(error: Exception) -> str:
    if isinstance(error, ProviderError):
        return error.code
    if isinstance(error, httpx.TimeoutException):
        return "timeout_error"
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 400:
            return "request_or_model_error"
        if status == 401:
            return "authentication_error"
        if status == 402:
            return "balance_error"
        if status == 403:
            return "permission_error"
        if status == 404:
            return "base_url_or_model_error"
        if status == 422:
            return "request_schema_error"
        if status == 429:
            return "rate_limit_or_balance_error"
        if status >= 500:
            return "provider_service_error"
        return "http_error"
    if isinstance(error, httpx.RequestError):
        return "connection_or_base_url_error"
    if isinstance(error, ValidationError):
        return "schema_validation_error"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_response_json"
    return "invalid_response_envelope"


def _extract_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```").strip()
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        stripped = stripped[first_brace : last_brace + 1]
    stripped = re.sub(r",\s*([}\]])", r"\1", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProviderError("invalid_provider_json", "模型返回了无法解析的结构化结果") from exc
    if not isinstance(value, dict):
        raise ProviderError("invalid_provider_schema", "模型返回结果不是 JSON 对象")
    return value


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        provider_name: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = httpx.Timeout(timeout_seconds)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.transport = transport
        self.last_call_metadata = ProviderCallMetadata()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        schema: type[SchemaT],
        normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> SchemaT:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        self.last_call_metadata = ProviderCallMetadata()
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self.headers,
            transport=self.transport,
        ) as client:
            for attempt in range(self.max_retries + 1):
                response: httpx.Response | None = None
                body: object = None
                try:
                    response = await client.post("chat/completions", json=payload)
                    response.raise_for_status()
                    body = response.json()
                    if not isinstance(body, dict):
                        raise TypeError("Provider response envelope must be an object")
                    content = body["choices"][0]["message"]["content"]
                    parsed = _extract_json(content)
                    result = schema.model_validate(normalize(parsed) if normalize else parsed)
                    self.last_call_metadata = _call_metadata(
                        response,
                        body,
                        schema_valid=True,
                        error_type=None,
                    )
                    return result
                except (
                    httpx.HTTPError,
                    KeyError,
                    TypeError,
                    ValueError,
                    ValidationError,
                    ProviderError,
                ) as exc:
                    last_error = exc
                    self.last_call_metadata = _call_metadata(
                        response,
                        body,
                        schema_valid=False,
                        error_type=_classify_provider_error(exc),
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2**attempt, 4))
        raise ProviderError(
            "provider_unavailable",
            f"{self.provider_name} 调用失败，已安全降级到人工复检",
        ) from last_error


class BailianVisionProvider:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    async def inspect(self, image: bytes, context: InspectionContext) -> VisionInspectionResult:
        schema_text = json.dumps(VisionInspectionResult.model_json_schema(), ensure_ascii=False)
        prompt = (
            "你是工业视觉质检模型。只输出满足 JSON Schema 的 JSON，不要输出 Markdown。"
            f"产品上下文：{context.model_dump_json()}。JSON Schema：{schema_text}"
        )
        image_url = (
            f"data:{context.image_mime_type};base64,{base64.b64encode(image).decode('ascii')}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        return await self.client.complete(
            messages,
            VisionInspectionResult,
            normalize=_normalize_vision_confidence,
        )


class DeepSeekReasoningProvider:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        schema_text = json.dumps(AnalysisResult.model_json_schema(), ensure_ascii=False)
        prompt = (
            "你是机器视觉异常处置决策助手。根据输入推断根因和动作。"
            "stop_line 必须 requires_human_approval=true。只输出 JSON。"
            f"输入：{request.model_dump_json()}。JSON Schema：{schema_text}"
        )
        return await self.client.complete(
            [{"role": "user", "content": prompt}],
            AnalysisResult,
        )


def _normalize_vision_confidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept provider confidence as either 0..1 or percentage, then validate strictly."""

    normalized = dict(payload)
    overall = normalized.get("overall_confidence")
    if isinstance(overall, (int, float)) and 1 < overall <= 100:
        normalized["overall_confidence"] = overall / 100
    defects = normalized.get("defects")
    if isinstance(defects, list):
        normalized_defects: list[Any] = []
        for defect in defects:
            if not isinstance(defect, dict):
                normalized_defects.append(defect)
                continue
            normalized_defect = dict(defect)
            confidence = normalized_defect.get("confidence")
            if isinstance(confidence, (int, float)) and 1 < confidence <= 100:
                normalized_defect["confidence"] = confidence / 100
            normalized_defects.append(normalized_defect)
        normalized["defects"] = normalized_defects
    return normalized
