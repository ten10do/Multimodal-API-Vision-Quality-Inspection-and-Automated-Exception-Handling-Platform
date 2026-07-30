import asyncio
import re
from pathlib import Path
from time import perf_counter

from PIL import Image

from app.config import get_settings
from app.enums import RiskLevel
from app.providers import get_reasoning_provider, get_vision_provider
from app.providers.base import ProviderError
from app.providers.http import OpenAICompatibleClient, ProviderCallMetadata
from app.schemas import (
    AnalysisRequest,
    AnalysisResult,
    Defect,
    InspectionContext,
    VisionInspectionResult,
)

SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.:/-]")
ERROR_SUMMARIES = {
    "none": "none",
    "configuration_error": "required_provider_configuration_is_missing",
    "timeout_error": "provider_request_timed_out",
    "authentication_error": "provider_rejected_credentials",
    "balance_error": "provider_balance_is_insufficient",
    "permission_error": "provider_denied_model_or_workspace_access",
    "base_url_or_model_error": "configured_endpoint_or_model_was_not_found",
    "request_or_model_error": "provider_rejected_request_or_model",
    "request_schema_error": "provider_rejected_request_schema",
    "rate_limit_or_balance_error": "provider_rate_limit_or_balance_rejected_request",
    "provider_service_error": "provider_service_failed",
    "connection_or_base_url_error": "provider_endpoint_could_not_be_reached",
    "invalid_provider_json": "provider_returned_unparseable_structured_output",
    "invalid_provider_schema": "provider_returned_invalid_structured_output",
    "schema_validation_error": "pydantic_schema_validation_failed",
    "invalid_response_json": "provider_http_body_was_not_json",
    "invalid_response_envelope": "provider_response_envelope_was_invalid",
    "provider_unavailable": "provider_call_failed_safely",
    "metadata_unavailable": "safe_call_metadata_was_unavailable",
}


def minimal_image() -> bytes:
    image_path = Path(__file__).resolve().parents[3] / "sample-data" / "mock-pass.png"
    with Image.open(image_path) as image:
        image.verify()
        if image.width <= 10 or image.height <= 10:
            raise ValueError("Smoke image does not meet provider dimension requirements")
    return image_path.read_bytes()


def safe_label(value: str, fallback: str) -> str:
    normalized = SAFE_LABEL.sub("_", value.strip())[:120]
    return normalized or fallback


def report_result(
    *,
    provider: str,
    model: str,
    started_at: float,
    status: str,
    error_type: str,
    metadata: ProviderCallMetadata,
) -> None:
    elapsed_ms = (perf_counter() - started_at) * 1000
    http_status = str(metadata.http_status) if metadata.http_status is not None else "not_called"
    schema_status = "passed" if metadata.schema_valid else "not_passed"
    prompt_tokens = (
        str(metadata.prompt_tokens) if metadata.prompt_tokens is not None else "not_provided"
    )
    completion_tokens = (
        str(metadata.completion_tokens)
        if metadata.completion_tokens is not None
        else "not_provided"
    )
    total_tokens = (
        str(metadata.total_tokens) if metadata.total_tokens is not None else "not_provided"
    )
    cached_tokens = (
        str(metadata.cached_prompt_tokens)
        if metadata.cached_prompt_tokens is not None
        else "not_provided"
    )
    cache_miss_tokens = (
        str(metadata.cache_miss_prompt_tokens)
        if metadata.cache_miss_prompt_tokens is not None
        else "not_provided"
    )
    safe_error_type = safe_label(error_type, "none")
    error_summary = ERROR_SUMMARIES.get(safe_error_type, "provider_call_failed_safely")
    print(
        f"provider={provider} "
        f"model={safe_label(model, 'not-configured')} "
        f"http_status={http_status} "
        f"elapsed_ms={elapsed_ms:.0f} "
        f"status={status} "
        f"schema={schema_status} "
        f"prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} "
        f"total_tokens={total_tokens} "
        f"cached_prompt_tokens={cached_tokens} "
        f"cache_miss_prompt_tokens={cache_miss_tokens} "
        "cost_estimate=calculated_from_official_pricing_after_call "
        f"error_type={safe_error_type} "
        f"error_summary={error_summary}"
    )


def call_metadata(provider: object) -> ProviderCallMetadata:
    client: object = getattr(provider, "client", None)
    if isinstance(client, OpenAICompatibleClient):
        return client.last_call_metadata
    return ProviderCallMetadata(error_type="metadata_unavailable")


async def smoke_bailian() -> bool:
    settings = get_settings()
    started_at = perf_counter()
    missing = settings.missing_bailian_config()
    if missing:
        report_result(
            provider="Bailian",
            model=settings.bailian_model,
            started_at=started_at,
            status="error",
            error_type="configuration_error",
            metadata=ProviderCallMetadata(error_type="configuration_error"),
        )
        return False
    context = InspectionContext(
        product_code="SMOKE-PRODUCT",
        batch_code="SMOKE-BATCH",
        image_mime_type="image/png",
        quality_rules=["Return a structured inspection result"],
    )
    provider = get_vision_provider(settings)
    try:
        result = await provider.inspect(minimal_image(), context)
        VisionInspectionResult.model_validate(result.model_dump(mode="json"))
        report_result(
            provider="Bailian",
            model=settings.bailian_model,
            started_at=started_at,
            status="success",
            error_type="none",
            metadata=call_metadata(provider),
        )
        return True
    except ProviderError as exc:
        report_result(
            provider="Bailian",
            model=settings.bailian_model,
            started_at=started_at,
            status="error",
            error_type=call_metadata(provider).error_type or exc.code,
            metadata=call_metadata(provider),
        )
        return False
    except Exception as exc:
        report_result(
            provider="Bailian",
            model=settings.bailian_model,
            started_at=started_at,
            status="error",
            error_type=call_metadata(provider).error_type or type(exc).__name__,
            metadata=call_metadata(provider),
        )
        return False


async def smoke_deepseek() -> bool:
    settings = get_settings()
    started_at = perf_counter()
    missing = settings.missing_deepseek_config()
    if missing:
        report_result(
            provider="DeepSeek",
            model=settings.deepseek_model,
            started_at=started_at,
            status="error",
            error_type="configuration_error",
            metadata=ProviderCallMetadata(error_type="configuration_error"),
        )
        return False
    context = InspectionContext(
        product_code="SMOKE-PRODUCT",
        batch_code="SMOKE-BATCH",
        image_mime_type="image/jpeg",
        quality_rules=["Critical defects require approval"],
    )
    vision = VisionInspectionResult(
        is_defective=True,
        overall_confidence=0.9,
        defects=[
            Defect(
                defect_type="smoke_test_defect",
                confidence=0.9,
                severity=RiskLevel.MEDIUM,
                description="Synthetic structured input for the provider smoke test",
            )
        ],
        summary="Synthetic provider smoke input",
    )
    provider = get_reasoning_provider(settings)
    try:
        result = await provider.analyze(AnalysisRequest(vision_result=vision, context=context))
        AnalysisResult.model_validate(result.model_dump(mode="json"))
        report_result(
            provider="DeepSeek",
            model=settings.deepseek_model,
            started_at=started_at,
            status="success",
            error_type="none",
            metadata=call_metadata(provider),
        )
        return True
    except ProviderError as exc:
        report_result(
            provider="DeepSeek",
            model=settings.deepseek_model,
            started_at=started_at,
            status="error",
            error_type=call_metadata(provider).error_type or exc.code,
            metadata=call_metadata(provider),
        )
        return False
    except Exception as exc:
        report_result(
            provider="DeepSeek",
            model=settings.deepseek_model,
            started_at=started_at,
            status="error",
            error_type=call_metadata(provider).error_type or type(exc).__name__,
            metadata=call_metadata(provider),
        )
        return False


async def main() -> int:
    settings = get_settings()
    if settings.ai_mode != "real":
        print("Refusing paid API smoke: set AI_MODE=real explicitly.")
        return 2
    settings.bailian_max_retries = 0
    settings.deepseek_max_retries = 0
    results = [await smoke_bailian(), await smoke_deepseek()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
