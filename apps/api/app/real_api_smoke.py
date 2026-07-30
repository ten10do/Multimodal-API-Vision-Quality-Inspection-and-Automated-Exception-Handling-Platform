import asyncio
import re
from io import BytesIO
from time import perf_counter

from PIL import Image

from app.config import get_settings
from app.enums import RiskLevel
from app.providers import get_reasoning_provider, get_vision_provider
from app.providers.base import ProviderError
from app.schemas import (
    AnalysisRequest,
    AnalysisResult,
    Defect,
    InspectionContext,
    VisionInspectionResult,
)

SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.:/-]")


def minimal_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


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
) -> None:
    elapsed_ms = (perf_counter() - started_at) * 1000
    print(
        f"provider={provider} "
        f"model={safe_label(model, 'not-configured')} "
        f"elapsed_ms={elapsed_ms:.0f} "
        f"status={status} "
        f"error_type={safe_label(error_type, 'none')}"
    )


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
        )
        return False
    context = InspectionContext(
        product_code="SMOKE-PRODUCT",
        batch_code="SMOKE-BATCH",
        image_mime_type="image/jpeg",
        quality_rules=["Return a structured inspection result"],
    )
    try:
        result = await get_vision_provider(settings).inspect(minimal_image(), context)
        VisionInspectionResult.model_validate(result.model_dump(mode="json"))
        report_result(
            provider="Bailian",
            model=settings.bailian_model,
            started_at=started_at,
            status="success",
            error_type="none",
        )
        return True
    except ProviderError as exc:
        report_result(
            provider="Bailian",
            model=settings.bailian_model,
            started_at=started_at,
            status="error",
            error_type=exc.code,
        )
        return False
    except Exception as exc:
        report_result(
            provider="Bailian",
            model=settings.bailian_model,
            started_at=started_at,
            status="error",
            error_type=type(exc).__name__,
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
    try:
        result = await get_reasoning_provider(settings).analyze(
            AnalysisRequest(vision_result=vision, context=context)
        )
        AnalysisResult.model_validate(result.model_dump(mode="json"))
        report_result(
            provider="DeepSeek",
            model=settings.deepseek_model,
            started_at=started_at,
            status="success",
            error_type="none",
        )
        return True
    except ProviderError as exc:
        report_result(
            provider="DeepSeek",
            model=settings.deepseek_model,
            started_at=started_at,
            status="error",
            error_type=exc.code,
        )
        return False
    except Exception as exc:
        report_result(
            provider="DeepSeek",
            model=settings.deepseek_model,
            started_at=started_at,
            status="error",
            error_type=type(exc).__name__,
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
