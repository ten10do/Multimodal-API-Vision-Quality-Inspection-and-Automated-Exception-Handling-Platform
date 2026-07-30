import asyncio
from io import BytesIO

from PIL import Image

from app.config import get_settings
from app.enums import RiskLevel
from app.providers import get_reasoning_provider, get_vision_provider
from app.providers.base import ProviderError
from app.schemas import (
    AnalysisRequest,
    Defect,
    InspectionContext,
    VisionInspectionResult,
)


def minimal_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


async def smoke_bailian() -> bool:
    settings = get_settings()
    missing = settings.missing_bailian_config()
    if missing:
        print(f"Bailian: configuration_error ({', '.join(missing)})")
        return False
    context = InspectionContext(
        product_code="SMOKE-PRODUCT",
        batch_code="SMOKE-BATCH",
        image_mime_type="image/jpeg",
        quality_rules=["Return a structured inspection result"],
    )
    try:
        result = await get_vision_provider(settings).inspect(minimal_image(), context)
        print(
            "Bailian: success "
            f"(defective={result.is_defective}, confidence={result.overall_confidence:.2f})"
        )
        return True
    except ProviderError as exc:
        print(f"Bailian: provider_error ({exc.code})")
        return False
    except Exception as exc:
        print(f"Bailian: {type(exc).__name__}")
        return False


async def smoke_deepseek() -> bool:
    settings = get_settings()
    missing = settings.missing_deepseek_config()
    if missing:
        print(f"DeepSeek: configuration_error ({', '.join(missing)})")
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
        print(
            "DeepSeek: success "
            f"(risk={result.risk_level.value}, disposition={result.disposition.value})"
        )
        return True
    except ProviderError as exc:
        print(f"DeepSeek: provider_error ({exc.code})")
        return False
    except Exception as exc:
        print(f"DeepSeek: {type(exc).__name__}")
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
