from app.config import Settings, get_settings
from app.providers.base import ReasoningProvider, VisionProvider
from app.providers.http import (
    BailianVisionProvider,
    DeepSeekReasoningProvider,
    OpenAICompatibleClient,
)
from app.providers.mock import MockReasoningProvider, MockVisionProvider


def get_vision_provider(settings: Settings | None = None) -> VisionProvider:
    settings = settings or get_settings()
    if settings.ai_mode == "mock":
        return MockVisionProvider()
    settings.validate_bailian_config()
    return BailianVisionProvider(
        OpenAICompatibleClient(
            provider_name="bailian",
            api_key=settings.bailian_api_key,
            base_url=settings.bailian_base_url,
            model=settings.bailian_model,
            timeout_seconds=settings.bailian_timeout_seconds,
            max_retries=settings.bailian_max_retries,
        )
    )


def get_reasoning_provider(settings: Settings | None = None) -> ReasoningProvider:
    settings = settings or get_settings()
    if settings.ai_mode == "mock":
        return MockReasoningProvider()
    settings.validate_deepseek_config()
    return DeepSeekReasoningProvider(
        OpenAICompatibleClient(
            provider_name="deepseek",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            timeout_seconds=settings.deepseek_timeout_seconds,
            max_retries=settings.deepseek_max_retries,
        )
    )
