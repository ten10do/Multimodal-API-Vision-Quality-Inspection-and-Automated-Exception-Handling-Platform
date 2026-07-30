from io import BytesIO

import pytest
from PIL import Image

from app import real_api_smoke
from app.config import get_settings
from app.enums import Disposition, RiskLevel
from app.providers.http import OpenAICompatibleClient, ProviderCallMetadata
from app.schemas import AnalysisResult, VisionInspectionResult


def test_real_api_smoke_uses_repository_legal_image() -> None:
    content = real_api_smoke.minimal_image()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(content)) as image:
        assert image.width > 10
        assert image.height > 10


async def test_real_api_smoke_refuses_mock_mode_without_network(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = get_settings()
    previous = settings.ai_mode
    settings.ai_mode = "mock"
    try:
        assert await real_api_smoke.main() == 2
        output = capsys.readouterr().out
        assert "AI_MODE=real" in output
        assert settings.bailian_api_key not in output or not settings.bailian_api_key
        assert settings.deepseek_api_key not in output or not settings.deepseek_api_key
    finally:
        settings.ai_mode = previous


async def test_real_api_smoke_calls_each_provider_once_with_safe_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    previous = {
        "ai_mode": settings.ai_mode,
        "bailian_api_key": settings.bailian_api_key,
        "bailian_base_url": settings.bailian_base_url,
        "bailian_model": settings.bailian_model,
        "bailian_max_retries": settings.bailian_max_retries,
        "deepseek_api_key": settings.deepseek_api_key,
        "deepseek_base_url": settings.deepseek_base_url,
        "deepseek_model": settings.deepseek_model,
        "deepseek_max_retries": settings.deepseek_max_retries,
    }

    class VisionStub:
        calls = 0

        def __init__(self) -> None:
            self.client = OpenAICompatibleClient(
                provider_name="bailian",
                api_key=bailian_placeholder,
                base_url="https://bailian.invalid/v1",
                model="vision-smoke-model",
                timeout_seconds=1,
                max_retries=0,
            )

        async def inspect(self, image: bytes, context: object) -> VisionInspectionResult:
            self.calls += 1
            self.client.last_call_metadata = ProviderCallMetadata(
                http_status=200,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                schema_valid=True,
            )
            return VisionInspectionResult(
                is_defective=False,
                overall_confidence=0.99,
                defects=[],
                summary="Smoke response",
            )

    class ReasoningStub:
        calls = 0

        def __init__(self) -> None:
            self.client = OpenAICompatibleClient(
                provider_name="deepseek",
                api_key=deepseek_placeholder,
                base_url="https://deepseek.invalid/v1",
                model="reasoning-smoke-model",
                timeout_seconds=1,
                max_retries=0,
            )

        async def analyze(self, request: object) -> AnalysisResult:
            self.calls += 1
            self.client.last_call_metadata = ProviderCallMetadata(
                http_status=200,
                prompt_tokens=8,
                completion_tokens=4,
                total_tokens=12,
                schema_valid=True,
            )
            return AnalysisResult(
                risk_level=RiskLevel.LOW,
                probable_causes=["No defect"],
                recommended_actions=["Release"],
                disposition=Disposition.RELEASE,
                requires_human_approval=False,
                rationale="Validated smoke response",
            )

    bailian_placeholder = "bailian-test-placeholder"
    deepseek_placeholder = "deepseek-test-placeholder"
    vision = VisionStub()
    reasoning = ReasoningStub()
    try:
        settings.ai_mode = "real"
        settings.bailian_api_key = bailian_placeholder
        settings.bailian_base_url = "https://bailian.invalid/v1"
        settings.bailian_model = "vision-smoke-model"
        settings.bailian_max_retries = 9
        settings.deepseek_api_key = deepseek_placeholder
        settings.deepseek_base_url = "https://deepseek.invalid/v1"
        settings.deepseek_model = "reasoning-smoke-model"
        settings.deepseek_max_retries = 9
        monkeypatch.setattr(real_api_smoke, "get_vision_provider", lambda _: vision)
        monkeypatch.setattr(real_api_smoke, "get_reasoning_provider", lambda _: reasoning)

        assert await real_api_smoke.main() == 0
        assert vision.calls == 1
        assert reasoning.calls == 1
        assert settings.bailian_max_retries == 0
        assert settings.deepseek_max_retries == 0

        output = capsys.readouterr().out
        assert "provider=Bailian model=vision-smoke-model" in output
        assert "provider=DeepSeek model=reasoning-smoke-model" in output
        assert output.count("status=success") == 2
        assert output.count("error_type=none") == 2
        assert output.count("http_status=200") == 2
        assert output.count("schema=passed") == 2
        assert "elapsed_ms=" in output
        assert bailian_placeholder not in output
        assert deepseek_placeholder not in output
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)
