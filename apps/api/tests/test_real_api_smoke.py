import pytest

from app.config import get_settings
from app.real_api_smoke import main


async def test_real_api_smoke_refuses_mock_mode_without_network(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = get_settings()
    previous = settings.ai_mode
    settings.ai_mode = "mock"
    try:
        assert await main() == 2
        output = capsys.readouterr().out
        assert "AI_MODE=real" in output
        assert settings.bailian_api_key not in output or not settings.bailian_api_key
        assert settings.deepseek_api_key not in output or not settings.deepseek_api_key
    finally:
        settings.ai_mode = previous
