import pytest

from app.enums import Disposition, RiskLevel
from app.providers.mock import MockReasoningProvider, MockVisionProvider
from app.schemas import AnalysisRequest
from tests.factories import inspection_context, png_for_bucket, vision_result


@pytest.mark.parametrize(
    ("bucket", "expected_risk", "defective"),
    [
        (0, None, False),
        (1, RiskLevel.MEDIUM, True),
        (2, RiskLevel.HIGH, True),
        (3, RiskLevel.CRITICAL, True),
    ],
)
async def test_mock_vision_provider_four_deterministic_scenarios(
    bucket: int, expected_risk: RiskLevel | None, defective: bool
) -> None:
    result = await MockVisionProvider().inspect(png_for_bucket(bucket), inspection_context())
    assert result.is_defective is defective
    if expected_risk is None:
        assert result.defects == []
    else:
        assert result.defects[0].severity == expected_risk


@pytest.mark.parametrize(
    ("risk", "disposition", "approval"),
    [
        (None, Disposition.RELEASE, False),
        (RiskLevel.MEDIUM, Disposition.MANUAL_REVIEW, False),
        (RiskLevel.HIGH, Disposition.REJECT, False),
        (RiskLevel.CRITICAL, Disposition.STOP_LINE, True),
    ],
)
async def test_mock_reasoning_provider_four_dispositions(
    risk: RiskLevel | None,
    disposition: Disposition,
    approval: bool,
) -> None:
    result = await MockReasoningProvider().analyze(
        AnalysisRequest(
            vision_result=vision_result(risk),
            context=inspection_context(),
        )
    )
    assert result.disposition == disposition
    assert result.requires_human_approval is approval
