import pytest
from pydantic import ValidationError

from app.providers.http import _normalize_vision_confidence
from app.schemas import AnalysisResult, VisionInspectionResult
from app.workflow import enforce_quality_rules


def test_stop_line_schema_rejects_missing_approval() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {
                "risk_level": "critical",
                "probable_causes": ["设备失控"],
                "recommended_actions": ["停线"],
                "disposition": "stop_line",
                "requires_human_approval": False,
                "rationale": "存在严重结构损伤",
            }
        )


@pytest.mark.parametrize(
    ("is_defective", "defects"),
    [
        (True, []),
        (
            False,
            [
                {
                    "defect_type": "scratch",
                    "confidence": 0.8,
                    "severity": "medium",
                    "description": "visible scratch",
                }
            ],
        ),
    ],
)
def test_vision_schema_requires_defect_flag_consistency(
    is_defective: bool, defects: list[dict[str, object]]
) -> None:
    with pytest.raises(ValidationError):
        VisionInspectionResult.model_validate(
            {
                "is_defective": is_defective,
                "overall_confidence": 0.9,
                "defects": defects,
                "summary": "inconsistent payload",
            }
        )


def test_vision_schema_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        VisionInspectionResult.model_validate(
            {
                "is_defective": False,
                "overall_confidence": 1.01,
                "defects": [],
                "summary": "invalid confidence",
            }
        )


def test_analysis_schema_rejects_unknown_risk() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {
                "risk_level": "catastrophic",
                "probable_causes": ["unknown"],
                "recommended_actions": ["review"],
                "disposition": "manual_review",
                "requires_human_approval": False,
                "rationale": "invalid enum",
            }
        )


def test_schema_forbids_unknown_model_fields() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {
                "risk_level": "low",
                "probable_causes": ["无异常"],
                "recommended_actions": ["放行"],
                "disposition": "release",
                "requires_human_approval": False,
                "rationale": "符合规则",
                "untrusted_extra": "must not pass",
            }
        )


def test_provider_percentage_confidence_is_normalized() -> None:
    normalized = _normalize_vision_confidence(
        {
            "is_defective": True,
            "overall_confidence": 92,
            "defects": [
                {
                    "defect_type": "scratch",
                    "confidence": 88,
                    "severity": "medium",
                    "description": "surface scratch",
                }
            ],
            "summary": "one defect",
        }
    )
    result = VisionInspectionResult.model_validate(normalized)
    assert result.overall_confidence == 0.92
    assert result.defects[0].confidence == 0.88


def test_quality_rules_prevent_ai_from_downgrading_critical_defect() -> None:
    vision = VisionInspectionResult.model_validate(
        {
            "is_defective": True,
            "overall_confidence": 0.95,
            "defects": [
                {
                    "defect_type": "crack",
                    "confidence": 0.91,
                    "severity": "critical",
                    "description": "structural crack",
                }
            ],
            "summary": "critical defect",
        }
    )
    unsafe_analysis = AnalysisResult.model_validate(
        {
            "risk_level": "low",
            "probable_causes": ["unknown"],
            "recommended_actions": ["release"],
            "disposition": "release",
            "requires_human_approval": False,
            "rationale": "incorrect model downgrade",
        }
    )
    enforced = enforce_quality_rules(vision, unsafe_analysis)
    assert enforced.risk_level.value == "critical"
    assert enforced.disposition.value == "stop_line"
    assert enforced.requires_human_approval is True
