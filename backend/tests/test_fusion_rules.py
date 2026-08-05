"""Phase 6: Vision Fusion input to the Quality Rule Engine (6E/6F)."""

from __future__ import annotations

from types import SimpleNamespace

from app.enums import QualityResult, Severity
from app.quality.engine import DefectInput, QualityRuleEngine


def _rule(defect_type: str, action: str, confidence: float = 0.0, severity: str = "low", priority: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        defect_type=defect_type,
        min_confidence=confidence,
        max_area_ratio=1.0,
        action=QualityResult(action),
        severity=Severity(severity),
        priority=priority,
        rule_version=1,
        enabled=True,
    )


def _engine():
    return QualityRuleEngine(
        [
            _rule("crazing", "FAIL", confidence=0.5, severity="high"),
            _rule("*", "REVIEW", confidence=0.3, severity="medium", priority=100),
        ]
    )


def test_unknown_anomaly_always_review():
    """6F: UNKNOWN_ANOMALY -> REVIEW regardless of detections."""
    engine = _engine()
    d = engine.evaluate([], fusion_class="UNKNOWN_ANOMALY")
    assert d.quality_result == QualityResult.REVIEW
    assert "unknown anomaly" in d.reason

    # even with a low-confidence detection present, fusion governs
    d2 = engine.evaluate(
        [DefectInput(class_id=0, class_name="crazing", confidence=0.2, defect_area_ratio=0.01)],
        fusion_class="UNKNOWN_ANOMALY",
    )
    assert d2.quality_result == QualityResult.REVIEW


def test_normal_candidate_pass():
    engine = _engine()
    d = engine.evaluate([], fusion_class="NORMAL_CANDIDATE")
    assert d.quality_result == QualityResult.PASS


def test_known_defect_uses_rules():
    engine = _engine()
    d = engine.evaluate(
        [DefectInput(class_id=0, class_name="crazing", confidence=0.9, defect_area_ratio=0.1)],
        fusion_class="KNOWN_DEFECT",
    )
    assert d.quality_result == QualityResult.FAIL


def test_known_defect_with_anomaly_uses_rules():
    """The anomaly channel does not force FAIL; per-detection rules decide."""
    engine = _engine()
    d = engine.evaluate(
        [DefectInput(class_id=0, class_name="crazing", confidence=0.9, defect_area_ratio=0.1)],
        fusion_class="KNOWN_DEFECT_WITH_ANOMALY",
    )
    assert d.quality_result == QualityResult.FAIL

    # low-confidence unmatched detection + anomaly -> REVIEW (rule, not fusion)
    d2 = engine.evaluate(
        [DefectInput(class_id=0, class_name="crazing", confidence=0.2, defect_area_ratio=0.01)],
        fusion_class="KNOWN_DEFECT_WITH_ANOMALY",
    )
    assert d2.quality_result == QualityResult.REVIEW
