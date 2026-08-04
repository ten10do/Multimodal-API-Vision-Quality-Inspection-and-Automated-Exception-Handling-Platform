from __future__ import annotations

import pytest

from app.enums import QualityResult, Severity
from app.quality.engine import DefectInput, QualityRuleEngine, Rule


def make_rule(
    defect_type: str,
    action: QualityResult,
    severity: Severity,
    priority: int = 100,
    min_confidence: float = 0.0,
    max_area_ratio: float = 1.0,
    enabled: bool = True,
    rule_version: int = 1,
) -> Rule:
    return type(
        "Rule",
        (),
        {
            "defect_type": defect_type,
            "min_confidence": min_confidence,
            "max_area_ratio": max_area_ratio,
            "action": action,
            "severity": severity,
            "priority": priority,
            "rule_version": rule_version,
            "enabled": enabled,
        },
    )()


def det(class_name: str, confidence: float, area_ratio: float) -> DefectInput:
    return DefectInput(class_id=0, class_name=class_name, confidence=confidence, defect_area_ratio=area_ratio)


def test_no_detections_is_pass():
    engine = QualityRuleEngine([])
    decision = engine.evaluate([])
    assert decision.quality_result == QualityResult.PASS
    assert decision.severity == Severity.LOW
    assert "no detections" in decision.reason


def test_critical_defect_is_fail():
    rules = [make_rule("crazing", QualityResult.FAIL, Severity.HIGH, priority=10)]
    decision = QualityRuleEngine(rules).evaluate([det("crazing", 0.9, 0.2)])
    assert decision.quality_result == QualityResult.FAIL
    assert decision.severity == Severity.HIGH


def test_low_confidence_no_rule_match_is_review():
    rules = [make_rule("scratches", QualityResult.PASS, Severity.LOW, min_confidence=0.8)]
    decision = QualityRuleEngine(rules).evaluate([det("scratches", 0.4, 0.05)])
    assert decision.quality_result == QualityResult.REVIEW
    assert "no rule matched" in decision.reason


def test_area_threshold_selects_review_over_pass():
    rules = [
        make_rule("scratches", QualityResult.PASS, Severity.LOW, priority=10, max_area_ratio=0.3),
        make_rule("scratches", QualityResult.REVIEW, Severity.MEDIUM, priority=20, max_area_ratio=1.0),
    ]
    small = QualityRuleEngine(rules).evaluate([det("scratches", 0.9, 0.1)])
    assert small.quality_result == QualityResult.PASS
    large = QualityRuleEngine(rules).evaluate([det("scratches", 0.9, 0.6)])
    assert large.quality_result == QualityResult.REVIEW
    assert large.severity == Severity.MEDIUM


def test_multiple_defects_worst_action_wins():
    rules = [
        make_rule("scratches", QualityResult.PASS, Severity.LOW, priority=10),
        make_rule("crazing", QualityResult.FAIL, Severity.CRITICAL, priority=10),
    ]
    decision = QualityRuleEngine(rules).evaluate(
        [det("scratches", 0.95, 0.05), det("crazing", 0.9, 0.1)]
    )
    assert decision.quality_result == QualityResult.FAIL
    assert decision.severity == Severity.CRITICAL


def test_rule_priority_lower_number_wins():
    rules = [
        make_rule("patches", QualityResult.REVIEW, Severity.MEDIUM, priority=5),
        make_rule("patches", QualityResult.FAIL, Severity.HIGH, priority=10),
    ]
    decision = QualityRuleEngine(rules).evaluate([det("patches", 0.85, 0.4)])
    assert decision.quality_result == QualityResult.REVIEW
    assert decision.severity == Severity.MEDIUM


def test_disabled_rule_ignored():
    rules = [
        make_rule("patches", QualityResult.FAIL, Severity.HIGH, priority=1, enabled=False),
        make_rule("patches", QualityResult.PASS, Severity.LOW, priority=2, enabled=True),
    ]
    decision = QualityRuleEngine(rules).evaluate([det("patches", 0.9, 0.1)])
    assert decision.quality_result == QualityResult.PASS


def test_rule_version_only_active_version_considered():
    rules = [
        make_rule("patches", QualityResult.FAIL, Severity.HIGH, rule_version=1),
        make_rule("patches", QualityResult.PASS, Severity.LOW, rule_version=2),
    ]
    decision = QualityRuleEngine(rules).evaluate([det("patches", 0.9, 0.1)])
    assert decision.quality_result == QualityResult.PASS
    assert decision.rule_version == 2


def test_wildcard_rule_matches_any_class():
    rules = [make_rule("*", QualityResult.REVIEW, Severity.MEDIUM, priority=1)]
    decision = QualityRuleEngine(rules).evaluate([det("unknown_class", 0.7, 0.3)])
    assert decision.quality_result == QualityResult.REVIEW


def test_output_contains_rule_version_and_reason():
    rules = [make_rule("crazing", QualityResult.FAIL, Severity.HIGH, priority=1, rule_version=3)]
    decision = QualityRuleEngine(rules).evaluate([det("crazing", 0.9, 0.2)])
    assert decision.rule_version == 3
    assert "rule[crazing]" in decision.reason
