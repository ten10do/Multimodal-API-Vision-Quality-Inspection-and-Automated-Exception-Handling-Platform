from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..enums import QualityResult, Severity

ACTION_RANK = {QualityResult.PASS: 0, QualityResult.REVIEW: 1, QualityResult.FAIL: 2}


class Rule(Protocol):
    """Minimal structural view of a quality rule (DB row or dict)."""

    defect_type: str
    min_confidence: float
    max_area_ratio: float
    action: QualityResult
    severity: Severity
    priority: int
    rule_version: int
    enabled: bool


@dataclass
class DefectInput:
    """Objective facts a rule engine may consume from the Vision Contract."""

    class_id: int
    class_name: str
    confidence: float
    defect_area_ratio: float


@dataclass
class Decision:
    """Engine output. severity/PASS/REVIEW/FAIL are produced here, never in the vision layer."""

    quality_result: QualityResult
    severity: Severity
    matched_rule: str | None
    rule_version: int
    reason: str


@dataclass
class QualityRuleEngine:
    rules: list[Rule] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._active_version = max((r.rule_version for r in self.rules), default=1)
        self._enabled: list[Rule] = [
            r for r in self.rules if r.enabled and r.rule_version == self._active_version
        ]
        self._enabled.sort(key=lambda r: r.priority)

    def evaluate(self, detections: list[DefectInput]) -> Decision:
        """Aggregate per-detection rule matches into a single decision.

        Priority semantics:
        - For each detection, the rule with the smallest ``priority`` value
          that matches wins (first-match-wins per detection).
        - A rule matches when defect_type equals the class_name (or '*'),
          confidence >= min_confidence and area_ratio <= max_area_ratio.
        - The final decision takes the worst action across matched defects
          (FAIL > REVIEW > PASS) and the highest severity among the defects
          that produced that action.
        - No detections always yields PASS (business behaviour only, see
          docs/01-negative-sample-strategy.md).
        - A detection with no matching rule yields REVIEW with a clear reason.
        """
        if not detections:
            return Decision(QualityResult.PASS, Severity.LOW, None, self._active_version, "no detections")

        per_defect: list[tuple[QualityResult, Severity, str, str]] = []
        for det in detections:
            matched = self._match(det)
            if matched is None:
                per_defect.append(
                    (QualityResult.REVIEW, Severity.MEDIUM, None, f"no rule matched for '{det.class_name}'")
                )
                continue
            rule, reason = matched
            per_defect.append((rule.action, rule.severity, rule.defect_type, reason))

        worst = max(per_defect, key=lambda t: ACTION_RANK[t[0]])
        final_action, final_severity, matched_rule, reason = worst

        worst_severity = final_severity
        if final_action == QualityResult.FAIL:
            fail_defects = [t for t in per_defect if t[0] == QualityResult.FAIL]
            worst_severity = max((t[1] for t in fail_defects), key=lambda s: _SEVERITY_RANK[s])
        elif final_action == QualityResult.REVIEW:
            review_defects = [t for t in per_defect if t[0] == QualityResult.REVIEW]
            worst_severity = max((t[1] for t in review_defects), key=lambda s: _SEVERITY_RANK[s])

        reasons = "; ".join(t[3] for t in per_defect)
        return Decision(final_action, worst_severity, matched_rule, self._active_version, reasons)

    def _match(self, det: DefectInput) -> tuple[Rule, str] | None:
        for rule in self._enabled:
            if rule.defect_type != "*" and rule.defect_type != det.class_name:
                continue
            if det.confidence < rule.min_confidence:
                continue
            if det.defect_area_ratio > rule.max_area_ratio:
                continue
            return rule, (
                f"rule[{rule.defect_type}] conf>={rule.min_confidence} area<={rule.max_area_ratio} -> {rule.action.value}"
            )
        return None


_SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}
