"""Promotion gate (8F): CANDIDATE -> PRODUCTION must pass measurable gates.

Pure functions, no DB coupling. A candidate can only be promoted when all
configured thresholds pass AND the domain is validated. In particular, a
cross-domain MVTec PatchCore (steel_domain_validated=false) can never be
promoted to a steel production model even if its AUROC is perfect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    passed: bool = False
    checks: list[dict] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks, "blocked": self.blocked}


def _check(name: str, ok: bool, got: Any, required: Any, domain_blocked: bool = False) -> dict:
    return {
        "check": name,
        "passed": bool(ok),
        "got": got,
        "required": required,
        "blocked_by_domain": domain_blocked,
    }


def evaluate_yolo(
    *,
    metrics: dict,
    thresholds: dict,
    domain_validated: bool,
    required_domain: str,
) -> GateResult:
    """YOLO promotion gate.

    thresholds: {mAP50: float, recall: float, latency_p95_ms: float}
    metrics:    {mAP50: float, recall: float, latency_p95_ms: float}
    """
    result = GateResult()
    m = thresholds
    checks = [
        _check("mAP50", metrics.get("mAP50", -1.0) >= m["mAP50"], metrics.get("mAP50"), m["mAP50"]),
        _check("recall", metrics.get("recall", -1.0) >= m["recall"], metrics.get("recall"), m["recall"]),
        _check(
            "latency_p95",
            metrics.get("latency_p95_ms", float("inf")) <= m["latency_p95_ms"],
            metrics.get("latency_p95_ms"),
            m["latency_p95_ms"],
        ),
    ]
    result.checks = checks
    if not domain_validated:
        result.blocked.append(f"domain not validated for {required_domain}")
        result.checks.append(_check("domain", False, domain_validated, True, domain_blocked=True))
    if not all(c["passed"] for c in checks):
        result.blocked.extend(c["check"] for c in checks if not c["passed"])
    result.passed = all(c["passed"] for c in checks) and domain_validated
    return result


def evaluate_patchcore(
    *,
    metrics: dict,
    thresholds: dict,
    domain_validated: bool,
    required_domain: str,
) -> GateResult:
    """PatchCore promotion gate.

    thresholds: {image_auroc: float, pixel_auroc: float, latency_ms: float}
    metrics:    {image_auroc: float, pixel_auroc: float, latency_ms: float}
    """
    result = GateResult()
    m = thresholds
    checks = [
        _check("image_auroc", metrics.get("image_auroc", -1.0) >= m["image_auroc"], metrics.get("image_auroc"), m["image_auroc"]),
        _check("pixel_auroc", metrics.get("pixel_auroc", -1.0) >= m["pixel_auroc"], metrics.get("pixel_auroc"), m["pixel_auroc"]),
        _check(
            "latency",
            metrics.get("latency_ms", float("inf")) <= m["latency_ms"],
            metrics.get("latency_ms"),
            m["latency_ms"],
        ),
    ]
    result.checks = checks
    if not domain_validated:
        result.blocked.append(f"domain not validated for {required_domain}")
        result.checks.append(_check("domain", False, domain_validated, True, domain_blocked=True))
    if not all(c["passed"] for c in checks):
        result.blocked.extend(c["check"] for c in checks if not c["passed"])
    result.passed = all(c["passed"] for c in checks) and domain_validated
    return result


DEFAULT_YOLO_THRESHOLDS = {
    "mAP50": 0.6,
    "recall": 0.6,
    "latency_p95_ms": 120.0,  # GPU p95; the CPU fallback benchmark is higher
}

DEFAULT_PATCHCORE_THRESHOLDS = {
    "image_auroc": 0.9,
    "pixel_auroc": 0.9,
    "latency_ms": 2000.0,
}


def evaluate(model_type: str, *, metrics: dict, thresholds: dict | None = None,
             domain_validated: bool, required_domain: str) -> GateResult:
    if model_type == "yolo":
        return evaluate_yolo(
            metrics=metrics, thresholds=thresholds or DEFAULT_YOLO_THRESHOLDS,
            domain_validated=domain_validated, required_domain=required_domain,
        )
    if model_type == "patchcore":
        return evaluate_patchcore(
            metrics=metrics, thresholds=thresholds or DEFAULT_PATCHCORE_THRESHOLDS,
            domain_validated=domain_validated, required_domain=required_domain,
        )
    return GateResult(passed=False, blocked=[f"unknown model_type {model_type}"])
