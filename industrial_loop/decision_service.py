"""Phase 2 — Decision Engine: D3 inference result -> production decision.

Fail-close contract:
  * any inference/system anomaly (error, non-finite score, missing lineage,
    threshold mismatch) => HOLD + AI_SYSTEM_FAILURE, never PASS;
  * image_score >= frozen threshold => REJECT + DEFECT_DETECTED;
  * otherwise PASS + NORMAL (optionally HOLD + LOW_CONFIDENCE inside a
    conservative guard band, disabled by default).

The engine never tunes the threshold: it refuses to judge with a threshold
that differs from the frozen release value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import DecisionPolicy
from .events import (
    Decision,
    InspectionEvent,
    ReasonCode,
    new_trace_id,
    utc_now_iso,
)


@dataclass(frozen=True)
class D3InferenceResult:
    """Adapter view over whatever the (unchanged) D3 inference service returned."""

    ok: bool
    model_version: str | None = None
    artifact_version: str | None = None
    image_score: float | None = None
    pixel_score: float | None = None
    threshold: float | None = None
    confidence: dict | None = None
    heatmap_reference: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    # Additive fail-safe kind (edge/drift layer): "ai_system_failure" (default)
    # or "data_distribution_shift". Only affects the HOLD reason code.
    kind: str = "ai_system_failure"

    @staticmethod
    def failure(
        error: str, latency_ms: float | None = None, *, kind: str = "ai_system_failure"
    ) -> "D3InferenceResult":
        return D3InferenceResult(ok=False, error=error, latency_ms=latency_ms, kind=kind)


@dataclass
class DecisionStats:
    total: int = 0
    pass_count: int = 0
    reject_count: int = 0
    hold_count: int = 0
    hold_reasons: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "total": self.total,
            "pass": self.pass_count,
            "reject": self.reject_count,
            "hold": self.hold_count,
            "hold_reasons": dict(self.hold_reasons),
        }


class DecisionEngine:
    def __init__(self, policy: DecisionPolicy | None = None) -> None:
        self.policy = policy or DecisionPolicy()
        self.stats = DecisionStats()

    # -- fail-close evaluation ------------------------------------------------

    def _hold(self, reason: ReasonCode, detail: str) -> tuple[Decision, ReasonCode, str]:
        return Decision.HOLD, reason, detail

    def evaluate(self, result: D3InferenceResult) -> tuple[Decision, ReasonCode, str | None]:
        """Pure decision rule. Returns (decision, reason_code, error_detail)."""
        hold_reason = (
            ReasonCode.DATA_DISTRIBUTION_SHIFT
            if result.kind == "data_distribution_shift"
            else ReasonCode.AI_SYSTEM_FAILURE
        )
        if not result.ok or result.error:
            return self._hold(hold_reason, result.error or "inference_failed")
        if result.image_score is None or result.threshold is None:
            return self._hold(hold_reason, "missing_image_score_or_threshold")
        if not (math.isfinite(result.image_score) and math.isfinite(result.threshold)):
            return self._hold(hold_reason, "non_finite_image_score_or_threshold")
        if result.model_version is None or result.artifact_version is None:
            return self._hold(hold_reason, "missing_model_or_artifact_version")
        if result.model_version != self.policy.expected_model_version:
            return self._hold(
                hold_reason,
                f"model_version_mismatch:{result.model_version}",
            )
        if result.threshold != self.policy.reject_threshold:
            return self._hold(
                hold_reason,
                f"threshold_lineage_mismatch:{result.threshold}",
            )
        if result.image_score >= result.threshold:
            return Decision.REJECT, ReasonCode.DEFECT_DETECTED, None
        if self.policy.hold_margin_ratio > 0.0:
            margin_ratio = (result.threshold - result.image_score) / result.threshold
            if margin_ratio < self.policy.hold_margin_ratio:
                return self._hold(
                    ReasonCode.LOW_CONFIDENCE,
                    f"margin_ratio {margin_ratio:.6f} < guard {self.policy.hold_margin_ratio}",
                )
        return Decision.PASS, ReasonCode.NORMAL, None

    # -- event construction ---------------------------------------------------

    def decide(
        self,
        result: D3InferenceResult,
        *,
        product_id: str,
        batch_id: str,
        camera_id: str,
        trace_id: str | None = None,
    ) -> InspectionEvent:
        decision, reason, detail = self.evaluate(result)

        def _clean(value: float | None) -> float | None:
            return value if value is not None and math.isfinite(value) else None

        event = InspectionEvent(
            trace_id=trace_id or new_trace_id(),
            timestamp=utc_now_iso(),
            product_id=product_id,
            batch_id=batch_id,
            camera_id=camera_id,
            model_version=result.model_version or self.policy.expected_model_version,
            artifact_version=result.artifact_version or "unknown",
            image_score=_clean(result.image_score),
            pixel_score=_clean(result.pixel_score),
            threshold=_clean(result.threshold),
            confidence=result.confidence,
            decision=decision,
            reason_code=reason,
            heatmap_reference=result.heatmap_reference,
            error_detail=detail,
            latency_ms=result.latency_ms,
        )
        self.stats.total += 1
        if decision is Decision.PASS:
            self.stats.pass_count += 1
        elif decision is Decision.REJECT:
            self.stats.reject_count += 1
        else:
            self.stats.hold_count += 1
            self.stats.hold_reasons[reason.value] = self.stats.hold_reasons.get(reason.value, 0) + 1
        return event
