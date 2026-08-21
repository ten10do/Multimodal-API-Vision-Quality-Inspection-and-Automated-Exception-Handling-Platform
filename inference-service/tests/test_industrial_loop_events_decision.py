"""Industrial closed-loop tests: event schema + decision engine (Phases 1-2)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_loop.config import DecisionPolicy, FROZEN_THRESHOLD, MODEL_VERSION
from industrial_loop.decision_service import D3InferenceResult, DecisionEngine
from industrial_loop.events import (
    Decision,
    InspectionEvent,
    OperatorStatus,
    PlcStatus,
    ReasonCode,
)

POLICY = DecisionPolicy()


def _ok_result(**overrides) -> D3InferenceResult:
    base = dict(
        ok=True,
        model_version=MODEL_VERSION,
        artifact_version="steel-patchcore-d3-release@1.3.0",
        image_score=FROZEN_THRESHOLD * 0.95,
        pixel_score=0.10,
        threshold=FROZEN_THRESHOLD,
        confidence={"kind": "absolute_threshold_margin_ratio", "value": 0.05, "calibrated_probability": False},
        heatmap_reference="sim://heatmap/P1.png",
        latency_ms=12.0,
    )
    base.update(overrides)
    return D3InferenceResult(**base)


def _decide(engine, result):
    return engine.decide(result, product_id="P1", batch_id="B1", camera_id="CAM-01")


# --- Phase 1: inspection_event schema ----------------------------------------

class TestInspectionEventSchema:
    def test_required_fields_present(self):
        event = InspectionEvent(
            product_id="P0001", batch_id="B-001", camera_id="CAM-01",
            model_version=MODEL_VERSION, artifact_version="rel@1.3.0",
            image_score=0.80, pixel_score=0.05, threshold=FROZEN_THRESHOLD,
            decision=Decision.PASS, reason_code=ReasonCode.NORMAL,
            heatmap_reference="sim://h.png",
        )
        payload = event.model_dump()
        for field in (
            "id", "timestamp", "product_id", "batch_id", "camera_id", "model_version",
            "artifact_version", "image_score", "pixel_score", "decision", "reason_code",
            "heatmap_reference", "operator_status", "plc_status", "mes_status",
        ):
            assert field in payload

    def test_pass_example_pairing(self):
        event = InspectionEvent(
            product_id="P", batch_id="B", camera_id="C", model_version="m", artifact_version="a",
            image_score=0.5, threshold=FROZEN_THRESHOLD, decision=Decision.PASS, reason_code=ReasonCode.NORMAL,
        )
        assert event.decision is Decision.PASS and event.reason_code is ReasonCode.NORMAL

    def test_reject_requires_score_at_or_above_threshold(self):
        with pytest.raises(ValueError):
            InspectionEvent(
                product_id="P", batch_id="B", camera_id="C", model_version="m", artifact_version="a",
                image_score=0.5, threshold=FROZEN_THRESHOLD,
                decision=Decision.REJECT, reason_code=ReasonCode.DEFECT_DETECTED,
            )

    def test_invalid_decision_reason_pairs_rejected(self):
        with pytest.raises(ValueError):
            InspectionEvent(
                product_id="P", batch_id="B", camera_id="C", model_version="m", artifact_version="a",
                image_score=0.9, threshold=FROZEN_THRESHOLD,
                decision=Decision.PASS, reason_code=ReasonCode.DEFECT_DETECTED,
            )
        with pytest.raises(ValueError):
            InspectionEvent(
                product_id="P", batch_id="B", camera_id="C", model_version="m", artifact_version="a",
                decision=Decision.HOLD, reason_code=ReasonCode.NORMAL, error_detail="x",
            )

    def test_ai_failure_hold_needs_error_detail(self):
        with pytest.raises(ValueError):
            InspectionEvent(
                product_id="P", batch_id="B", camera_id="C", model_version="m", artifact_version="a",
                decision=Decision.HOLD, reason_code=ReasonCode.AI_SYSTEM_FAILURE,
            )
        event = InspectionEvent(
            product_id="P", batch_id="B", camera_id="C", model_version="m", artifact_version="a",
            decision=Decision.HOLD, reason_code=ReasonCode.AI_SYSTEM_FAILURE, error_detail="gpu stream lost",
        )
        assert event.image_score is None  # scores may be absent on AI failure

    def test_non_finite_scores_rejected(self):
        with pytest.raises(ValueError):
            InspectionEvent(
                product_id="P", batch_id="B", camera_id="C", model_version="m", artifact_version="a",
                image_score=float("nan"), threshold=FROZEN_THRESHOLD,
                decision=Decision.PASS, reason_code=ReasonCode.NORMAL,
            )

    def test_traceability_ids_and_immutable_enrichment(self):
        event = InspectionEvent(
            product_id="P", batch_id="B", camera_id="C", model_version="m", artifact_version="a",
            image_score=0.9, threshold=FROZEN_THRESHOLD,
            decision=Decision.REJECT, reason_code=ReasonCode.DEFECT_DETECTED,
        )
        assert event.id and event.trace_id and event.timestamp.endswith("Z")
        enriched = event.with_updates(plc_status=PlcStatus.ACK_REJECT_SIGNAL)
        assert enriched.plc_status is PlcStatus.ACK_REJECT_SIGNAL
        assert event.plc_status is PlcStatus.NOT_APPLIED  # original untouched


# --- Phase 2: decision engine -------------------------------------------------

class TestDecisionEngine:
    def test_pass_below_threshold(self):
        engine = DecisionEngine(POLICY)
        event = _decide(engine, _ok_result())
        assert event.decision is Decision.PASS
        assert event.reason_code is ReasonCode.NORMAL
        assert engine.stats.snapshot()["pass"] == 1

    def test_reject_at_and_above_threshold(self):
        engine = DecisionEngine(POLICY)
        at = _decide(engine, _ok_result(image_score=FROZEN_THRESHOLD))
        above = _decide(engine, _ok_result(image_score=FROZEN_THRESHOLD * 1.03))
        assert at.decision is Decision.REJECT and above.decision is Decision.REJECT
        assert at.reason_code is ReasonCode.DEFECT_DETECTED

    @pytest.mark.parametrize(
        "result",
        [
            D3InferenceResult.failure("gpu stream error"),
            _ok_result(ok=False, error="timeout"),
            _ok_result(image_score=None),
            _ok_result(threshold=None),
            _ok_result(model_version=None),
            _ok_result(artifact_version=None),
            _ok_result(model_version="rogue-1.4.0"),
            _ok_result(threshold=FROZEN_THRESHOLD * 1.01),
            _ok_result(image_score=float("inf")),
        ],
    )
    def test_fail_close_any_anomaly_is_hold_never_pass(self, result):
        engine = DecisionEngine(POLICY)
        event = _decide(engine, result)
        assert event.decision is Decision.HOLD
        assert event.reason_code is ReasonCode.AI_SYSTEM_FAILURE
        assert event.error_detail

    def test_low_confidence_guard_band_routes_to_review_not_pass(self):
        policy = DecisionPolicy(hold_margin_ratio=0.02)
        engine = DecisionEngine(policy)
        borderline = FROZEN_THRESHOLD * (1 - 0.01)  # 1% below threshold, inside 2% band
        event = _decide(engine, _ok_result(image_score=borderline))
        assert event.decision is Decision.HOLD
        assert event.reason_code is ReasonCode.LOW_CONFIDENCE
        # clearly-normal product still passes
        clear = _decide(engine, _ok_result(image_score=FROZEN_THRESHOLD * 0.97))
        assert clear.decision is Decision.PASS

    def test_policy_refuses_bogus_threshold(self):
        with pytest.raises(ValueError):
            DecisionPolicy(reject_threshold=0.0)
        with pytest.raises(ValueError):
            DecisionPolicy(hold_margin_ratio=1.0)

    def test_engine_records_hold_reason_counts(self):
        engine = DecisionEngine(POLICY)
        _decide(engine, _ok_result())
        _decide(engine, D3InferenceResult.failure("x"))
        snapshot = engine.stats.snapshot()
        assert snapshot["total"] == 2
        assert snapshot["hold_reasons"] == {"AI_SYSTEM_FAILURE": 1}
