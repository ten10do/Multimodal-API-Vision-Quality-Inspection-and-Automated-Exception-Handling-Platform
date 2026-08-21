"""Industrial closed-loop tests: human review workflow (Phase 5)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_loop.config import FROZEN_THRESHOLD, MODEL_VERSION
from industrial_loop.decision_service import D3InferenceResult, DecisionEngine
from industrial_loop.events import Decision, OperatorStatus, ReasonCode
from industrial_loop.human_review import HumanReviewWorkflow, ReviewOutcome
from industrial_loop.mes_service import MesService, WorkOrderStatus


def _event(decision: Decision):
    engine = DecisionEngine()
    if decision is Decision.REJECT:
        result = D3InferenceResult(
            ok=True, model_version=MODEL_VERSION, artifact_version="a",
            image_score=FROZEN_THRESHOLD * 1.04, pixel_score=0.7, threshold=FROZEN_THRESHOLD,
        )
    elif decision is Decision.HOLD:
        result = D3InferenceResult.failure("camera stream glitch")
    else:
        result = D3InferenceResult(
            ok=True, model_version=MODEL_VERSION, artifact_version="a",
            image_score=FROZEN_THRESHOLD * 0.9, pixel_score=0.05, threshold=FROZEN_THRESHOLD,
        )
    return engine.decide(result, product_id=f"P-{decision.value}", batch_id="B", camera_id="CAM-01")


class TestHumanReviewWorkflow:
    def test_reject_and_hold_enqueued_pass_skipped(self):
        flow = HumanReviewWorkflow()
        assert flow.enqueue(_event(Decision.REJECT)) is True
        assert flow.enqueue(_event(Decision.HOLD)) is True
        assert flow.enqueue(_event(Decision.PASS)) is False
        assert flow.pending_count() == 2

    def test_enqueue_is_idempotent(self):
        flow = HumanReviewWorkflow()
        event = _event(Decision.REJECT)
        assert flow.enqueue(event) and flow.enqueue(event) is False
        assert flow.pending_count() == 1

    def test_confirm_defect_updates_status_and_closes_mes(self):
        mes = MesService()
        flow = HumanReviewWorkflow(mes=mes)
        event = _event(Decision.REJECT)
        mes.create_from_event(event)
        flow.enqueue(event)
        record, updated = flow.submit(
            event.id, reviewer="op-1", outcome=ReviewOutcome.CONFIRM_DEFECT, comment="visible inclusion"
        )
        assert record.reviewer == "op-1" and record.reviewed_at.endswith("Z")
        assert updated.operator_status is OperatorStatus.CONFIRMED_DEFECT
        order = mes.find_by_event(event.id)
        assert order.status is WorkOrderStatus.CLOSED and order.closed_reason == "confirmed_defect"
        assert flow.pending_count() == 0

    def test_false_alarm_closes_mes_as_false_alarm(self):
        mes = MesService()
        flow = HumanReviewWorkflow(mes=mes)
        event = _event(Decision.REJECT)
        mes.create_from_event(event)
        flow.enqueue(event)
        _, updated = flow.submit(event.id, reviewer="op-2", outcome=ReviewOutcome.FALSE_ALARM)
        assert updated.operator_status is OperatorStatus.FALSE_ALARM
        assert mes.find_by_event(event.id).closed_reason == "false_alarm"

    def test_request_recheck_keeps_item_pending(self):
        flow = HumanReviewWorkflow()
        event = _event(Decision.REJECT)
        flow.enqueue(event)
        _, updated = flow.submit(event.id, reviewer="op-3", outcome=ReviewOutcome.REQUEST_RECHECK)
        assert updated.operator_status is OperatorStatus.RECHECK_REQUESTED
        assert flow.pending_count() == 1
        assert flow.recheck_counts()[event.id] == 1
        # second pass resolves it
        _, final = flow.submit(event.id, reviewer="op-3", outcome=ReviewOutcome.CONFIRM_DEFECT)
        assert final.operator_status is OperatorStatus.CONFIRMED_DEFECT
        assert flow.pending_count() == 0

    def test_unknown_event_and_blank_reviewer_rejected(self):
        flow = HumanReviewWorkflow()
        assert flow.submit("nope", reviewer="op", outcome=ReviewOutcome.CONFIRM_DEFECT) == (None, None)
        event = _event(Decision.REJECT)
        flow.enqueue(event)
        with pytest.raises(ValueError):
            flow.submit(event.id, reviewer="   ", outcome=ReviewOutcome.CONFIRM_DEFECT)

    def test_records_are_complete_for_audit(self):
        flow = HumanReviewWorkflow()
        event = _event(Decision.HOLD)
        flow.enqueue(event)
        flow.submit(event.id, reviewer="op-9", outcome=ReviewOutcome.REQUEST_RECHECK, comment="recheck cam")
        record = flow.records()[0]
        assert record.event_id == event.id and record.trace_id == event.trace_id
        assert record.ai_decision == "HOLD" and record.outcome == ReviewOutcome.REQUEST_RECHECK
        assert record.comment == "recheck cam"
        counts = flow.counts()
        assert counts["total"] == 1 and counts["pending"] == 1
