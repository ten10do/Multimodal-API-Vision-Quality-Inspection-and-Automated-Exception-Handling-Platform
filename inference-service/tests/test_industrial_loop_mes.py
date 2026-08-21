"""Industrial closed-loop tests: MES work orders (Phase 4)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from industrial_loop.config import FROZEN_THRESHOLD, MODEL_VERSION
from industrial_loop.decision_service import D3InferenceResult, DecisionEngine
from industrial_loop.events import Decision
from industrial_loop.mes_service import MesService, WorkOrderStatus, severity_for


def _reject_event(score: float | None = None) -> object:
    engine = DecisionEngine()
    result = D3InferenceResult(
        ok=True,
        model_version=MODEL_VERSION,
        artifact_version="rel@1.3.0",
        image_score=score if score is not None else FROZEN_THRESHOLD * 1.05,
        pixel_score=0.8,
        threshold=FROZEN_THRESHOLD,
    )
    return engine.decide(result, product_id="P1", batch_id="B-1", camera_id="CAM-01")


class TestMesWorkOrders:
    def test_reject_creates_order_with_required_fields(self):
        mes = MesService()
        event = _reject_event()
        order = mes.create_from_event(event)
        assert order is not None
        assert order.status is WorkOrderStatus.OPEN
        assert order.batch_id == "B-1"
        assert order.defect_type == "steel-surface-anomaly"
        assert order.image_id == "P1"
        assert order.severity in {"LOW", "MEDIUM", "HIGH"}
        assert order.work_order_id.startswith("wo-")

    def test_pass_creates_nothing(self):
        mes = MesService()
        engine = DecisionEngine()
        event = engine.decide(
            D3InferenceResult(
                ok=True, model_version=MODEL_VERSION, artifact_version="a",
                image_score=FROZEN_THRESHOLD * 0.9, pixel_score=0.05, threshold=FROZEN_THRESHOLD,
            ),
            product_id="P2", batch_id="B-1", camera_id="CAM-01",
        )
        assert event.decision is Decision.PASS
        assert mes.create_from_event(event) is None
        assert mes.counts()["total"] == 0

    def test_idempotent_per_event(self):
        mes = MesService()
        event = _reject_event()
        first = mes.create_from_event(event)
        again = mes.create_from_event(event)
        assert first.work_order_id == again.work_order_id
        assert mes.counts()["total"] == 1

    def test_lifecycle_open_processing_closed(self):
        mes = MesService()
        order = mes.create_from_event(_reject_event())
        order = mes.advance(order.work_order_id)
        assert order.status is WorkOrderStatus.PROCESSING
        order = mes.advance(order.work_order_id)
        assert order.status is WorkOrderStatus.CLOSED
        with pytest.raises(ValueError):
            mes.advance(order.work_order_id)  # already closed

    def test_close_records_reason_and_reviewer(self):
        mes = MesService()
        order = mes.create_from_event(_reject_event())
        closed = mes.close(order.work_order_id, reason="false_alarm", reviewed_by="op-3")
        assert closed.status is WorkOrderStatus.CLOSED
        assert closed.closed_reason == "false_alarm" and closed.reviewed_by == "op-3"

    def test_severity_from_margin(self):
        assert severity_for(_reject_event(FROZEN_THRESHOLD * 1.05)) == "HIGH"
        assert severity_for(_reject_event(FROZEN_THRESHOLD * 1.001)) == "MEDIUM"

    def test_find_by_event_and_counts(self):
        mes = MesService()
        event = _reject_event()
        order = mes.create_from_event(event)
        assert mes.find_by_event(event.id).work_order_id == order.work_order_id
        assert mes.counts() == {"OPEN": 1, "PROCESSING": 0, "CLOSED": 0, "total": 1}
