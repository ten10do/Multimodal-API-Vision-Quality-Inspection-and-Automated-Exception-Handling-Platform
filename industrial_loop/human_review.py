"""Phase 5 — human review workflow for AI REJECT / HOLD events.

Every REJECT or HOLD is enqueued for operator review. The operator answers
CONFIRM_DEFECT / FALSE_ALARM / REQUEST_RECHECK; the workflow records who,
when, what and why, updates the event's ``operator_status`` and drives the
MES work order to a closed state (false alarm or confirmed defect).
"""
from __future__ import annotations

import threading
import uuid
from enum import Enum

from pydantic import BaseModel, Field

from .events import (
    Decision,
    InspectionEvent,
    OperatorStatus,
    utc_now_iso,
)


class ReviewOutcome(str, Enum):
    CONFIRM_DEFECT = "CONFIRM_DEFECT"
    FALSE_ALARM = "FALSE_ALARM"
    REQUEST_RECHECK = "REQUEST_RECHECK"


_OUTCOME_TO_STATUS = {
    ReviewOutcome.CONFIRM_DEFECT: OperatorStatus.CONFIRMED_DEFECT,
    ReviewOutcome.FALSE_ALARM: OperatorStatus.FALSE_ALARM,
    ReviewOutcome.REQUEST_RECHECK: OperatorStatus.RECHECK_REQUESTED,
}


class ReviewRecord(BaseModel):
    review_id: str = Field(default_factory=lambda: f"rev-{uuid.uuid4().hex[:16]}")
    event_id: str
    trace_id: str
    ai_decision: str
    image_score: float | None = None
    reviewer: str
    outcome: ReviewOutcome
    comment: str = ""
    reviewed_at: str = Field(default_factory=utc_now_iso)


class HumanReviewWorkflow:
    """Queue + record keeping. MES closing is delegated via an optional hook."""

    def __init__(self, mes=None) -> None:  # noqa: ANN001 - MesService, optional to avoid a cycle
        self._mes = mes
        self._pending: dict[str, InspectionEvent] = {}
        self._records: list[ReviewRecord] = []
        self._rechecks: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- queue ----------------------------------------------------------------

    def enqueue(self, event: InspectionEvent) -> bool:
        """REJECT/HOLD go to review; PASS never does."""
        if event.decision not in (Decision.REJECT, Decision.HOLD):
            return False
        with self._lock:
            if event.id in self._pending:
                return False  # idempotent
            self._pending[event.id] = event
            return True

    def pending(self) -> list[InspectionEvent]:
        with self._lock:
            return sorted(self._pending.values(), key=lambda e: e.timestamp)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- submission -----------------------------------------------------------

    def submit(
        self,
        event_id: str,
        *,
        reviewer: str,
        outcome: ReviewOutcome,
        comment: str = "",
    ) -> tuple[ReviewRecord | None, InspectionEvent | None]:
        if not reviewer.strip():
            raise ValueError("reviewer name is required")
        with self._lock:
            event = self._pending.get(event_id)
            if event is None:
                return None, None
            record = ReviewRecord(
                event_id=event.id,
                trace_id=event.trace_id,
                ai_decision=event.decision.value,
                image_score=event.image_score,
                reviewer=reviewer.strip(),
                outcome=outcome,
                comment=comment,
            )
            self._records.append(record)
            new_status = _OUTCOME_TO_STATUS[outcome]
            updated = event.with_updates(operator_status=new_status)
            if outcome is ReviewOutcome.REQUEST_RECHECK:
                # stays in the queue for a second pass
                self._rechecks[event_id] = self._rechecks.get(event_id, 0) + 1
                self._pending[event_id] = updated
            else:
                del self._pending[event_id]
            mes_order = None
            if self._mes is not None and outcome is not ReviewOutcome.REQUEST_RECHECK:
                mes_order = self._mes.find_by_event(event_id)
                if mes_order is not None:
                    reason = "confirmed_defect" if outcome is ReviewOutcome.CONFIRM_DEFECT else "false_alarm"
                    self._mes.close(mes_order.work_order_id, reason=reason, reviewed_by=record.reviewer)
            return record, updated

    # -- queries --------------------------------------------------------------

    def records(self) -> list[ReviewRecord]:
        with self._lock:
            return list(self._records)

    def recheck_counts(self) -> dict:
        with self._lock:
            return dict(self._rechecks)

    def counts(self) -> dict:
        with self._lock:
            base = {o.value: 0 for o in ReviewOutcome}
            for record in self._records:
                base[record.outcome.value] += 1
            base["total"] = len(self._records)
            base["pending"] = len(self._pending)
            return base
