"""Phase 4 — MES work-order mock service.

An AI REJECT automatically creates a work order; operators work it
OPEN -> PROCESSING -> CLOSED. Idempotent per inspection event: replaying the
same event never duplicates a ticket.
"""
from __future__ import annotations

import threading
import uuid
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .events import InspectionEvent, utc_now_iso


class WorkOrderStatus(str, Enum):
    OPEN = "OPEN"
    PROCESSING = "PROCESSING"
    CLOSED = "CLOSED"


class WorkOrder(BaseModel):
    work_order_id: str = Field(default_factory=lambda: f"wo-{uuid.uuid4().hex[:16]}")
    event_id: str
    batch_id: str
    defect_type: str
    image_id: str
    severity: str
    status: WorkOrderStatus = WorkOrderStatus.OPEN
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    closed_reason: str | None = None
    reviewed_by: str | None = None

    @field_validator("severity")
    @classmethod
    def _severity(cls, value: str) -> str:
        if value not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("severity must be LOW/MEDIUM/HIGH")
        return value


def severity_for(event: InspectionEvent) -> str:
    """Deterministic severity from the score margin above the frozen threshold."""
    if event.image_score is None or event.threshold is None:
        return "MEDIUM"
    ratio = event.image_score / event.threshold
    if ratio >= 1.02:
        return "HIGH"
    if ratio >= 1.0:
        return "MEDIUM"
    return "LOW"


class MesService:
    def __init__(self) -> None:
        self._orders: dict[str, WorkOrder] = {}       # work_order_id -> order
        self._by_event: dict[str, str] = {}           # event_id -> work_order_id
        self._lock = threading.Lock()

    # -- creation -------------------------------------------------------------

    def create_from_event(self, event: InspectionEvent) -> WorkOrder | None:
        """Auto-create a ticket for an AI REJECT (idempotent by event id)."""
        if event.decision.value != "REJECT":
            return None
        with self._lock:
            existing = self._by_event.get(event.id)
            if existing is not None:
                return self._orders[existing]
            order = WorkOrder(
                event_id=event.id,
                batch_id=event.batch_id,
                defect_type="steel-surface-anomaly",
                image_id=event.product_id,
                severity=severity_for(event),
            )
            self._orders[order.work_order_id] = order
            self._by_event[event.id] = order.work_order_id
            return order

    # -- lifecycle ------------------------------------------------------------

    def _get(self, work_order_id: str) -> WorkOrder:
        order = self._orders.get(work_order_id)
        if order is None:
            raise KeyError(f"unknown work_order_id {work_order_id}")
        return order

    def advance(self, work_order_id: str) -> WorkOrder:
        with self._lock:
            order = self._get(work_order_id)
            if order.status is WorkOrderStatus.OPEN:
                updated = order.model_copy(update={"status": WorkOrderStatus.PROCESSING, "updated_at": utc_now_iso()})
            elif order.status is WorkOrderStatus.PROCESSING:
                updated = order.model_copy(update={"status": WorkOrderStatus.CLOSED, "updated_at": utc_now_iso(), "closed_reason": "completed"})
            else:
                raise ValueError(f"work order {work_order_id} already CLOSED")
            self._orders[work_order_id] = updated
            return updated

    def close(self, work_order_id: str, *, reason: str, reviewed_by: str | None = None) -> WorkOrder:
        with self._lock:
            order = self._get(work_order_id)
            updated = order.model_copy(
                update={
                    "status": WorkOrderStatus.CLOSED,
                    "closed_reason": reason,
                    "reviewed_by": reviewed_by,
                    "updated_at": utc_now_iso(),
                }
            )
            self._orders[work_order_id] = updated
            return updated

    # -- queries --------------------------------------------------------------

    def get(self, work_order_id: str) -> WorkOrder:
        with self._lock:
            return self._get(work_order_id)

    def find_by_event(self, event_id: str) -> WorkOrder | None:
        with self._lock:
            wo = self._by_event.get(event_id)
            return self._orders.get(wo) if wo else None

    def list(self, status: WorkOrderStatus | None = None) -> list[WorkOrder]:
        with self._lock:
            orders = list(self._orders.values())
        if status is not None:
            orders = [o for o in orders if o.status is status]
        return sorted(orders, key=lambda o: o.created_at)

    def counts(self) -> dict:
        base = {s.value: 0 for s in WorkOrderStatus}
        for order in self.list():
            base[order.status.value] += 1
        base["total"] = len(self.list())
        return base
