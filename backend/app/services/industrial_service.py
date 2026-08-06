"""Industrial orchestration service (Phase 7).

Quality Decision -> Industrial Command -> PLC Adapter -> ACK -> plc_events
persistence -> inspection industrial state -> MES sync.

Fail-safe: any PLC communication failure or unknown state lands in
SAFE_HOLD; RELEASE is only ever produced by an explicit PASS result.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..industrial.commands import IndustrialCommand, decision_to_command
from ..industrial.mes_adapter import get_mes_adapter
from ..industrial.plc_adapter import PlcNack, PlcUnreachable, get_plc_adapter
from ..industrial.state_machine import next_state, terminal_for
from ..models import Inspection, PlcEvent

logger = logging.getLogger(__name__)

EVENT_KIND = {"inspection": "inspection", "final": "final"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndustrialService:
    def __init__(self) -> None:
        self.plc = get_plc_adapter()
        self.mes = get_mes_adapter()
        settings = get_settings()
        self.plc_max_retries = settings.plc_max_retries
        self.plc_enabled = settings.plc_enabled
        self.mes_enabled = settings.mes_enabled

    async def process_result(
        self,
        session: AsyncSession,
        inspection: Inspection,
        *,
        final_quality_result: str | None,
        process_status: str,
        review_resolved: bool = False,
        review_decision: str | None = None,
        reviewed_by: str | None = None,
    ) -> None:
        """Translate a final business result into an industrial command and
        execute it end-to-end (PLC + persistence + MES).

        Three-layer semantics (see docs):
        - desired_command   = what the system wants the field layer to do
        - execution_status  = whether the command was really sent / ACKed
        - industrial_state  = the product's actual field state

        plc_enabled=False records NOT_INTEGRATED: the PLC was never engaged.
        It is NOT a communication fault and MUST NOT be faked as SAFE_HOLD,
        HELD, RELEASED or REJECTED. desired_command still expresses what the
        system would have asked the PLC to do.
        """
        product = inspection.product
        product_id = product.product_id if product else inspection.product_id
        line = product.production_line if product else ""
        station = product.station if product else ""

        command, _ = decision_to_command(
            inspection_id=inspection.inspection_id,
            product_id=product_id,
            final_quality_result=final_quality_result,
            process_status=process_status,
            review_resolved=review_resolved,
            review_decision=review_decision,
            timestamp=_utcnow(),
        )
        inspection.desired_command = command.command_type

        if not self.plc_enabled:
            # NOT INTEGRATED: compute the desired command, never call the
            # adapter, never fabricate an ACK, never claim field execution.
            inspection.execution_status = "NOT_INTEGRATED"
            inspection.industrial_state = "NOT_INTEGRATED"
            inspection.industrial_final_state = "NOT_INTEGRATED"
            inspection.plc_command = command.command_type
            inspection.plc_status = "NOT_INTEGRATED"
            inspection.plc_adapter_type = "none"
            inspection.plc_reason_code = "plc_integration_disabled"
            inspection.plc_latency_ms = None
            session.add(
                PlcEvent(
                    command_id=command.command_id,
                    product_id=product_id,
                    inspection_id=inspection.inspection_id,
                    command=command.command_type,
                    desired_command=command.command_type,
                    execution_status="NOT_INTEGRATED",
                    industrial_state="NOT_INTEGRATED",
                    adapter_type="none",
                    request_payload=command.to_payload(),
                    response=None,
                    status="NOT_INTEGRATED",
                    retry_count=0,
                    latency_ms=None,
                    reason_code="plc_integration_disabled",
                    acknowledged_at=None,
                )
            )
            # MES sync is best-effort even when the PLC is not integrated:
            # a MES failure must never roll back the completed AI inspection
            # and never change NOT_INTEGRATED (12).
            if not self.mes_enabled:
                inspection.mes_sync_status = "PENDING"
            else:
                try:
                    await self._sync_mes(
                        session, inspection,
                        product_id=product_id, batch_id=inspection.batch_id, line=line, station=station,
                        final_result=final_quality_result, reviewed_by=reviewed_by,
                        industrial_state="NOT_INTEGRATED",
                    )
                    inspection.mes_sync_status = "SYNCED"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("mes sync failed inspection=%s: %s", inspection.inspection_id, exc)
                    inspection.mes_sync_status = "FAILED"
            return

        # ---- PLC integration enabled: really talk to the field layer ----
        current = inspection.industrial_state or "CAPTURED"
        inspection.industrial_state = next_state(current, "command_pending")

        plc_status = "PENDING"
        execution_status = "PENDING"
        latency_ms: float | None = None
        response: dict | None = None
        attempts = 0
        acknowledged_at: datetime | None = None

        last_error: Exception | None = None
        for attempt in range(self.plc_max_retries + 1):
            attempts = attempt + 1
            try:
                result = await self.plc.send_command(command)
                latency_ms = result.latency_ms
                response = result.response
                if result.duplicate:
                    plc_status = "DUPLICATE_SUPPRESSED"
                    execution_status = "DUPLICATE_SUPPRESSED"
                else:
                    plc_status = "ACK"
                    execution_status = "ACK"
                    acknowledged_at = datetime.now(timezone.utc)
                break
            except PlcUnreachable as exc:
                last_error = exc
                plc_status = "TIMEOUT" if "timeout" in str(exc).lower() else "ERROR"
                execution_status = plc_status
                logger.warning("plc attempt=%d failed inspection=%s: %s", attempt + 1, inspection.inspection_id, exc)
                continue
            except PlcNack as exc:
                last_error = exc
                plc_status = "NACK"
                execution_status = "NACK"
                logger.warning("plc nack inspection=%s: %s", inspection.inspection_id, exc)
                break  # NACK is a definitive answer; no retry
        if plc_status not in ("ACK", "DUPLICATE_SUPPRESSED"):
            response = {"error": str(last_error or "plc_failed"), "attempts": attempts}

        # update inspection industrial fields
        inspection.plc_command = command.command_type
        inspection.plc_status = plc_status
        inspection.execution_status = execution_status
        inspection.plc_adapter_type = self.plc.name
        inspection.plc_reason_code = command.reason_code
        inspection.plc_latency_ms = latency_ms

        if plc_status in ("ACK", "DUPLICATE_SUPPRESSED"):
            inspection.industrial_state = next_state(inspection.industrial_state or "CAPTURED", "plc_ack")
            terminal = terminal_for(command.command_type, acked=True)
            if process_status == "FAILED" or command.reason_code in (
                "system_failed", "inference_timeout", "plc_unreachable", "unknown_state",
            ):
                # system failure even with a HOLD ACK lands in SAFE_HOLD:
                # the HOLD was a safe-state command, not a review hold.
                terminal = "SAFE_HOLD"
            inspection.industrial_final_state = terminal
        elif plc_status == "NACK":
            # the PLC definitively refused the command; product state is
            # ambiguous -> COMMAND_FAILED (fail-safe, never RELEASE)
            inspection.industrial_state = next_state(inspection.industrial_state or "CAPTURED", "plc_error")
            inspection.industrial_final_state = "COMMAND_FAILED"
        else:
            # timeout / offline / transport error -> SAFE_HOLD
            inspection.industrial_state = next_state(inspection.industrial_state or "CAPTURED", "plc_unreachable")
            inspection.industrial_final_state = "SAFE_HOLD"

        # persist the plc event (audit + idempotency evidence); the recorded
        # industrial_state is the real resolved product state
        session.add(
            PlcEvent(
                command_id=command.command_id,
                product_id=product_id,
                inspection_id=inspection.inspection_id,
                command=command.command_type,
                desired_command=command.command_type,
                execution_status=execution_status,
                industrial_state=inspection.industrial_final_state,
                adapter_type=self.plc.name,
                request_payload=command.to_payload(),
                response=response,
                status=plc_status,
                retry_count=max(0, attempts - 1),
                latency_ms=latency_ms,
                reason_code=command.reason_code,
                acknowledged_at=acknowledged_at,
            )
        )

        # MES sync (best-effort, non-blocking for the PLC flow). The MES must
        # see the REAL industrial state: PLC RELEASE + MES failure keeps
        # RELEASED; PLC failure must not be reported to the MES as RELEASED.
        # mes_enabled=False records PENDING (bounded retry later) and never
        # rolls back the completed inspection.
        if not self.mes_enabled:
            inspection.mes_sync_status = "PENDING"
        else:
            try:
                await self._sync_mes(
                    session, inspection,
                    product_id=product_id, batch_id=inspection.batch_id, line=line, station=station,
                    final_result=final_quality_result, reviewed_by=reviewed_by,
                    industrial_state=inspection.industrial_final_state or inspection.industrial_state,
                )
                inspection.mes_sync_status = "SYNCED"
            except Exception as exc:  # noqa: BLE001
                logger.warning("mes sync failed inspection=%s: %s", inspection.inspection_id, exc)
                inspection.mes_sync_status = "FAILED"

    async def _sync_mes(
        self,
        session: AsyncSession,
        inspection: Inspection,
        *,
        product_id: str,
        batch_id: str | None,
        line: str,
        station: str,
        final_result: str | None,
        reviewed_by: str | None,
        industrial_state: str | None,
    ) -> None:
        await self.mes.post_inspection_result(
            inspection_id=inspection.inspection_id,
            product_id=product_id,
            batch_id=batch_id,
            line=line,
            station=station,
            ai_result=inspection.quality_result.value if inspection.quality_result else None,
            model_version=inspection.model_version,
            rule_version=inspection.rule_version,
            industrial_state=industrial_state,
            timestamp=_utcnow(),
        )
        if final_result:
            await self.mes.post_final_quality_result(
                inspection_id=inspection.inspection_id,
                product_id=product_id,
                batch_id=batch_id,
                final_result=final_result,
                reviewed_by=reviewed_by,
                industrial_state=industrial_state,
                timestamp=_utcnow(),
            )


def get_industrial_service() -> IndustrialService:
    return IndustrialService()
