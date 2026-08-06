"""Industrial decision / command layer (Phase 7).

Boundary: the AI system must NOT drive physical actuators directly. A Quality
Result is translated here into an industrial command with an explicit reason
code, and the command is executed through a PLC adapter.

Fail-safe rule: any inability to determine quality (inference timeout,
backend unavailable, PLC communication failure, unknown state) leads to
HOLD / SAFE_STATE. It is never allowed to default to RELEASE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandType = Literal["RELEASE", "REJECT", "HOLD", "RESET"]

# product FAIL and system FAILED share "not released" but the industrial
# reasons are different and MUST be recorded with different reason codes.
ReasonCode = Literal[
    "quality_pass",      # AI/human PASS -> release
    "product_defect",    # product FAIL -> reject
    "review_pending",    # AI REVIEW -> hold until human decision
    "review_pass",       # human PASS -> release
    "review_fail",       # human FAIL -> reject
    "system_failed",     # processing failure (inference error etc.)
    "inference_timeout",  # specific system failure cause
    "plc_unreachable",   # PLC offline / communication failure (fail-safe hold)
    "unknown_state",     # state not recognized -> fail-safe hold
    "plc_integration_disabled",  # PLC never engaged: NOT_INTEGRATED, not a fault
]


@dataclass(frozen=True)
class IndustrialCommand:
    command_id: str
    product_id: str
    inspection_id: str
    command_type: CommandType
    reason_code: ReasonCode
    timestamp: str

    def to_payload(self) -> dict:
        return {
            "command_id": self.command_id,
            "product_id": self.product_id,
            "inspection_id": self.inspection_id,
            "command_type": self.command_type,
            "reason_code": self.reason_code,
            "timestamp": self.timestamp,
        }


def command_id_for(inspection_id: str, command_type: CommandType) -> str:
    """Idempotency key: the same inspection + command type always maps to the
    same command_id, so network retries cannot double-execute."""
    return f"cmd-{inspection_id}-{command_type}"


def decision_to_command(
    *,
    inspection_id: str,
    product_id: str,
    final_quality_result: str | None,
    process_status: str,
    review_resolved: bool = False,
    review_decision: str | None = None,
    timestamp: str,
) -> tuple[IndustrialCommand, bool]:
    """Map a final business result to an industrial command.

    Returns (command, is_hold_command). Never returns RELEASE on uncertainty.
    """
    from datetime import datetime, timezone

    ts = timestamp or datetime.now(timezone.utc).isoformat()

    # system FAILED -> SAFE HOLD with a system reason code.
    # process_status arrives as the enum value ("failed"); normalize so the
    # mapping is robust to case (the field-layer meaning is case-insensitive).
    if process_status.upper() == "FAILED":
        reason: ReasonCode = "inference_timeout" if "timeout" in str(final_quality_result or "").lower() else "system_failed"
        return (
            IndustrialCommand(command_id_for(inspection_id, "HOLD"), product_id, inspection_id, "HOLD", reason, ts),
            True,
        )

    # REVIEW handled by a human -> final result decides
    if review_resolved:
        if review_decision == "PASS" or final_quality_result == "PASS":
            return (
                IndustrialCommand(command_id_for(inspection_id, "RELEASE"), product_id, inspection_id, "RELEASE", "review_pass", ts),
                False,
            )
        return (
            IndustrialCommand(command_id_for(inspection_id, "REJECT"), product_id, inspection_id, "REJECT", "review_fail", ts),
            False,
        )

    # plain AI/human result
    if final_quality_result == "PASS":
        return (
            IndustrialCommand(command_id_for(inspection_id, "RELEASE"), product_id, inspection_id, "RELEASE", "quality_pass", ts),
            False,
        )
    if final_quality_result == "FAIL":
        return (
            IndustrialCommand(command_id_for(inspection_id, "REJECT"), product_id, inspection_id, "REJECT", "product_defect", ts),
            False,
        )
    if final_quality_result == "REVIEW":
        return (
            IndustrialCommand(command_id_for(inspection_id, "HOLD"), product_id, inspection_id, "HOLD", "review_pending", ts),
            True,
        )

    # unknown state -> fail-safe hold
    return (
        IndustrialCommand(command_id_for(inspection_id, "HOLD"), product_id, inspection_id, "HOLD", "unknown_state", ts),
        True,
    )
