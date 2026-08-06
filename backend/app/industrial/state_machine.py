"""Industrial state machine (Phase 7).

Per-product, explainable business state. Unknown states must never default to
RELEASE; the fail-safe terminal is SAFE_HOLD.
"""

from __future__ import annotations

from typing import Literal

IndustrialState = Literal[
    "CAPTURED",
    "INSPECTING",
    "PASS",
    "FAIL",
    "REVIEW",
    "PLC_COMMAND_PENDING",
    "PLC_ACKNOWLEDGED",
    "RELEASED",
    "REJECTED",
    "HELD",
    "PROCESSING_FAILED",
    "SAFE_HOLD",
    "NOT_INTEGRATED",
    "COMMAND_FAILED",
]

# terminal states
TERMINAL = {"RELEASED", "REJECTED", "HELD", "SAFE_HOLD", "NOT_INTEGRATED", "COMMAND_FAILED"}


def next_state(current: IndustrialState, event: str) -> IndustrialState:
    """Transition table. Any unhandled combination falls to SAFE_HOLD."""
    transitions: dict[tuple[IndustrialState, str], IndustrialState] = {
        ("CAPTURED", "inspecting"): "INSPECTING",
        ("INSPECTING", "quality_pass"): "PASS",
        ("INSPECTING", "product_defect"): "FAIL",
        ("INSPECTING", "review_pending"): "REVIEW",
        ("INSPECTING", "system_failed"): "PROCESSING_FAILED",
        ("PASS", "command_pending"): "PLC_COMMAND_PENDING",
        ("FAIL", "command_pending"): "PLC_COMMAND_PENDING",
        ("REVIEW", "command_pending"): "PLC_COMMAND_PENDING",
        ("PROCESSING_FAILED", "command_pending"): "PLC_COMMAND_PENDING",
        # NOT_INTEGRATED is a stable terminal: no PLC ever engaged, and it is
        # never overwritten by a later command (there is none).
        ("CAPTURED", "not_integrated"): "NOT_INTEGRATED",
        ("PLC_COMMAND_PENDING", "plc_ack"): "PLC_ACKNOWLEDGED",
        ("PLC_COMMAND_PENDING", "plc_error"): "COMMAND_FAILED",
        ("PLC_COMMAND_PENDING", "plc_unreachable"): "SAFE_HOLD",
        ("PLC_ACKNOWLEDGED", "released"): "RELEASED",
        ("PLC_ACKNOWLEDGED", "rejected"): "REJECTED",
        ("PLC_ACKNOWLEDGED", "held"): "HELD",
        ("PLC_ACKNOWLEDGED", "safe_hold"): "SAFE_HOLD",
        ("HELD", "review_pass"): "PASS",
        ("HELD", "review_fail"): "FAIL",
        ("REVIEW", "review_pass"): "PASS",
        ("REVIEW", "review_fail"): "FAIL",
    }
    return transitions.get((current, event), "SAFE_HOLD")


def terminal_for(command_type: str, acked: bool = True) -> IndustrialState:
    """Map an ACKed PLC command to the terminal industrial state."""
    if not acked:
        return "SAFE_HOLD"
    return {
        "RELEASE": "RELEASED",
        "REJECT": "REJECTED",
        "HOLD": "HELD",
        "RESET": "HELD",
    }.get(command_type, "SAFE_HOLD")
