"""Phase 4/5 — PLC-driven camera trigger flow + fail-close bridge.

Industrial cycle implemented here:

    PLC READY/RUNNING -> camera trigger -> capture -> inference -> decision -> PLC action

The service owns the trigger-id namespace (``PLC_TRIGGER_000001``...), enforces
the safety interlock (a STOPped line refuses triggers) and converts acquisition
failures into ``D3InferenceResult.failure(...)`` so the existing decision
engine routes them to HOLD — the camera layer never invents a PASS.
"""
from __future__ import annotations

from dataclasses import dataclass

from industrial_loop.decision_service import D3InferenceResult
from industrial_loop.events import utc_now_iso
from industrial_loop.plc_adapter import InMemoryPlc, PLCState

from .camera_base import (
    CameraAdapter,
    CameraError,
    CameraHealthState,
    CameraInterlockError,
    CaptureStatus,
)
from .frames import CameraFrame


@dataclass(frozen=True)
class TriggerRecord:
    trigger_id: str
    camera_id: str
    timestamp: str
    source: str = "PLC"
    plc_state_before: str = "RUNNING"

    def short(self) -> dict:
        return {
            "trigger_id": self.trigger_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "plc_state_before": self.plc_state_before,
        }


class CameraTriggerService:
    """PLC-gated trigger + capture orchestration for one camera."""

    def __init__(self, plc: InMemoryPlc | None = None, *, source: str = "PLC") -> None:
        self.plc = plc
        self.source = source
        self._counter = 0

    @property
    def triggers_issued(self) -> int:
        return self._counter

    def request_trigger(self, camera: CameraAdapter) -> TriggerRecord:
        """Fire one trigger if the PLC interlock allows it."""
        plc_state = None
        if self.plc is not None:
            plc_state = self.plc.state
            if plc_state is PLCState.STOP:
                raise CameraInterlockError(
                    f"PLC interlock: line STOP, trigger refused for {camera.camera_id}"
                )
        self._counter += 1
        info = camera.trigger(f"{self.source}_TRIGGER_{self._counter:06d}")
        return TriggerRecord(
            trigger_id=info.trigger_id,
            camera_id=info.camera_id,
            timestamp=info.timestamp,
            source=info.source,
            plc_state_before=(plc_state.value if plc_state else "UNKNOWN"),
        )

    def capture_cycle(self, camera: CameraAdapter) -> tuple[TriggerRecord, CameraFrame]:
        """One full trigger+capture cycle (acquisition errors propagate)."""
        record = self.request_trigger(camera)
        return record, camera.capture()


def safe_inference_result(
    frame: CameraFrame | None,
    health_snapshot: dict | None = None,
    *,
    error: str | None = None,
) -> D3InferenceResult | None:
    """Fail-close bridge from the acquisition layer to the decision engine.

    Returns ``None`` when the frame is healthy (normal inference should run);
    otherwise returns a failure result that the decision engine maps to
    HOLD / AI_SYSTEM_FAILURE. Never returns a passing result.
    """
    if error is not None:
        return D3InferenceResult.failure(f"camera_error:{error}")
    state = (health_snapshot or {}).get("state", CameraHealthState.ONLINE.value)
    if state != CameraHealthState.ONLINE.value:
        # an unhealthy camera dominates: even a "good-looking" frame is not
        # trustworthy, so route to HOLD before anything else
        return D3InferenceResult.failure(f"camera_health_{state.lower()}")
    if frame is None:
        return D3InferenceResult.failure("camera_error:no_frame")
    if frame.capture_status is not CaptureStatus.SUCCESS:
        return D3InferenceResult.failure(
            f"camera_capture_failed:{frame.error_detail or 'unknown'}",
            latency_ms=frame.capture_latency_ms,
        )
    return None
