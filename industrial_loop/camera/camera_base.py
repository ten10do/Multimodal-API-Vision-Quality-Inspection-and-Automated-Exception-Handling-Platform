"""Phase 1 — abstract camera adapter interface (industrial-SDK style).

The interface mirrors the shape of a typical machine-vision camera SDK
(connect / trigger / capture / disconnect + health) while staying strictly
vendor-neutral: no GigE Vision, USB3 Vision or vendor SDK types leak in here.
Concrete adapters (today the ``VirtualFileCamera``; tomorrow a GenICam-based
one) implement this contract, so the rest of the closed loop only depends on
the abstraction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (frames -> camera_base)
    from .frames import CameraFrame


class CaptureStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class CameraHealthState(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"


class CameraError(Exception):
    """Base class for acquisition-layer failures."""


class CameraConnectionError(CameraError):
    """The camera is not connected / was disconnected (fail-close -> HOLD)."""


class CameraNotTriggeredError(CameraError):
    """capture() was called without an armed trigger in on-demand mode."""


class CameraInterlockError(CameraError):
    """The PLC interlock refused the trigger (line STOPped)."""


@dataclass(frozen=True)
class TriggerInfo:
    """Result of arming/firing one acquisition trigger."""

    trigger_id: str
    camera_id: str
    timestamp: str
    source: str = "PLC"


@dataclass
class CameraHealthMonitor:
    """Phase 5 — camera health state and counters.

    Semantics: OFFLINE until connect(); ONLINE while captures succeed; ERROR
    after a failed capture (recovers to ONLINE on the next success); OFFLINE
    again after disconnect(). A non-ONLINE state must route the affected
    product to HOLD via the decision engine (fail-close), never to PASS.
    """

    camera_id: str
    state: CameraHealthState = CameraHealthState.OFFLINE
    last_capture_time: str | None = None
    frame_count: int = 0
    failure_count: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def mark_online(self) -> None:
        with self._lock:
            self.state = CameraHealthState.ONLINE

    def mark_offline(self) -> None:
        with self._lock:
            self.state = CameraHealthState.OFFLINE

    def record_success(self, timestamp: str) -> None:
        with self._lock:
            self.state = CameraHealthState.ONLINE
            self.frame_count += 1
            self.last_capture_time = timestamp

    def record_failure(self, timestamp: str) -> None:
        with self._lock:
            self.state = CameraHealthState.ERROR
            self.failure_count += 1
            self.last_capture_time = timestamp

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "state": self.state.value,
                "last_capture_time": self.last_capture_time,
                "frame_count": self.frame_count,
                "failure_count": self.failure_count,
            }

    @property
    def is_online(self) -> bool:
        return self.state is CameraHealthState.ONLINE


class CameraAdapter(ABC):
    """Abstract industrial camera adapter.

    Lifecycle: ``connect()`` opens the device/session; each shot is
    ``trigger()`` (arm/fire, PLC-driven in production) followed by
    ``capture()``; ``disconnect()`` closes the session. ``health_check()``
    probes the device; ``get_status()`` reports connection + counters.
    Adapters are context managers so ``with camera:`` maps to connect/
    disconnect like SDK session objects.
    """

    def __init__(self, camera_id: str) -> None:
        if not camera_id:
            raise ValueError("camera_id is required")
        self.camera_id = camera_id
        self.health = CameraHealthMonitor(camera_id=camera_id)

    @abstractmethod
    def connect(self) -> None:
        """Open the camera session. Idempotent."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the camera session. Idempotent."""

    @abstractmethod
    def trigger(self, trigger_id: str | None = None) -> TriggerInfo:
        """Arm/fire one acquisition trigger (software or PLC-driven)."""

    @abstractmethod
    def capture(self) -> CameraFrame:
        """Acquire one frame stamped with the pending trigger id."""

    @abstractmethod
    def health_check(self) -> dict:
        """Probe device health; returns the health snapshot plus connectivity."""

    @abstractmethod
    def get_status(self) -> dict:
        """Full adapter status (connection, cursor/counters, health)."""

    def __enter__(self) -> "CameraAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()
