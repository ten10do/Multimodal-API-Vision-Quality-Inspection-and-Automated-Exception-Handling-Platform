"""Virtual industrial camera acquisition layer (peripheral, vendor-neutral).

Architecture position:

    Dataset / Replay -> CameraAdapter -> CameraFrame -> D3 Inference
        -> Decision Engine -> PLC / MES

This package simulates the factory image-acquisition layer. It never touches
the D3 model, weights, bank or threshold: it only produces ``CameraFrame``
objects that feed the existing inference pipeline unchanged.
"""
from industrial_loop.camera.camera_base import (
    CameraAdapter,
    CameraConnectionError,
    CameraError,
    CameraHealthMonitor,
    CameraHealthState,
    CameraInterlockError,
    CameraNotTriggeredError,
    CaptureStatus,
    TriggerInfo,
)
from industrial_loop.camera.frames import CameraFrame

__all__ = [
    "CameraAdapter",
    "CameraConnectionError",
    "CameraError",
    "CameraFrame",
    "CameraHealthMonitor",
    "CameraHealthState",
    "CameraInterlockError",
    "CameraNotTriggeredError",
    "CaptureStatus",
    "TriggerInfo",
]
