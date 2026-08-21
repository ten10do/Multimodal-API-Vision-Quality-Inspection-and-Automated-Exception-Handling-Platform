"""Phase 3 — VirtualFileCamera: replay existing images as an industrial camera.

Simulates a machine-vision camera whose "sensor" is a folder of steel strip
images. Sequential playback, batch replay and deterministic failure injection
are all seeded, so a given (dataset, seed) always produces the identical
frame sequence. The camera never modifies the dataset (read-only replay).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from industrial_loop.events import utc_now_iso

from .camera_base import (
    CameraAdapter,
    CameraConnectionError,
    CameraNotTriggeredError,
    TriggerInfo,
)
from .frames import CameraFrame

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def write_placeholder_png(path: Path, *, width: int = 1600, height: int = 256, gray: int = 128) -> None:
    """Write a minimal valid 8-bit grayscale PNG (no external deps at runtime)."""
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes([gray & 0xFF] * width) for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _probe_size(path: Path, fallback: tuple[int, int]) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as handle:
            return int(handle.width), int(handle.height)
    except Exception:  # noqa: BLE001 - probing is best-effort; replay must not crash
        return fallback


def _frame_prefix(camera_id: str) -> str:
    digits = "".join(ch for ch in camera_id if ch.isdigit())
    return f"CAM{digits}" if digits else "CAM"


class VirtualFileCamera(CameraAdapter):
    """Deterministic file-backed camera (sequential / batch replay)."""

    def __init__(
        self,
        dataset: str | Path | list[str | Path],
        *,
        camera_id: str = "steel-camera-01",
        frame_prefix: str | None = None,
        seed: int = 42,
        failure_rate: float = 0.0,
        loop: bool = True,
        default_size: tuple[int, int] = (1600, 256),
        require_trigger: bool = True,
    ) -> None:
        super().__init__(camera_id)
        if failure_rate < 0.0 or failure_rate > 1.0:
            raise ValueError("failure_rate must be within [0, 1]")
        if isinstance(dataset, (str, Path)):
            root = Path(dataset)
            if not root.exists():
                raise FileNotFoundError(f"camera dataset not found: {root}")
            if root.is_dir():
                files = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            else:
                files = [root]
        else:
            files = sorted(Path(p) for p in dataset)
        if not files:
            raise ValueError("camera dataset contains no readable images")
        self._images = files
        self.frame_prefix = frame_prefix or _frame_prefix(camera_id)
        self.failure_rate = failure_rate
        self.loop = loop
        self.default_size = default_size
        self.require_trigger = require_trigger
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._connected = False
        self._cursor = 0
        self._sequence = 0
        self._pending_trigger: TriggerInfo | None = None
        self._triggers_issued = 0
        self._soft_trigger_counter = 0

    # -- lifecycle ------------------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        self.health.mark_online()

    def disconnect(self) -> None:
        self._connected = False
        self._pending_trigger = None
        self.health.mark_offline()

    # -- trigger / capture ----------------------------------------------------

    def trigger(self, trigger_id: str | None = None) -> TriggerInfo:
        if not self._connected:
            raise CameraConnectionError(f"camera {self.camera_id} is not connected")
        if trigger_id is None:
            self._soft_trigger_counter += 1
            trigger_id = f"SW_TRIGGER_{self._soft_trigger_counter:06d}"
        self._triggers_issued += 1
        info = TriggerInfo(
            trigger_id=trigger_id, camera_id=self.camera_id, timestamp=utc_now_iso()
        )
        self._pending_trigger = info
        return info

    def capture(self) -> CameraFrame:
        started = time.perf_counter()
        if not self._connected:
            raise CameraConnectionError(f"camera {self.camera_id} is not connected")
        if self.require_trigger and self._pending_trigger is None:
            raise CameraNotTriggeredError(
                f"camera {self.camera_id} requires trigger() before capture()"
            )
        trigger = self._pending_trigger
        self._pending_trigger = None

        if self._cursor >= len(self._images):
            if not self.loop:
                raise CameraConnectionError("camera dataset exhausted (loop disabled)")
            self._cursor = 0
        image_path = self._images[self._cursor]
        self._cursor += 1

        self._sequence += 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        if self.failure_rate > 0.0 and float(self._rng.random()) < self.failure_rate:
            self.health.record_failure(utc_now_iso())
            return CameraFrame(
                frame_id=f"{self.frame_prefix}_{self._sequence:06d}",
                camera_id=self.camera_id,
                image_path=str(image_path),
                width=self.default_size[0],
                height=self.default_size[1],
                trigger_id=trigger.trigger_id if trigger else "UNTRIGGERED",
                capture_status="FAILED",
                sequence_number=self._sequence,
                capture_latency_ms=round(latency_ms, 3),
                error_detail="simulated_sensor_acquisition_failure",
            )

        width, height = _probe_size(image_path, self.default_size)
        self.health.record_success(utc_now_iso())
        return CameraFrame(
            frame_id=f"{self.frame_prefix}_{self._sequence:06d}",
            camera_id=self.camera_id,
            image_path=str(image_path),
            width=width,
            height=height,
            trigger_id=trigger.trigger_id if trigger else "UNTRIGGERED",
            capture_status="SUCCESS",
            sequence_number=self._sequence,
            capture_latency_ms=round(latency_ms, 3),
        )

    def capture_batch(self, size: int, *, trigger_source: str = "PLC") -> list[CameraFrame]:
        """Batch replay: size trigger+capture cycles in deterministic order."""
        frames = []
        for _ in range(size):
            self.trigger(f"{trigger_source}_BATCH_{self._triggers_issued + 1:06d}")
            frames.append(self.capture())
        return frames

    def reset(self) -> None:
        """Restart sequential playback from the first image (same session)."""
        self._cursor = 0
        self._sequence = 0
        self._rng = np.random.default_rng(self._seed)

    # -- status ---------------------------------------------------------------

    def health_check(self) -> dict:
        snapshot = self.health.snapshot()
        snapshot["connected"] = self._connected
        snapshot["dataset_images"] = len(self._images)
        return snapshot

    def get_status(self) -> dict:
        return {
            **self.health.snapshot(),
            "connected": self._connected,
            "cursor": self._cursor,
            "sequence": self._sequence,
            "triggers_issued": self._triggers_issued,
            "pending_trigger": self._pending_trigger.trigger_id if self._pending_trigger else None,
            "loop": self.loop,
            "failure_rate": self.failure_rate,
        }
