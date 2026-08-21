"""Phase 2 — frame event model.

A ``CameraFrame`` is the single contract between the acquisition layer and the
(existing, unchanged) inference pipeline: it carries the image reference plus
the industrial metadata (trigger id, camera id, capture status) required for
end-to-end traceability.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from industrial_loop.events import utc_now_iso

from .camera_base import CaptureStatus


class CameraFrame(BaseModel):
    frame_id: str = Field(description="unique frame id, e.g. CAM01_000001")
    camera_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    image_path: str
    width: int
    height: int
    trigger_id: str
    capture_status: CaptureStatus = CaptureStatus.SUCCESS
    sequence_number: int = 0
    capture_latency_ms: float | None = None
    error_detail: str | None = None

    @field_validator("frame_id", "camera_id", "trigger_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("frame identifiers must be non-empty")
        return value.strip()

    @field_validator("width", "height")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("frame dimensions must be positive")
        return value

    @model_validator(mode="after")
    def _validate_status(self) -> "CameraFrame":
        if self.capture_status is CaptureStatus.SUCCESS and self.error_detail:
            raise ValueError("SUCCESS frames cannot carry error_detail")
        if self.capture_status is CaptureStatus.FAILED and not self.error_detail:
            raise ValueError("FAILED frames must record error_detail")
        return self

    def image_file(self) -> Path:
        return Path(self.image_path)

    def load_image(self):
        """Decode the frame for the (unchanged) inference pipeline.

        Returns a PIL RGB image; the D3 predictor consumes PIL images, and the
        HTTP inference path can upload the same file bytes.
        """
        if self.capture_status is CaptureStatus.FAILED:
            raise ValueError(f"cannot load a FAILED frame: {self.error_detail}")
        from PIL import Image

        with Image.open(self.image_file()) as handle:
            return handle.convert("RGB")

    def short(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "trigger_id": self.trigger_id,
            "capture_status": self.capture_status.value,
        }
