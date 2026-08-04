"""Standardized vision inference contract (shared package).

Phase 1C decision: the vision pipeline emits objective detection facts only.
severity, PASS / REVIEW / FAIL and any quality judgement are produced
downstream by the Quality Rule Engine and MUST NOT appear in this layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

NEU_DET_CLASSES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)


class Detection(BaseModel):
    """A single detected defect, expressed as objective geometry."""

    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(ge=0, description="Class index, matches data.yaml order")
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: Tuple[float, float, float, float] = Field(
        description="Pixel coordinates (x1, y1, x2, y2), x1 < x2 and y1 < y2"
    )
    bbox_normalized: Tuple[float, float, float, float] = Field(
        description="Normalized coordinates in [0, 1] relative to image size"
    )
    defect_area_px: float = Field(ge=0.0, description="Bounding box area in pixels")
    defect_area_ratio: float = Field(ge=0.0, le=1.0, description="Bounding box area divided by image area")

    @field_validator("bbox_xyxy")
    @classmethod
    def _bbox_is_valid(cls, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"bbox must satisfy x1 < x2 and y1 < y2, got {bbox}")
        return bbox

    @field_validator("bbox_normalized")
    @classmethod
    def _bbox_normalized_in_range(cls, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        if not all(0.0 <= v <= 1.0 for v in bbox):
            raise ValueError(f"bbox_normalized must be in [0, 1], got {bbox}")
        return bbox


class InferenceResult(BaseModel):
    """Objective output of the vision pipeline for one image."""

    model_config = ConfigDict(extra="forbid")

    inspection_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    detections: list[Detection] = Field(default_factory=list)
    inference_latency_ms: float = Field(ge=0.0)
    device: str = Field(min_length=1, description="e.g. 'cuda:0' or 'cpu'")
    timestamp: str = Field(description="ISO 8601 UTC timestamp")

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = ["Detection", "InferenceResult", "NEU_DET_CLASSES", "utc_now_iso"]
