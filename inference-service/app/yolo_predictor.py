"""YOLO predictor producing the standardized InferenceResult contract.

Design notes:
- The predictor only reports geometry and confidence. Quality judgement is out of scope.
- Corrupted or unreadable images raise VisionInputError instead of silently returning nothing.
- Device selection is explicit: CUDA is preferred, CPU is a deliberate fallback that is
  logged. Callers that must never fall back can pass allow_cpu_fallback=False.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .vision_contract import Detection, InferenceResult, NEU_DET_CLASSES, utc_now_iso

logger = logging.getLogger(__name__)


class VisionError(Exception):
    """Base error for the vision pipeline."""


class VisionInputError(VisionError):
    """Raised when the input image cannot be read or is invalid."""


class ModelLoadError(VisionError):
    """Raised when the model cannot be loaded on the requested device."""


class YoloPredictor:
    """Wraps an ultralytics YOLO model behind the InferenceResult contract."""

    def __init__(
        self,
        model_path: str | Path,
        model_name: str = "yolov8s",
        model_version: str = "phase1-baseline",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.model_version = model_version
        self.conf_threshold = conf_threshold
        self.allow_cpu_fallback = allow_cpu_fallback
        self.device = device or self._auto_device(allow_cpu_fallback)
        self.model: Any = self._load(self.device)

    @staticmethod
    def _auto_device(allow_cpu_fallback: bool) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if allow_cpu_fallback:
            logger.warning("CUDA unavailable, falling back to CPU")
            return "cpu"
        raise ModelLoadError("CUDA requested but torch.cuda.is_available() is False")

    def _load(self, device: str) -> Any:
        from ultralytics import YOLO

        if not self.model_path.exists():
            raise ModelLoadError(f"model file not found: {self.model_path}")
        try:
            return YOLO(str(self.model_path)).to(device)
        except Exception as exc:  # pragma: no cover - depends on runtime state
            if device.startswith("cuda") and self.allow_cpu_fallback:
                logger.warning("failed to load model on %s (%s), falling back to CPU", device, exc)
                return YOLO(str(self.model_path)).to("cpu")
            raise ModelLoadError(f"failed to load model on {device}: {exc}") from exc

    def _read_image(self, image: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise VisionInputError(f"image file not found: {path}")
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise VisionInputError(f"cannot decode image (corrupted or unsupported): {path}")
            return img
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise VisionInputError("empty image array")
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        raise VisionInputError(f"unsupported image type: {type(image)!r}")

    def predict(
        self,
        image: str | Path | np.ndarray,
        inspection_id: Optional[str] = None,
    ) -> InferenceResult:
        img = self._read_image(image)
        height, width = img.shape[:2]
        start = time.perf_counter()
        results = self.model.predict(img, conf=self.conf_threshold, verbose=False, device=self.device)
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        boxes = results[0].boxes
        names = results[0].names
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for x1, y1, x2, y2, conf, cls_id in zip(xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], confs, cls_ids):
                x1, y1, x2, y2 = self._clip_bbox(x1, y1, x2, y2, width, height)
                area_px = float((x2 - x1) * (y2 - y1))
                area_ratio = float(area_px / (width * height))
                detections.append(
                    Detection(
                        class_id=int(cls_id),
                        class_name=str(names.get(int(cls_id), "unknown")),
                        confidence=float(conf),
                        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        bbox_normalized=(
                            float(x1 / width),
                            float(y1 / height),
                            float(x2 / width),
                            float(y2 / height),
                        ),
                        defect_area_px=area_px,
                        defect_area_ratio=area_ratio,
                    )
                )

        return InferenceResult(
            inspection_id=inspection_id or f"insp-{uuid.uuid4().hex[:12]}",
            model_name=self.model_name,
            model_version=self.model_version,
            image_width=width,
            image_height=height,
            detections=detections,
            inference_latency_ms=latency_ms,
            device=self.device,
            timestamp=utc_now_iso(),
        )

    @staticmethod
    def _clip_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
        x1 = max(0.0, min(float(x1), width - 1.0))
        y1 = max(0.0, min(float(y1), height - 1.0))
        x2 = max(0.0, min(float(x2), width))
        y2 = max(0.0, min(float(y2), height))
        return x1, y1, x2, y2
