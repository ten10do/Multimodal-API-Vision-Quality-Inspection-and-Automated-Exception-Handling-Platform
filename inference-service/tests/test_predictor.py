"""Predictor unit tests.

These tests must not import torch or load any model. They exercise the
predictor logic with a stubbed model that speaks the same box API, keeping
collection and execution stable on all machines.

Real-model behaviour lives in test_predictor_integration.py (marked
integration) and test_predictor_gpu.py (marked gpu).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.yolo_predictor import ModelLoadError, VisionInputError, YoloPredictor
from app.vision_contract import InferenceResult

CLASS_NAMES = {0: "crazing", 1: "inclusion", 2: "patches", 3: "pitted_surface", 4: "rolled-in_scale", 5: "scratches"}


class _ND:
    """Numpy-backed stand-in for a torch tensor exposing .cpu()/.numpy()."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class FakeBoxes:
    def __init__(self, data: list[tuple[float, float, float, float, float, int]]):
        self._rows = data
        if data:
            self.xyxy = _ND(np.array([[r[0], r[1], r[2], r[3]] for r in data], dtype=float))
            self.conf = _ND(np.array([[r[4]] for r in data], dtype=float))
            self.cls = _ND(np.array([[r[5]] for r in data], dtype=int))
        else:
            self.xyxy = _ND(np.empty((0, 4)))
            self.conf = _ND(np.empty((0, 1)))
            self.cls = _ND(np.empty((0, 1)))

    def __len__(self) -> int:
        return len(self._rows)


class FakeResult:
    def __init__(self, boxes: FakeBoxes):
        self.boxes = boxes
        self.names = CLASS_NAMES


class FakeModel:
    def __init__(self, data: list[tuple[float, float, float, float, float, int]]):
        self._data = data

    def predict(self, image, conf=0.25, verbose=False, device=None):
        return [FakeResult(FakeBoxes(self._data))]


def make_predictor(data: list[tuple[float, float, float, float, float, int]]) -> YoloPredictor:
    predictor = YoloPredictor.__new__(YoloPredictor)
    predictor.model = FakeModel(data)
    predictor.model_name = "yolov8s"
    predictor.model_version = "phase1-baseline"
    predictor.conf_threshold = 0.25
    predictor.device = "cpu"
    return predictor


def write_image(tmp_path: Path, name: str, gray_value: int = 128, size: tuple[int, int] = (200, 200)) -> Path:
    img = np.full((*size, 3), gray_value, dtype=np.uint8)
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path


def test_no_defect_output(tmp_path):
    path = write_image(tmp_path, "clean.png")
    result = make_predictor([]).predict(path)
    assert isinstance(result, InferenceResult)
    assert result.detections == []
    assert result.image_width == 200 and result.image_height == 200


def test_single_defect_output(tmp_path):
    path = write_image(tmp_path, "single.png")
    result = make_predictor([(10.0, 20.0, 60.0, 70.0, 0.9, 0)]).predict(path)
    assert len(result.detections) == 1
    d = result.detections[0]
    assert d.class_name == "crazing"
    assert d.confidence == pytest.approx(0.9)
    assert d.bbox_xyxy == (10.0, 20.0, 60.0, 70.0)
    assert d.defect_area_px == pytest.approx(50.0 * 50.0)
    assert d.defect_area_ratio == pytest.approx(2500.0 / 40000.0)


def test_multiple_defect_output(tmp_path):
    path = write_image(tmp_path, "multi.png")
    data = [(5.0, 5.0, 30.0, 30.0, 0.8, 0), (100.0, 100.0, 150.0, 160.0, 0.7, 5)]
    result = make_predictor(data).predict(path)
    assert len(result.detections) == 2
    assert {d.class_name for d in result.detections} == {"crazing", "scratches"}


def test_bbox_clipped_to_image_bounds(tmp_path):
    path = write_image(tmp_path, "clip.png")
    result = make_predictor([(-10.0, -5.0, 300.0, 260.0, 0.6, 1)]).predict(path)
    d = result.detections[0]
    x1, y1, x2, y2 = d.bbox_xyxy
    assert x1 >= 0 and y1 >= 0 and x2 <= 200 and y2 <= 200
    assert x1 < x2 and y1 < y2
    assert d.bbox_normalized == (0.0, 0.0, 1.0, 1.0)


def test_missing_image_file_raises(tmp_path):
    with pytest.raises(VisionInputError):
        make_predictor([]).predict(tmp_path / "does_not_exist.png")


def test_corrupted_image_raises(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not an image at all")
    with pytest.raises(VisionInputError):
        make_predictor([]).predict(path)


def test_empty_array_raises():
    with pytest.raises(VisionInputError):
        make_predictor([]).predict(np.zeros((0, 0, 3), dtype=np.uint8))


def test_model_load_failure_raises():
    with pytest.raises(ModelLoadError):
        YoloPredictor(model_path="no/such/weights.pt", allow_cpu_fallback=False)
