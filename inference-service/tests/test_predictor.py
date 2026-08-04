"""Predictor behavior tests.

Unit tests use a stubbed model so they stay deterministic and fast. The real
model integration test only runs when trained weights and dataset are present
(Phase 1 acceptance machine), and is skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import torch  # noqa: E402  imported first to stabilize native DLL load order on Windows
import numpy as np
import cv2
import pytest

from app.yolo_predictor import ModelLoadError, VisionInputError, YoloPredictor
from app.vision_contract import InferenceResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "model-training/runs/neu-det-yolov8s-baseline-2/weights/best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "model-training/datasets/neu-det-yolo/test/images"


class FakeBoxes:
    def __init__(self, data: list[tuple[float, float, float, float, float, int]]):
        self._rows = data
        if data:
            self.xyxy = torch.tensor([[r[0], r[1], r[2], r[3]] for r in data])
            self.conf = torch.tensor([[r[4]] for r in data])
            self.cls = torch.tensor([[r[5]] for r in data])
        else:
            self.xyxy = torch.empty(0, 4)
            self.conf = torch.empty(0, 1)
            self.cls = torch.empty(0, 1)

    def __len__(self):
        return len(self._rows)


class FakeResult:
    def __init__(self, boxes: FakeBoxes, names: dict):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, data: list[tuple[float, float, float, float, float, int]]):
        self._data = data
        self.names = {i: n for i, n in enumerate(["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"])}

    def predict(self, image, conf=0.25, verbose=False, device=None):
        return [FakeResult(FakeBoxes(self._data), self.names)]


def make_predictor(data: list[tuple[float, float, float, float, float, int]]) -> YoloPredictor:
    predictor = YoloPredictor.__new__(YoloPredictor)  # skip __init__
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


def test_explicit_cpu_device_works(tmp_path):
    # When CUDA exists this exercises the CPU path explicitly; when it does not,
    # the fallback flag already produced a CPU device in auto mode.
    if not (WEIGHTS.exists() and TEST_IMG_DIR.exists()):
        pytest.skip("trained weights or dataset missing")
    predictor = YoloPredictor(WEIGHTS, device="cpu", allow_cpu_fallback=False)
    image = sorted(TEST_IMG_DIR.glob("*.jpg"))[0]
    result = predictor.predict(image)
    assert result.device == "cpu"


@pytest.mark.skipif(not (WEIGHTS.exists() and TEST_IMG_DIR.exists()), reason="trained weights or dataset missing")
def test_real_model_end_to_end_contract():
    predictor = YoloPredictor(WEIGHTS, device=None)
    image = sorted(TEST_IMG_DIR.glob("*.jpg"))[0]
    result = predictor.predict(image)
    assert isinstance(result, InferenceResult)
    assert result.device.startswith("cuda") or result.device == "cpu"
    assert result.image_width == 200 and result.image_height == 200
    for d in result.detections:
        assert 0.0 <= d.confidence <= 1.0
        assert 0.0 <= d.defect_area_ratio <= 1.0
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 200
