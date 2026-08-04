"""Predictor integration tests.

Marked ``integration`` and excluded from the default pytest run
(addopts in pytest.ini). Requires trained weights and the dataset present,
which is the case on the Phase 1 acceptance machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_app.yolo_predictor import YoloPredictor
from inference_app.vision_contract import InferenceResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "model-training/runs/neu-det-yolov8s-baseline-2/weights/best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "model-training/datasets/neu-det-yolo/test/images"

pytestmark = pytest.mark.integration

requires_artifacts = pytest.mark.skipif(
    not (WEIGHTS.exists() and TEST_IMG_DIR.exists()),
    reason="trained weights or dataset missing",
)


@requires_artifacts
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


@requires_artifacts
def test_explicit_cpu_device_works():
    predictor = YoloPredictor(WEIGHTS, device="cpu", allow_cpu_fallback=False)
    image = sorted(TEST_IMG_DIR.glob("*.jpg"))[0]
    result = predictor.predict(image)
    assert result.device == "cpu"
