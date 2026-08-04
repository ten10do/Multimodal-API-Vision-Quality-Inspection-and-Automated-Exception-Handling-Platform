"""GPU-only predictor tests.

Marked ``gpu`` and excluded from the default pytest run. These tests load the
model on CUDA and are intended for the acceptance machine (RTX 5060) or CI
runners with a GPU.

torch is imported inside the test body only, so normal collection never loads
native libraries (see docs/03 known issues for the intermittent access
violation observed on this host).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.yolo_predictor import YoloPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "model-training/runs/neu-det-yolov8s-baseline-2/weights/best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "model-training/datasets/neu-det-yolo/test/images"

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not (WEIGHTS.exists() and TEST_IMG_DIR.exists()), reason="trained weights or dataset missing")
def test_gpu_forward_contract():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    predictor = YoloPredictor(WEIGHTS, device="cuda:0", allow_cpu_fallback=False)
    image = sorted(TEST_IMG_DIR.glob("*.jpg"))[0]
    result = predictor.predict(image)
    assert result.device == "cuda:0"
    assert result.inference_latency_ms >= 0
    torch.cuda.synchronize()
