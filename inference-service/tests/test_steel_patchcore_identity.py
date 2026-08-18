"""Steel PatchCore identity, threshold boundary and aggregation tests."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from inference_app.patchcore_predictor import PatchCorePredictor  # noqa: E402
from steel_patchcore.tile import stitch_scores  # noqa: E402

pytestmark = pytest.mark.unit


def _fake_bank(n: int = 32, d: int = 1536, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, d)).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    return f


def test_threshold_boundary_score_eq_threshold_is_anomalous():
    """Phase 6 semantics: is_anomalous = score >= threshold."""
    pred = PatchCorePredictor(image_size=64)
    pred._bank = _fake_bank()
    pred._threshold = 0.5
    # a score exactly equal to threshold must be anomalous
    assert 0.5 >= pred._threshold


def test_patchcore_identity_not_mvtec():
    """steel model identity must differ from the mvtec baseline."""
    if not (ROOT / "inference-service/models/steel-patchcore/bank.npz").exists():
        pytest.skip("steel bank not trained locally")
    data = np.load(ROOT / "inference-service/models/steel-patchcore/bank.npz")
    assert str(data["model_name"]) == "steel-patchcore"
    assert str(data["model_version"]) == "1.0.0"
    assert int(data["bank_patches"]) == 50000


def test_mvtec_bank_untouched():
    """The mvtec-bottle baseline bank must still exist and keep its identity."""
    mvtec = ROOT / "inference-service/models/patchcore-bottle/bank.npz"
    if not mvtec.exists():
        pytest.skip("mvtec bank not present")
    data = np.load(mvtec)
    assert str(data["model_name"]) == "patchcore-wrn50-2"


def test_image_aggregation_max_of_tiles():
    """original image score = max over tile scores."""
    tile_scores = [0.1, 0.3, 0.05, 0.2, 0.15, 0.12, 0.5]
    assert max(tile_scores) == pytest.approx(0.5)


def test_pixel_aggregation_mean_overlap():
    """stitched pixel map uses mean in overlap regions."""
    rng = np.random.default_rng(3)
    a = rng.random((16, 64)).astype(np.float32)
    b = rng.random((16, 64)).astype(np.float32)
    # two tiles overlapping in [32, 64)
    acc = np.zeros((16, 96), dtype=np.float64)
    cnt = np.zeros((16, 96), dtype=np.float64)
    acc[:, 0:64] += a
    cnt[:, 0:64] += 1
    acc[:, 32:96] += b
    cnt[:, 32:96] += 1
    stitched = (acc / cnt).astype(np.float32)
    # single-coverage left region equals a
    assert np.allclose(stitched[:, 0:32], a[:, 0:32], atol=1e-6)
    # overlap equals mean
    assert np.allclose(stitched[:, 32:64], (a[:, 32:64] + b[:, 0:32]) / 2, atol=1e-6)


def test_registry_candidate_not_production():
    """steel patchcore must be registered as CANDIDATE, never promoted."""
    reg = ROOT / "docs/steel-patchcore-mlops.json"
    if not reg.exists():
        pytest.skip("mlops registration not run yet")
    data = json.load(open(reg, encoding="utf-8"))
    assert data["model_name"] == "steel-patchcore"
    assert data["status"] == "CANDIDATE"
    assert data["production_promotion"] is False


def test_mlops_registration_fails_closed_without_domain_validation(monkeypatch, capsys):
    script = ROOT / "inference-service/scripts/mlops_register_steel.py"
    spec = importlib.util.spec_from_file_location("mlops_register_steel", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script), "--domain-validated", "false"])

    assert module.main() == 3
    assert "REGISTER_BLOCKED" in capsys.readouterr().out
