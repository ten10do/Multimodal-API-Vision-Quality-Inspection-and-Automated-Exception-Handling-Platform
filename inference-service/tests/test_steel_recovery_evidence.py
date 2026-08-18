"""Recovery split, raw evidence, aggregation, and immutability regressions."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from inference_app.patchcore_predictor import PatchCorePredictor  # noqa: E402
from steel_patchcore.recovery import (  # noqa: E402
    CANDIDATE_GRID,
    CAPTURE_ROLES,
    HOLDOUT_ROLES,
    baseline_score,
    build_recovery_split_manifest,
    candidate_score,
    normalize_raw_grid_to_predictor_map,
    raw_distance_grid_from_embeddings,
    stitch_raw_patch_grids,
    validate_recovery_split_manifest,
)


CAPTURE_SCRIPT = ROOT / "inference-service/scripts/capture_steel_recovery_evidence.py"
SPEC = importlib.util.spec_from_file_location("capture_steel_recovery_evidence", CAPTURE_SCRIPT)
assert SPEC and SPEC.loader
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)


def _source_manifest() -> dict:
    return {
        "splits": {
            "train_normal": ["train"],
            "validation_normal": ["validation"],
            "test_normal": ["normal"],
            "test_anomaly": [f"anomaly-{index:04d}" for index in range(6666)],
        }
    }


def test_recovery_split_is_deterministic_disjoint_and_complete():
    source = _source_manifest()
    kwargs = {"created_at": "2026-08-18T00:00:00Z", "seed": 42}
    first = build_recovery_split_manifest(source, "a" * 64, **kwargs)
    second = build_recovery_split_manifest(source, "a" * 64, **kwargs)

    assert first == second
    validate_recovery_split_manifest(first, source, "a" * 64)
    dev = first["recovery_dev_anomaly"]
    holdout = first["recovery_holdout_anomaly"]
    assert len(dev) == len(set(dev)) == 3333
    assert len(holdout) == len(set(holdout)) == 3333
    assert set(dev).isdisjoint(holdout)
    assert set(dev) | set(holdout) == set(source["splits"]["test_anomaly"])


def test_capture_roles_exclude_holdout_roles():
    assert set(CAPTURE_ROLES) == {
        "train_normal",
        "validation_normal",
        "recovery_dev_anomaly",
    }
    assert set(CAPTURE_ROLES).isdisjoint(HOLDOUT_ROLES)


def test_raw_stitch_derives_stride_shape_and_mean_overlap():
    grids = np.stack(
        [np.full((2, 2), 1.0, dtype=np.float32), np.full((2, 2), 3.0, dtype=np.float32)]
    )
    stitched, stride = stitch_raw_patch_grids(
        grids, [0, 2], tile_size=4, original_width=6
    )

    assert stride == 2
    assert stitched.shape == (2, 3)
    assert np.array_equal(stitched, np.asarray([[1.0, 2.0, 3.0]] * 2, dtype=np.float32))


def test_candidate_grid_is_frozen_and_semantics_are_deterministic():
    assert [candidate["id"] for candidate in CANDIDATE_GRID] == [
        "A0", "A1", "A2", "A3", "A4", "A5", "A6"
    ]
    raw = np.arange(28, dtype=np.float32).reshape(7, 2, 2)
    stitched = np.arange(1000, dtype=np.float32).reshape(20, 50)

    assert candidate_score("A0", raw, stitched) == 27.0
    assert candidate_score("A1", raw, stitched) == pytest.approx(
        np.percentile(stitched, 99.0, method="linear")
    )
    assert candidate_score("A2", raw, stitched) == pytest.approx(
        np.percentile(stitched, 99.5, method="linear")
    )
    assert candidate_score("A3", raw, stitched) == pytest.approx(
        np.percentile(stitched, 99.9, method="linear")
    )
    assert candidate_score("A4", raw, stitched) == 999.0
    assert candidate_score("A5", raw, stitched) == pytest.approx(np.mean([995, 996, 997, 998, 999]))
    assert candidate_score("A6", raw, stitched) == pytest.approx(np.mean(np.arange(990, 1000)))


def test_raw_grid_reconstructs_predictor_score_and_normalized_map(monkeypatch):
    embeddings = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    bank = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    predictor = PatchCorePredictor(image_size=16)
    predictor._bank = bank
    monkeypatch.setattr(predictor, "_embed", lambda _image: embeddings)

    predictor_map, predictor_score = predictor.score(Image.new("RGB", (16, 16)))
    raw = raw_distance_grid_from_embeddings(embeddings, bank, (2, 2))

    assert baseline_score(raw[np.newaxis, ...]) == predictor_score
    assert np.array_equal(
        normalize_raw_grid_to_predictor_map(raw, 16), predictor_map
    )


def test_shard_round_trip_is_checkpoint_reconstructible(tmp_path, monkeypatch):
    evidence_root = tmp_path / "raw/recovery-evidence"
    monkeypatch.setattr(CAPTURE, "DS", tmp_path)
    monkeypatch.setattr(CAPTURE, "EVIDENCE_ROOT", evidence_root)
    grids = np.arange(2 * 7 * 2 * 2, dtype=np.float32).reshape(2, 7, 2, 2) / 100.0
    scores = grids.max(axis=(2, 3))
    baseline = grids.max(axis=(1, 2, 3))
    ids = ["one", "two"]
    artifact = CAPTURE.save_shard(
        "train_normal", 0, ids, grids, scores, baseline, 2, (2, 6)
    )
    roles = {"train_normal": ids, "validation_normal": [], "recovery_dev_anomaly": []}
    expected = {
        "train_normal": dict(zip(ids, baseline.tolist())),
        "validation_normal": {},
        "recovery_dev_anomaly": {},
    }

    completed, artifacts, shape, stride, stitched_shape, error = CAPTURE.verify_shards(
        roles, expected
    )

    assert completed["train_normal"] == set(ids)
    assert artifacts == [artifact]
    assert shape == (2, 2)
    assert stride == 2
    assert stitched_shape == (2, 6)
    assert error == 0.0


def test_baseline_metrics_and_v1_identity_remain_immutable():
    metrics = json.loads(
        (ROOT / "docs/steel-patchcore-eval/metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["model"] == "steel-patchcore"
    assert metrics["model_version"] == "1.0.0"
    assert metrics["image_auroc"] == 0.4817
    assert metrics["pixel_auroc_mean_per_image"] == 0.8319
    assert metrics["aup_pro_mean_per_image"] == 0.5838
    assert metrics["operating_point"] == {
        "tp": 0,
        "fp": 1,
        "tn": 590,
        "fn": 6666,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "normal_fpr": 0.0017,
        "anomaly_recall": 0.0,
        "confusion_matrix": [[590, 1], [6666, 0]],
    }
