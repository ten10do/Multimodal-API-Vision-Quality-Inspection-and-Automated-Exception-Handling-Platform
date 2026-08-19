"""Representation investigation primitives regression tests (no GPU, no holdout)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.representation import (  # noqa: E402
    ANOMALY_PER_QUARTILE,
    FEATURE_LAYER_CANDIDATES,
    FEATURE_LAYER_GATE,
    NORMALIZATION_CANDIDATES,
    NORMALIZATION_GATE,
    REPRESENTATION_SEED,
    SUBSET_SIZES,
    build_representation_subset_manifest,
    feature_layer_gate_passed,
    normalization_gate_passed,
    reservoir_from_stream,
)


def test_candidate_definitions_are_frozen():
    assert [c["id"] for c in FEATURE_LAYER_CANDIDATES] == ["R0", "R1", "R2"]
    assert [c["dim"] for c in FEATURE_LAYER_CANDIDATES] == [1536, 512, 1024]
    assert [c["id"] for c in NORMALIZATION_CANDIDATES] == ["N0", "N1", "N2"]
    assert FEATURE_LAYER_GATE == {"auroc_min": 0.60, "delta_vs_r0": 0.10}
    assert NORMALIZATION_GATE == {"auroc_min": 0.60, "delta_vs_n0": 0.10}
    assert SUBSET_SIZES == {"train_normal": 1000, "validation_normal": 300, "recovery_dev_anomaly": 1000}
    assert ANOMALY_PER_QUARTILE == 250
    assert REPRESENTATION_SEED == 42


def test_reservoir_sampler_is_deterministic_and_bounded():
    stream = [np.arange(20, dtype=np.float32).reshape(10, 2) + i for i in range(3)]
    r1, seen1 = reservoir_from_stream(stream, budget=15, seed=42)
    r2, seen2 = reservoir_from_stream(stream, budget=15, seed=42)
    assert seen1 == seen2 == 30
    assert r1.shape == (15, 2)
    assert np.array_equal(r1, r2)


def test_reservoir_matches_algorithm_r_reference():
    rng_seed = 42
    data = np.arange(60, dtype=np.float32).reshape(20, 3)
    budget = 8
    got, seen = reservoir_from_stream([data], budget=budget, seed=rng_seed)

    # reference Algorithm R
    rng = np.random.default_rng(rng_seed)
    ref = np.zeros((budget, 3), dtype=np.float32)
    n = 0
    for row in data:
        n += 1
        if n <= budget:
            ref[n - 1] = row
        else:
            j = int(rng.integers(0, n))
            if j < budget:
                ref[j] = row
    assert seen == 20
    assert np.array_equal(got, ref)


def test_subset_manifest_deterministic_stratified_and_holdout_isolated():
    source = {
        "train_normal": [f"tn{i:05d}" for i in range(5000)],
        "validation_normal": [f"vn{i:04d}" for i in range(500)],
        "test_normal": [f"tst{i:04d}" for i in range(591)],
        "test_anomaly": [],  # unused by builder
    }
    dev = [f"ad{i:05d}" for i in range(3500)]
    holdout = [f"ah{i:05d}" for i in range(3333)]
    ratios = {i: v for i, v in zip(dev, np.linspace(0.0, 1.0, len(dev)))}

    kwargs = {
        "source_splits": source,
        "recovery_dev_anomaly": dev,
        "recovery_holdout_anomaly": holdout,
        "test_normal": source["test_normal"],
        "area_ratios": ratios,
        "source_split_sha256": "a" * 64,
        "recovery_split_sha256": "b" * 64,
        "created_at": "2026-08-19T00:00:00Z",
    }
    first = build_representation_subset_manifest(**kwargs)
    second = build_representation_subset_manifest(**kwargs)

    assert first == second
    assert first["train_normal_subset"] == source["train_normal"][:1000]
    assert first["validation_normal_subset"] == source["validation_normal"][:300]
    assert len(first["recovery_dev_anomaly_subset"]) == 1000
    assert first["anomaly_quartile_counts"] == {1: 250, 2: 250, 3: 250, 4: 250}

    all_ids = (
        first["train_normal_subset"]
        + first["validation_normal_subset"]
        + first["recovery_dev_anomaly_subset"]
    )
    assert len(all_ids) == len(set(all_ids))
    holdout_set = set(source["test_normal"]) | set(holdout)
    assert not (set(all_ids) & holdout_set)
    assert first["holdout_access_count"] == 0
    assert "manifest_payload_sha256" in first


def test_subset_manifest_rejects_holdout_contamination():
    source = {
        "train_normal": [f"tn{i:05d}" for i in range(5000)],
        "validation_normal": [f"vn{i:04d}" for i in range(500)],
        "test_normal": [],
        "test_anomaly": [],
    }
    dev = [f"ad{i:05d}" for i in range(3500)]
    ratios = {i: v for i, v in zip(dev, np.linspace(0.0, 1.0, len(dev)))}
    with pytest.raises(ValueError):
        build_representation_subset_manifest(
            source_splits=source,
            recovery_dev_anomaly=dev,
            recovery_holdout_anomaly=dev[:500],  # overlaps dev
            test_normal=[],
            area_ratios=ratios,
            source_split_sha256="a" * 64,
            recovery_split_sha256="b" * 64,
            created_at="2026-08-19T00:00:00Z",
        )


def test_feature_layer_gate_semantics():
    assert feature_layer_gate_passed(0.50, 0.61)
    assert not feature_layer_gate_passed(0.50, 0.599)
    assert not feature_layer_gate_passed(0.50, 0.59)
    assert not feature_layer_gate_passed(0.70, 0.71)
    assert not feature_layer_gate_passed(float("nan"), 0.7)
    assert normalization_gate_passed(0.50, 0.61)
    assert not normalization_gate_passed(0.50, 0.59)


def test_feature_view_shapes_and_row_major_flatten(tmp_path):
    torch = pytest.importorskip("torch")
    script = ROOT / "inference-service/scripts/run_steel_representation_experiment.py"
    spec = importlib.util.spec_from_file_location("_repr_experiment", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    h2 = torch.randn(1, 512, 8, 8)
    h3_up = torch.randn(1, 1024, 8, 8)
    views = module.feature_views(h2, h3_up)
    assert views["R0"].shape == (64, 1536)
    assert views["R1"].shape == (64, 512)
    assert views["R2"].shape == (64, 1024)
    for key, tensor in views.items():
        norms = tensor.norm(dim=1).cpu().numpy()
        assert np.allclose(norms, 1.0, atol=1e-3)