"""DINOv2 capacity cross-check (D2 = ViT-B/14) regression tests (CPU, no GPU)."""
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

from steel_patchcore.domain_representation import (  # noqa: E402
    D0_AUROC,
    D0_QUARTILES,
    adapted_input_side,
    cosine_1nn_distance,
    expected_patch_grid,
    l2_normalize,
    strip_non_patch_tokens,
)
from steel_patchcore.domain_representation_capacity import (  # noqa: E402
    CAPACITY_GATE,
    D1_AUROC,
    D1_QUARTILES,
    D1_SMALL_DEFECT_SIGNAL,
    D2_REFERENCE,
    capacity_gain,
    capacity_gate_passed,
    capacity_strong_signal,
    d2_reference_sha256,
)
from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.representation import reservoir_from_stream  # noqa: E402
from steel_patchcore.tile import TILE_X0  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
MANIFEST = DS / "representation_diagnostic_manifest.json"
FROZEN_SUBSET_MANIFEST_SHA = "8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075"

D2_WEIGHTS_SHA = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"


# 1. D2 model identity serialization + weights SHA lineage --------------------

def test_d2_reference_identity_frozen():
    assert D2_REFERENCE["model_identifier"] == "dinov2_vitb14"
    assert D2_REFERENCE["arch"] == "vit_base"
    assert D2_REFERENCE["embed_dim"] == 768
    assert D2_REFERENCE["depth"] == 12
    assert D2_REFERENCE["num_heads"] == 12
    assert D2_REFERENCE["patch_size"] == 14
    assert D2_REFERENCE["num_register_tokens"] == 0
    assert D2_REFERENCE["num_cls_tokens"] == 1
    assert D2_REFERENCE["implementation"] == "facebookresearch/dinov2 (official torch.hub)"


def test_d2_weights_sha_lineage():
    assert D2_REFERENCE["weights_sha256"] == D2_WEIGHTS_SHA
    assert D2_REFERENCE["weights_url"] == "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"


def test_d2_reference_serialization_deterministic():
    assert d2_reference_sha256() == d2_reference_sha256()
    assert len(d2_reference_sha256()) == 64


# 2. diagnostic manifest exact reuse + holdout isolation ----------------------

def test_diagnostic_manifest_reused_and_holdout_isolated():
    assert sha256_file(MANIFEST) == FROZEN_SUBSET_MANIFEST_SHA
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["holdout_access_count"] == 0
    assert len(m["train_normal_subset"]) == 1000
    assert len(m["validation_normal_subset"]) == 300
    assert len(m["recovery_dev_anomaly_subset"]) == 1000
    subset = set(m["train_normal_subset"]) | set(m["validation_normal_subset"]) | set(m["recovery_dev_anomaly_subset"])
    splits = json.loads((DS / "split_manifest.json").read_text(encoding="utf-8"))
    recovery = json.loads((DS / "recovery_split_manifest.json").read_text(encoding="utf-8"))
    forbidden = set(splits["splits"]["test_normal"]) | set(recovery["recovery_holdout_anomaly"])
    assert not (subset & forbidden)


# 3. patch-token extraction geometry + CLS exclusion --------------------------

def test_d2_geometry_is_18x18_embed_768():
    assert adapted_input_side(256, 14) == 252
    assert expected_patch_grid(252, 252, 14) == (18, 18)
    assert 18 * 18 == 324
    assert D2_REFERENCE["embed_dim"] == 768


def test_cls_and_register_excluded():
    seq = np.arange(12 * 5, dtype=np.float32).reshape(12, 5)
    patches = strip_non_patch_tokens(seq, num_cls_tokens=1, num_register_tokens=0)
    assert patches.shape == (11, 5)
    assert np.array_equal(patches, seq[1:])
    batched = np.stack([seq, seq + 100])
    assert strip_non_patch_tokens(batched, 1, 0).shape == (2, 11, 5)


# 4. cosine distance semantics ------------------------------------------------

def test_cosine_1nn_distance_is_one_minus_max_similarity():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    bank = np.array([[1.0, 0.0], [0.70710678, 0.70710678]], dtype=np.float64)
    dist = cosine_1nn_distance(emb, bank)
    assert dist[0] == pytest.approx(0.0, abs=1e-6)
    assert dist[1] == pytest.approx(1.0 - np.sqrt(0.5), abs=1e-6)
    # l2 normalizes both sides
    assert np.allclose(np.linalg.norm(l2_normalize(emb), axis=1), 1.0)


# 5. reservoir determinism ----------------------------------------------------

def test_reservoir_is_deterministic_for_seed():
    rng = np.random.default_rng(0)
    stream = [rng.standard_normal((37, 32)).astype(np.float32) for _ in range(5)]
    r1, seen1 = reservoir_from_stream([s.copy() for s in stream], budget=50, seed=42)
    r2, seen2 = reservoir_from_stream([s.copy() for s in stream], budget=50, seed=42)
    assert seen1 == seen2 == 5 * 37
    assert np.array_equal(r1, r2)


# 6. 7-tile A0 aggregation + train-only threshold -----------------------------

def test_seven_tile_a0_and_train_only_threshold():
    assert TILE_X0 == (0, 256, 512, 768, 1024, 1280, 1344)
    tile_scores = [0.31, 0.12, 0.45, 0.29, 0.63, 0.58, 0.60]
    assert max(tile_scores) == 0.63
    train = [0.10, 0.23, 0.05, 0.33]
    assert float(np.max(train)) == 0.33


# 7. capacity Gate semantics --------------------------------------------------

def test_capacity_gate_semantics():
    d0 = D0_AUROC
    assert capacity_gate_passed(d0, 0.703)
    assert not capacity_gate_passed(d0, 0.702)
    assert not capacity_gate_passed(d0, 0.699)
    assert not capacity_gate_passed(float("nan"), 0.80)
    assert CAPACITY_GATE == {"auroc_min": 0.70, "delta_vs_d0": 0.10, "strong_auroc": 0.80, "capacity_delta_vs_d1": 0.03}


def test_capacity_gain_and_strong_signal():
    assert capacity_gain(0.70, D1_AUROC)  # +0.0301
    assert not capacity_gain(0.699, D1_AUROC)  # +0.0291
    assert capacity_strong_signal(0.80)
    assert not capacity_strong_signal(0.79)


# 8. D0 / D1 / D2 frozen comparison -------------------------------------------

def test_d0_d1_frozen_comparison():
    assert D0_AUROC == 0.6029
    assert D1_AUROC == 0.6699
    assert D1_QUARTILES == {"Q1": 0.5843, "Q2": 0.6086, "Q3": 0.6607, "Q4": 0.8261}
    assert D1_SMALL_DEFECT_SIGNAL is True
    assert D0_QUARTILES == {"Q1": 0.4790, "Q2": 0.5305, "Q3": 0.6145, "Q4": 0.7876}
    # D1 already failed the (unchanged) gate; D2 is measured against the same bar.
    assert not capacity_gate_passed(D0_AUROC, D1_AUROC)


# 9. quartile metric ----------------------------------------------------------

def test_normal_vs_quartile_auroc_exists():
    from steel_patchcore.aggregation import normal_vs_quartile_auroc

    normal = np.array([0.1, 0.2, 0.3, 0.4])
    anomaly = np.array([0.9, 0.8, 0.7])
    assert normal_vs_quartile_auroc(normal, anomaly) == 1.0


# 10. runtime isolation (D2 bank + baseline/D1 immutability + resume) ---------

def test_runtime_isolation_baseline_and_d1_bank_immutability_and_resume():
    script = ROOT / "inference-service/scripts/run_steel_domain_representation_capacity_experiment.py"
    spec = importlib.util.spec_from_file_location("_d2_experiment", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXPECTED_FROZEN_BANK_SHA == FROZEN_BANK_SHA
    assert str(module.FRZ).replace("\\", "/").endswith("inference-service/models/steel-patchcore/bank.npz")
    assert "steel-domain-representation-capacity" in module.RUN_ROOT.name
    assert module.D2_DIR.name == "D2-dinov2-b14"
    # D2 bank isolated from D1 bank (different run root / candidate dir)
    d1_script = ROOT / "inference-service/scripts/run_steel_domain_representation_experiment.py"
    d1_spec = importlib.util.spec_from_file_location("_d1_experiment", d1_script)
    d1_module = importlib.util.module_from_spec(d1_spec)
    d1_spec.loader.exec_module(d1_module)
    assert d1_module.D1_DIR != module.D2_DIR
    assert module.D2_DIR != module.FRZ.parent
    # resumable checkpoints
    assert module.BANK_PATH.name == "bank.npz"
    assert module.SCORES_PATH.name == "scores.json"
    assert module.BANK_PATH.parent == module.D2_DIR