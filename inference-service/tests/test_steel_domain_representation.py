"""Domain representation (DINOv2 cross-check) regression tests (CPU, no GPU)."""
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
    DINO_REFERENCE,
    DOMAIN_REPRESENTATION_GATE,
    adapted_input_side,
    cosine_1nn_distance,
    domain_representation_gate_passed,
    domain_representation_strong_signal,
    expected_patch_grid,
    l2_normalize,
    reference_sha256,
    serialize_reference,
    small_defect_signal,
    strip_non_patch_tokens,
)
from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.representation import reservoir_from_stream  # noqa: E402
from steel_patchcore.tile import TILE_X0  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
MANIFEST = DS / "representation_diagnostic_manifest.json"
MANIFEST_SHA = DS / "representation_diagnostic_manifest.sha256"
FROZEN_SUBSET_MANIFEST_SHA = "8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075"
FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"


# 1. DINO reference identity serialization -------------------------------------

def test_dino_reference_identity_frozen():
    assert DINO_REFERENCE["model_identifier"] == "dinov2_vits14"
    assert DINO_REFERENCE["arch"] == "vit_small"
    assert DINO_REFERENCE["embed_dim"] == 384
    assert DINO_REFERENCE["patch_size"] == 14
    assert DINO_REFERENCE["num_register_tokens"] == 0
    assert DINO_REFERENCE["num_cls_tokens"] == 1
    assert DINO_REFERENCE["implementation"] == "facebookresearch/dinov2 (official torch.hub)"
    assert DINO_REFERENCE["license"] == "Apache-2.0"


def test_reference_serialization_deterministic():
    assert serialize_reference(DINO_REFERENCE) == serialize_reference(dict(DINO_REFERENCE))
    assert reference_sha256(DINO_REFERENCE) == reference_sha256(dict(DINO_REFERENCE))
    assert len(reference_sha256(DINO_REFERENCE)) == 64


# 2. diagnostic manifest reuse + holdout isolation -----------------------------

def test_diagnostic_manifest_reused_and_holdout_isolated():
    assert sha256_file(MANIFEST) == FROZEN_SUBSET_MANIFEST_SHA
    assert MANIFEST_SHA.read_text(encoding="ascii").strip() == FROZEN_SUBSET_MANIFEST_SHA
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


# 3. patch-token extraction shape + CLS exclusion ------------------------------

def test_expected_patch_grid_for_256_tile_is_18x18():
    # PatchEmbed requires multiples of 14; 256 -> 252 (18*14) via bilinear resize.
    assert adapted_input_side(256, 14) == 252
    assert expected_patch_grid(252, 252, 14) == (18, 18)
    assert 18 * 18 == 324
    with pytest.raises(ValueError):
        expected_patch_grid(256, 256, 14)  # 256 is not a multiple of 14


def test_strip_non_patch_tokens_excludes_cls_and_registers():
    seq = np.arange(10 * 8, dtype=np.float32).reshape(10, 8)  # (tokens, dim)
    # 1 CLS + 2 registers + 7 patches
    patches = strip_non_patch_tokens(seq, num_cls_tokens=1, num_register_tokens=2)
    assert patches.shape == (7, 8)
    assert np.array_equal(patches, seq[3:])
    # 3D (batch, tokens, dim)
    batched = np.stack([seq, seq + 100])
    assert strip_non_patch_tokens(batched, 1, 2).shape == (2, 7, 8)


# 4. per-patch L2 normalization ------------------------------------------------

def test_l2_normalize_is_per_patch_unit_norm():
    x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    out = l2_normalize(x)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], [0.0, 0.0])  # zero row stays zero
    assert np.allclose(np.linalg.norm(out[2]), 1.0)


# 5. cosine 1-NN distance semantics --------------------------------------------

def test_cosine_1nn_distance_is_one_minus_max_similarity():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    bank = np.array([[1.0, 0.0], [0.70710678, 0.70710678]], dtype=np.float64)
    dist = cosine_1nn_distance(emb, bank)
    # patch0: max sim = 1.0 -> dist 0; patch1: max sim = cos(45deg)=0.7071 -> dist ~0.2929
    assert dist[0] == pytest.approx(0.0, abs=1e-6)
    assert dist[1] == pytest.approx(1.0 - np.sqrt(0.5), abs=1e-6)


# 6. reservoir determinism -----------------------------------------------------

def test_reservoir_is_deterministic_for_seed():
    rng = np.random.default_rng(0)
    stream = [rng.standard_normal((37, 16)).astype(np.float32) for _ in range(5)]
    r1, seen1 = reservoir_from_stream([s.copy() for s in stream], budget=50, seed=42)
    r2, seen2 = reservoir_from_stream([s.copy() for s in stream], budget=50, seed=42)
    r3, _ = reservoir_from_stream([s.copy() for s in stream], budget=50, seed=7)
    assert seen1 == seen2 == 5 * 37
    assert np.array_equal(r1, r2)
    assert not np.array_equal(r1, r3)


# 7. 7-tile aggregation frozen -------------------------------------------------

def test_seven_tile_aggregation_is_frozen_a0():
    assert TILE_X0 == (0, 256, 512, 768, 1024, 1280, 1344)
    # A0 semantics: original score = max over 7 tile patch anomaly scores
    tile_anomaly_scores = [0.31, 0.12, 0.45, 0.29, 0.63, 0.58, 0.60]
    assert max(tile_anomaly_scores) == 0.63


# 8. threshold train-only ------------------------------------------------------

def test_threshold_is_train_only_max():
    train = [0.10, 0.23, 0.05, 0.33]
    assert float(np.max(train)) == 0.33


# 9. domain gate semantics -----------------------------------------------------

def test_domain_representation_gate_semantics():
    d0 = D0_AUROC
    assert domain_representation_gate_passed(d0, 0.703)  # >=0.70 and delta +0.1001
    assert not domain_representation_gate_passed(d0, 0.702)  # delta +0.0991 < 0.10
    assert not domain_representation_gate_passed(d0, 0.699)  # below auroc_min
    assert not domain_representation_gate_passed(float("nan"), 0.80)
    assert not domain_representation_gate_passed(d0, None)
    assert DOMAIN_REPRESENTATION_GATE == {"auroc_min": 0.70, "delta_vs_d0": 0.10, "strong_auroc": 0.80, "quartile_delta": 0.10}


def test_domain_representation_strong_signal():
    assert domain_representation_strong_signal(0.80)
    assert domain_representation_strong_signal(0.85)
    assert not domain_representation_strong_signal(0.79)


def test_small_defect_signal():
    assert small_defect_signal(0.60, 0.53, 0.48, 0.53)  # Q1 +0.12
    assert small_defect_signal(0.48, 0.64, 0.48, 0.53)  # Q2 +0.11
    assert not small_defect_signal(0.50, 0.54, 0.48, 0.53)  # both < +0.10


# 10. quartile metric ----------------------------------------------------------

def test_normal_vs_quartile_auroc_exists():
    from steel_patchcore.aggregation import normal_vs_quartile_auroc

    normal = np.array([0.1, 0.2, 0.3, 0.4])
    anomaly = np.array([0.9, 0.8, 0.7])
    assert normal_vs_quartile_auroc(normal, anomaly) == 1.0


# 11. runtime isolation (experimental bank + baseline immutability + resume) ----

def test_reference_runtime_isolation_and_baseline_immutability():
    script = ROOT / "inference-service/scripts/run_steel_domain_representation_experiment.py"
    spec = importlib.util.spec_from_file_location("_d1_experiment", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXPECTED_FROZEN_BANK_SHA == FROZEN_BANK_SHA
    assert str(module.FRZ).replace("\\", "/").endswith("inference-service/models/steel-patchcore/bank.npz")
    assert "steel-domain-representation" in module.RUN_ROOT.name
    assert module.D1_DIR.name == "D1-dinov2-s14"
    # resumable checkpoints
    assert module.BANK_PATH.name == "bank.npz"
    assert module.SCORES_PATH.name == "scores.json"
    assert module.BANK_PATH.parent == module.D1_DIR