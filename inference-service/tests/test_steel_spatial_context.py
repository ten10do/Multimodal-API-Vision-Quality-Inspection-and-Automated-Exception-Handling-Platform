"""Spatial scale & local context regression tests (no GPU, no holdout)."""
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

from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.spatial_context import (  # noqa: E402
    BANK_BUDGET,
    PATCH_SCALE_CANDIDATES,
    PATCH_SCALE_GATE,
    SPATIAL_CONTEXT_CANDIDATES,
    SPATIAL_CONTEXT_GATE,
    SPATIAL_SEED,
    average_pool_same,
    context_embed,
    patch_scale_gate_passed,
    select_best_spatial_candidate,
    spatial_context_gate_passed,
)

DS = ROOT / "model-training/datasets/severstal-steel"
MANIFEST = DS / "representation_diagnostic_manifest.json"
MANIFEST_SHA = DS / "representation_diagnostic_manifest.sha256"
FROZEN_SUBSET_MANIFEST_SHA = "8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075"
FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"


# 1. protocol immutability ----------------------------------------------------

def test_spatial_and_patch_candidate_definitions_frozen():
    assert [c["id"] for c in SPATIAL_CONTEXT_CANDIDATES] == ["S0", "S1", "S2"]
    assert [c["context"] for c in SPATIAL_CONTEXT_CANDIDATES] == [None, 3, 5]
    assert [c["dim"] for c in SPATIAL_CONTEXT_CANDIDATES] == [1024, 1024, 1024]
    assert [c["id"] for c in PATCH_SCALE_CANDIDATES] == ["P0", "P1"]
    assert PATCH_SCALE_CANDIDATES[1]["context"] == 3
    assert PATCH_SCALE_CANDIDATES[1]["dim"] == 512
    assert SPATIAL_CONTEXT_GATE == {"auroc_min": 0.65, "delta_vs_s0": 0.10}
    assert PATCH_SCALE_GATE == {"auroc_min": 0.60, "delta_vs_r1": 0.10, "q1_delta_vs_r1": 0.10}
    assert BANK_BUDGET == 50000
    assert SPATIAL_SEED == 42


# 2. pooling semantics --------------------------------------------------------

def test_average_pool_same_preserves_grid():
    torch = pytest.importorskip("torch")
    x = torch.rand(1, 4, 32, 32)
    for k in (3, 5):
        assert average_pool_same(x, k).shape == (1, 4, 32, 32)


def test_3x3_pooling_mean_and_padding_semantics():
    torch = pytest.importorskip("torch")
    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 1, 1] = 1.0
    y = average_pool_same(x, 3)
    # center cell: full 3x3 window -> 1/9
    assert abs(float(y[0, 0, 1, 1]) - 1 / 9) < 1e-5
    # corner cell: only 4 valid neighbor cells (count_include_pad=False) -> 1/4
    assert abs(float(y[0, 0, 0, 0]) - 1 / 4) < 1e-5


def test_5x5_pooling_mean_and_padding_semantics():
    torch = pytest.importorskip("torch")
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 2, 2] = 1.0
    y = average_pool_same(x, 5)
    assert abs(float(y[0, 0, 2, 2]) - 1 / 25) < 1e-5
    assert abs(float(y[0, 0, 0, 0]) - 1 / 9) < 1e-5


# 3. normalization order ------------------------------------------------------

def test_context_embed_is_pool_then_l2():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    x = torch.rand(1, 8, 6, 6)
    emb = context_embed(x, 3)
    assert emb.shape == (36, 8)
    ref = F.normalize(average_pool_same(x, 3).permute(0, 2, 3, 1).reshape(1, 36, 8)[0], dim=1)
    assert torch.allclose(emb, ref)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


# 4. gate semantics -----------------------------------------------------------

def test_spatial_context_gate_semantics():
    assert spatial_context_gate_passed(0.53, 0.65)
    assert not spatial_context_gate_passed(0.53, 0.64)  # below auroc_min
    assert not spatial_context_gate_passed(0.60, 0.65)  # delta only +0.05
    assert not spatial_context_gate_passed(float("nan"), 0.70)
    assert not spatial_context_gate_passed(0.53, None)


def test_patch_scale_gate_semantics():
    assert patch_scale_gate_passed(0.42, 0.35, 0.60, 0.45)
    assert not patch_scale_gate_passed(0.42, 0.35, 0.59, 0.45)  # auroc < 0.60
    assert not patch_scale_gate_passed(0.42, 0.35, 0.60, 0.44)  # q1 delta +0.09
    assert not patch_scale_gate_passed(0.42, 0.35, 0.51, 0.45)  # auroc delta +0.09


# 5. candidate selection determinism -----------------------------------------

def test_candidate_selection_determinism_tiebreak():
    results = {
        "S1": {"image_auroc": 0.660, "q1_auroc": 0.50, "q2_auroc": 0.50, "gate_passed": True},
        "S2": {"image_auroc": 0.665, "q1_auroc": 0.49, "q2_auroc": 0.49, "gate_passed": True},
    }
    # diff 0.005 < 0.01 -> tie-break by Q1+Q2 mean: S1 (0.50) > S2 (0.49)
    assert select_best_spatial_candidate(results) == "S1"


def test_candidate_selection_no_pass_and_clear_winner():
    results = {
        "S1": {"image_auroc": 0.64, "q1_auroc": 0.5, "q2_auroc": 0.5, "gate_passed": False},
        "S2": {"image_auroc": 0.66, "q1_auroc": 0.5, "q2_auroc": 0.5, "gate_passed": True},
    }
    assert select_best_spatial_candidate(results) == "S2"
    assert select_best_spatial_candidate({"S1": {"image_auroc": 0.64, "q1_auroc": 0.5, "q2_auroc": 0.5, "gate_passed": False}}) is None


# 6. diagnostic manifest reuse (no resample) ----------------------------------

def test_diagnostic_manifest_reused_and_frozen():
    assert sha256_file(MANIFEST) == FROZEN_SUBSET_MANIFEST_SHA
    assert MANIFEST_SHA.read_text(encoding="ascii").strip() == FROZEN_SUBSET_MANIFEST_SHA
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["subset_counts"] == {"train_normal": 1000, "validation_normal": 300, "recovery_dev_anomaly": 1000}
    assert m["anomaly_quartile_counts"] == {"1": 250, "2": 250, "3": 250, "4": 250}
    assert m["holdout_access_count"] == 0


# 7. holdout isolation --------------------------------------------------------

def test_subset_has_no_holdout_contamination():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    subset = set(m["train_normal_subset"]) | set(m["validation_normal_subset"]) | set(m["recovery_dev_anomaly_subset"])
    splits = json.loads((DS / "split_manifest.json").read_text(encoding="utf-8"))
    recovery = json.loads((DS / "recovery_split_manifest.json").read_text(encoding="utf-8"))
    forbidden = set(splits["splits"]["test_normal"]) | set(recovery["recovery_holdout_anomaly"])
    assert not (subset & forbidden)


# 8. experimental bank isolation + baseline immutability ----------------------

def test_experiment_bank_isolation_and_baseline_immutability():
    script = ROOT / "inference-service/scripts/run_steel_spatial_context_experiment.py"
    spec = importlib.util.spec_from_file_location("_spatial_experiment", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXPECTED_FROZEN_BANK_SHA == FROZEN_BANK_SHA
    assert "steel-spatial-context" in module.RUN_ROOT.name
    # experimental banks live under the gitignored run root, not the model dir
    assert str(module.FRZ).replace("\\", "/").endswith("inference-service/models/steel-patchcore/bank.npz")
    assert module.FRZ != (module.RUN_ROOT / "anything")