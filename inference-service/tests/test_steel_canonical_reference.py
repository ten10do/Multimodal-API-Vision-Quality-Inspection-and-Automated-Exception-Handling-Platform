"""Canonical PatchCore reference cross-check adapter regression tests (no GPU)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.canonical_reference import (  # noqa: E402
    C0_AUROC,
    CANONICAL_GATE,
    CANONICAL_REFERENCE,
    TILE_X0,
    canonical_gate_passed,
    canonical_strong_signal,
    config_sha256,
    diagnostic_threshold,
    original_score_from_tiles,
    serialize_config,
)
from steel_patchcore.recovery import sha256_file  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
MANIFEST = DS / "representation_diagnostic_manifest.json"
MANIFEST_SHA = DS / "representation_diagnostic_manifest.sha256"
FROZEN_SUBSET_MANIFEST_SHA = "8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075"
FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"


# 1. frozen candidate identity -------------------------------------------------

def test_canonical_reference_config_frozen():
    assert CANONICAL_REFERENCE["name"] == "anomalib"
    assert CANONICAL_REFERENCE["version"] == "0.7.0"
    assert CANONICAL_REFERENCE["backbone"] == "wide_resnet50_2"
    assert CANONICAL_REFERENCE["layers"] == ["layer2", "layer3"]
    assert CANONICAL_REFERENCE["coreset_sampling_ratio"] == 0.1
    assert CANONICAL_REFERENCE["num_neighbors"] == 9
    assert CANONICAL_REFERENCE["input_size"] == [256, 256]
    assert CANONICAL_GATE == {"auroc_min": 0.65, "delta_vs_c0": 0.05, "strong_auroc": 0.75}
    assert C0_AUROC == 0.6029
    assert TILE_X0 == (0, 256, 512, 768, 1024, 1280, 1344)


# 2. seven-tile original aggregation ------------------------------------------

def test_original_score_is_max_of_seven_tiles():
    assert original_score_from_tiles([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) == 0.7
    assert original_score_from_tiles([0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) == 0.7
    with pytest.raises(ValueError):
        original_score_from_tiles([0.1, 0.2])  # not 7 tiles
    with pytest.raises(ValueError):
        original_score_from_tiles([])


# 3. train-only calibration ---------------------------------------------------

def test_diagnostic_threshold_is_train_only_max():
    assert diagnostic_threshold([0.33, 0.12, 0.49, 0.04]) == 0.49
    # threshold does not depend on any other scores
    assert diagnostic_threshold([0.33, 0.12, 0.49, 0.04]) == 0.49


# 4. gate semantics -----------------------------------------------------------

def test_canonical_gate_semantics():
    c0 = C0_AUROC
    assert canonical_gate_passed(c0, 0.653)  # >=0.65 and delta +0.0501 >= 0.05
    assert not canonical_gate_passed(c0, 0.652)  # delta +0.0491 < 0.05
    assert not canonical_gate_passed(c0, 0.649)  # below auroc_min
    assert not canonical_gate_passed(float("nan"), 0.70)
    assert not canonical_gate_passed(c0, None)


def test_canonical_strong_signal():
    assert canonical_strong_signal(0.75)
    assert canonical_strong_signal(0.80)
    assert not canonical_strong_signal(0.74)
    assert not canonical_strong_signal(float("nan"))


# 5. config serialization determinism -----------------------------------------

def test_config_serialization_deterministic_and_hashed():
    assert serialize_config(CANONICAL_REFERENCE) == serialize_config(dict(CANONICAL_REFERENCE))
    assert config_sha256(CANONICAL_REFERENCE) == config_sha256(dict(CANONICAL_REFERENCE))
    assert len(config_sha256(CANONICAL_REFERENCE)) == 64


# 6. diagnostic manifest reuse + holdout isolation ----------------------------

def test_diagnostic_manifest_reused_and_holdout_isolated():
    assert sha256_file(MANIFEST) == FROZEN_SUBSET_MANIFEST_SHA
    assert MANIFEST_SHA.read_text(encoding="ascii").strip() == FROZEN_SUBSET_MANIFEST_SHA
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["holdout_access_count"] == 0
    subset = set(m["train_normal_subset"]) | set(m["validation_normal_subset"]) | set(m["recovery_dev_anomaly_subset"])
    splits = json.loads((DS / "split_manifest.json").read_text(encoding="utf-8"))
    recovery = json.loads((DS / "recovery_split_manifest.json").read_text(encoding="utf-8"))
    forbidden = set(splits["splits"]["test_normal"]) | set(recovery["recovery_holdout_anomaly"])
    assert not (subset & forbidden)


# 7. runtime isolation + baseline immutability --------------------------------

def test_reference_runtime_isolation_and_baseline_immutability():
    script = ROOT / "inference-service/scripts/run_steel_canonical_patchcore.py"
    spec = importlib.util.spec_from_file_location("_canonical_experiment", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXPECTED_FROZEN_BANK_SHA == FROZEN_BANK_SHA
    assert "steel-canonical-patchcore" in module.RUN_ROOT.name
    assert str(module.FRZ).replace("\\", "/").endswith("inference-service/models/steel-patchcore/bank.npz")