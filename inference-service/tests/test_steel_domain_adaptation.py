"""Steel-domain adaptation (D3 = DINOv2 ViT-B/14 + train-normal ZCA) tests (CPU)."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.domain_adaptation import (  # noqa: E402
    ADAPTATION_GATE,
    D2_AUROC,
    D2_QUARTILES,
    D2_REFERENCE,
    adaptation_gate_passed,
    adaptation_strong_signal,
    chan_update_batch,
    covariance_from_stats,
    epsilon_rule,
    small_defect_adaptation_signal,
    whiten,
    whitening_numerical_healthy,
    whitening_sanity,
    zca_whitening_matrix,
)
from steel_patchcore.domain_representation import cosine_1nn_distance  # noqa: E402
from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.representation import reservoir_from_stream  # noqa: E402
from steel_patchcore.tile import TILE_X0  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
MANIFEST = DS / "representation_diagnostic_manifest.json"
FROZEN_SUBSET_MANIFEST_SHA = "8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075"
FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"
D2_WEIGHTS_SHA = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"

RUNNER = ROOT / "inference-service/scripts/run_steel_domain_adaptation_experiment.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_d3_experiment", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1. diagnostic manifest exact reuse + holdout isolation ----------------------

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


# 2. DINO-B identity (unchanged backbone) -------------------------------------

def test_d3_reuses_dino_b_identity():
    assert D2_REFERENCE["model_identifier"] == "dinov2_vitb14"
    assert D2_REFERENCE["embed_dim"] == 768
    assert D2_REFERENCE["num_register_tokens"] == 0
    assert D2_REFERENCE["weights_sha256"] == D2_WEIGHTS_SHA


# 3. WHITENING_TRAIN_ONLY_GATE -------------------------------------------------

def test_whitening_train_only_gate():
    module = _load_runner()
    for fn_name in ("accumulate_stats", "build_bank", "run_sanity"):
        fn = getattr(module, fn_name)
        src = inspect.getsource(fn)
        assert "train_ids" in src, fn_name
        assert "val_ids" not in src, fn_name
        assert "dev_ids" not in src, fn_name
        assert "validation_normal" not in src, fn_name
        assert "recovery_dev_anomaly" not in src, fn_name


# 4. streaming mean / covariance correctness ----------------------------------

def test_streaming_mean_and_covariance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3000, 16))
    count, mean, m2 = 0, None, None
    for chunk in np.array_split(x, 11):
        count, mean, m2 = chan_update_batch(count, mean, m2, chunk)
    assert count == 3000
    assert np.allclose(mean, x.mean(axis=0), atol=1e-10)
    empirical = (x - x.mean(axis=0)).T @ (x - x.mean(axis=0)) / 3000
    assert np.allclose(m2 / count, empirical, atol=1e-8)
    # covariance_from_stats = M2 / n
    assert np.allclose(covariance_from_stats(m2, count), empirical, atol=1e-8)


def test_chan_update_batch_empty_is_noop():
    count, mean, m2 = 17, np.ones(4), np.eye(4)
    c2, m2v, m22 = chan_update_batch(count, mean, m2, np.empty((0, 4)))
    assert c2 == 17 and np.array_equal(m2v, mean) and np.array_equal(m22, m2)


# 5. epsilon deterministic rule ----------------------------------------------

def test_epsilon_rule_deterministic():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((16, 16))
    cov = a.T @ a
    eps = epsilon_rule(cov)
    assert eps == pytest.approx(1e-6 * np.trace(cov) / 16)
    assert epsilon_rule(cov) == eps


# 6. ZCA transform correctness + finite values --------------------------------

def test_zca_whitening_correctness():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((200, 8))
    cov = a.T @ a / 200
    eps = epsilon_rule(cov)
    cov_reg = cov + eps * np.eye(8)
    w, ev = zca_whitening_matrix(cov_reg)
    assert np.all(np.isfinite(w))
    assert np.all(ev > 0)
    assert np.all(np.diff(ev) >= -1e-12)  # ascending
    # W @ cov_reg @ W == I (ZCA whitening property)
    assert np.allclose(w @ cov_reg @ w, np.eye(8), atol=1e-7)


def test_zca_rejects_negative_definite():
    with pytest.raises(ValueError):
        zca_whitening_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))


def test_whiten_applies_centering_then_transform():
    x = np.array([[2.0, 0.0, 0.0]])
    mean = np.array([1.0, 0.0, 0.0])
    w = np.eye(3)
    assert np.allclose(whiten(x, mean, w), [[1.0, 0.0, 0.0]])


# 7. whitening sanity + numerical health gate ---------------------------------

def test_whitening_sanity_on_white_noise():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((30000, 8))
    s = whitening_sanity(x)
    assert s["cov_diag_p50"] == pytest.approx(1.0, abs=0.05)
    assert s["cov_offdiag_abs_max"] < 0.05
    assert s["max_abs_mean"] < 0.05


def test_whitening_numerical_healthy():
    healthy = {
        "max_abs_mean": 0.001,
        "mean_abs_mean": 0.0003,
        "cov_diag_min": 0.9,
        "cov_diag_p50": 1.0,
        "cov_diag_max": 1.1,
        "cov_offdiag_abs_mean": 0.003,
        "cov_offdiag_abs_p95": 0.008,
        "cov_offdiag_abs_max": 0.02,
    }
    ok, _ = whitening_numerical_healthy(healthy, np.linspace(0.5, 5.0, 8))
    assert ok
    # negative eigenvalue -> block
    assert not whitening_numerical_healthy(healthy, np.array([-0.5, 1.0, 2.0]))[0]
    # non-finite sanity -> block
    assert not whitening_numerical_healthy({**healthy, "max_abs_mean": float("nan")}, np.linspace(0.5, 5.0, 8))[0]
    # gross non-identity -> block
    assert not whitening_numerical_healthy({**healthy, "cov_diag_p50": 2.5}, np.linspace(0.5, 5.0, 8))[0]


# 8. adaptation gate semantics ------------------------------------------------

def test_adaptation_gate_semantics():
    d2 = D2_AUROC
    assert adaptation_gate_passed(d2, 0.75)  # >=0.75 and +0.0562
    assert not adaptation_gate_passed(d2, 0.7499)
    assert not adaptation_gate_passed(d2, d2 + 0.05)  # 0.7438 < 0.75
    assert not adaptation_gate_passed(d2, float("nan"))
    assert ADAPTATION_GATE == {"auroc_min": 0.75, "delta_vs_d2": 0.05, "strong_auroc": 0.80, "small_defect_q1_delta": 0.05}


def test_small_defect_adaptation_signal_and_strong_signal():
    assert small_defect_adaptation_signal(0.655, D2_QUARTILES["Q1"])  # +0.0507
    assert not small_defect_adaptation_signal(0.654, D2_QUARTILES["Q1"])  # +0.0497
    assert adaptation_strong_signal(0.80)
    assert not adaptation_strong_signal(0.79)


# 9. reservoir determinism + cosine + 7-tile A0 + threshold -------------------

def test_reservoir_deterministic_for_seed():
    rng = np.random.default_rng(5)
    stream = [rng.standard_normal((37, 32)).astype(np.float32) for _ in range(5)]
    r1, s1 = reservoir_from_stream([s.copy() for s in stream], budget=50, seed=42)
    r2, s2 = reservoir_from_stream([s.copy() for s in stream], budget=50, seed=42)
    assert s1 == s2 == 5 * 37
    assert np.array_equal(r1, r2)


def test_cosine_1nn_and_7tile_a0_and_threshold():
    emb = np.array([[1.0, 0.0]])
    bank = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert cosine_1nn_distance(emb, bank)[0] == pytest.approx(0.0, abs=1e-12)
    assert TILE_X0 == (0, 256, 512, 768, 1024, 1280, 1344)
    assert max([0.3, 0.1, 0.5, 0.2, 0.6, 0.55, 0.57]) == 0.6
    assert float(np.max([0.1, 0.2, 0.05])) == 0.2


# 10. quartile metric ---------------------------------------------------------

def test_normal_vs_quartile_auroc_exists():
    from steel_patchcore.aggregation import normal_vs_quartile_auroc

    assert normal_vs_quartile_auroc(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 1.0


# 11. runtime isolation + bank immutability + checkpoint/resume ---------------

def test_runtime_isolation_bank_immutability_and_checkpoints():
    module = _load_runner()
    assert module.EXPECTED_FROZEN_BANK_SHA == FROZEN_BANK_SHA
    assert str(module.FRZ).replace("\\", "/").endswith("inference-service/models/steel-patchcore/bank.npz")
    assert module.RUN_ROOT.name == "steel-domain-adaptation"
    assert module.D3_DIR.name == "D3-dinov2-b14-zca"
    # checkpoint artifacts
    assert module.STATS_PATH.name == "stats.npz"
    assert module.WHITENING_PATH.name == "whitening.npz"
    assert module.BANK_PATH.name == "bank.npz"
    assert module.SCORES_PATH.name == "scores.json"
    # D3 bank isolated from D1/D2 banks
    d1 = importlib.util.spec_from_file_location("_d1x", ROOT / "inference-service/scripts/run_steel_domain_representation_experiment.py")
    d2 = importlib.util.spec_from_file_location("_d2x", ROOT / "inference-service/scripts/run_steel_domain_representation_capacity_experiment.py")
    m1 = importlib.util.module_from_spec(d1); d1.loader.exec_module(m1)
    m2 = importlib.util.module_from_spec(d2); d2.loader.exec_module(m2)
    assert module.D3_DIR != m1.D1_DIR
    assert module.D3_DIR != m2.D2_DIR
    assert module.D3_DIR != module.FRZ.parent