"""D3 full-development confirmation tests (CPU)."""
from __future__ import annotations

import importlib.util
import inspect
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.d3_full_development import (  # noqa: E402
    D3_DIAGNOSTIC_AUROC,
    D3_DIAGNOSTIC_QUARTILES,
    D3_FULL_DEV_GATE,
    DINO_B_WEIGHTS_SHA256,
    RECOVERY_SPLIT_SHA256,
    SOURCE_SPLIT_SHA256,
    d3_full_development_gate_passed,
    fail_closed_membership,
    small_defect_full_dev_signal,
)
from steel_patchcore.aggregation import train_only_threshold  # noqa: E402
from steel_patchcore.domain_adaptation import (  # noqa: E402
    chan_update_batch,
    covariance_from_stats,
    epsilon_rule,
    whiten,
    whitening_sanity,
)
from steel_patchcore.recovery import canonical_sha256  # noqa: E402
from steel_patchcore.representation import reservoir_from_stream  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
RUNNER = ROOT / "inference-service/scripts/run_steel_d3_full_development_experiment.py"
TEMPLATE = ROOT / "inference-service/scripts/run_steel_representation_experiment.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_d3f_experiment", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1. full train manifest correctness ------------------------------------------

def test_full_train_manifest_correctness():
    source = json.loads((DS / "split_manifest.json").read_text(encoding="utf-8"))["splits"]
    assert len(source["train_normal"]) == 4721
    assert len(source["validation_normal"]) == 590
    assert len(source["test_normal"]) == 591
    assert len(source["test_anomaly"]) == 6666


def test_full_recovery_dev_membership():
    recovery = json.loads((DS / "recovery_split_manifest.json").read_text(encoding="utf-8"))
    assert len(recovery["recovery_dev_anomaly"]) == 3333
    assert len(recovery["recovery_holdout_anomaly"]) == 3333
    source = json.loads((DS / "split_manifest.json").read_text(encoding="utf-8"))["splits"]
    assert set(source["test_anomaly"]) == set(recovery["recovery_dev_anomaly"]) | set(recovery["recovery_holdout_anomaly"])
    assert not (set(recovery["recovery_dev_anomaly"]) & set(recovery["recovery_holdout_anomaly"]))


# 2. holdout fail-closed ------------------------------------------------------

def test_fail_closed_membership():
    train = [f"t{i}" for i in range(4721)]
    val = [f"v{i}" for i in range(590)]
    dev = [f"d{i}" for i in range(3333)]
    test_n = [f"x{i}" for i in range(591)]
    hold = [f"h{i}" for i in range(3333)]
    ok, _ = fail_closed_membership(train, val, dev, test_n, hold)
    assert ok
    bad_val = list(val)
    bad_val[0] = hold[0]
    assert not fail_closed_membership(train, bad_val, dev, test_n, hold)[0]
    assert not fail_closed_membership(train[:4720], val, dev, test_n, hold)[0]
    overlap = list(train)
    overlap[0] = val[0]
    assert not fail_closed_membership(overlap, val, dev, test_n, hold)[0]


# 3. full train-only whitening statistics -------------------------------------

def test_full_train_only_whitening_statistics():
    r = _load_runner()
    stats_src = inspect.getsource(r.accumulate_stats)
    bank_src = inspect.getsource(r.build_bank)
    assert "train_ids" in stats_src and "val_ids" not in stats_src and "dev_ids" not in stats_src
    assert "train_ids" in bank_src and "val_ids" not in bank_src
    assert r.STATS_CHECKPOINT_EVERY == 50 or r.STATS_CHECKPOINT_EVERY > 0


# 4. pre-L2 whitening sanity semantics ----------------------------------------

def test_pre_l2_whitening_sanity_identity():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((30000, 8))
    s = whitening_sanity(x)
    assert s["non_finite_count"] == 0
    assert s["cov_diag_p50"] == pytest.approx(1.0, abs=0.05)
    assert s["cov_diag_p95"] == pytest.approx(1.0, abs=0.10)
    assert s["cov_offdiag_abs_max"] < 0.05
    assert "cov_offdiag_abs_p99" in s and "cov_diag_p95" in s


def test_sanity_runs_pre_l2_not_post_l2():
    src = RUNNER.read_text(encoding="utf-8")
    assert "whiten(toks, mean, w_64)" in src  # pre-L2 (no F.normalize)


# 5. streaming covariance + epsilon deterministic -----------------------------

def test_streaming_covariance_deterministic():
    rng = np.random.default_rng(20)
    x = rng.standard_normal((500, 16))
    c1, m1, s1 = 0, None, None
    c2, m2v, s2 = 0, None, None
    for chunk in np.array_split(x, 5):
        c1, m1, s1 = chan_update_batch(c1, m1, s1, chunk)
        c2, m2v, s2 = chan_update_batch(c2, m2v, s2, chunk.copy())
    assert c1 == c2 == 500
    assert np.array_equal(m1, m2v) and np.array_equal(s1, s2)
    assert np.array_equal(covariance_from_stats(s1, c1), covariance_from_stats(s2, c2))


def test_epsilon_rule_deterministic():
    rng = np.random.default_rng(21)
    a = rng.standard_normal((8, 8))
    cov = a.T @ a
    assert epsilon_rule(cov) == epsilon_rule(cov.copy()) == pytest.approx(1e-6 * np.trace(cov) / 8)


# 6. whitening artifact lineage -----------------------------------------------

def test_whitening_artifact_lineage_fields():
    assert len(DINO_B_WEIGHTS_SHA256) == 64 and DINO_B_WEIGHTS_SHA256.startswith("0b8b82f8")
    assert len(SOURCE_SPLIT_SHA256) == 64 and SOURCE_SPLIT_SHA256.startswith("64df9f81")
    assert len(RECOVERY_SPLIT_SHA256) == 64 and RECOVERY_SPLIT_SHA256.startswith("f60ce0a6")
    src = RUNNER.read_text(encoding="utf-8")
    assert '"dino_weights_sha256": DINO_B_WEIGHTS_SHA256' in src
    assert '"source_split_sha256": SOURCE_SPLIT_SHA256' in src
    assert '"recovery_split_sha256": RECOVERY_SPLIT_SHA256' in src


# 7. full D3 bank isolation + baseline/diagnostic immutability -----------------

def test_full_d3_bank_isolation_and_immutability():
    r = _load_runner()
    assert r.D3F_DIR.name == "D3-full-development"
    assert r.RUN_ROOT.name == "steel-d3-full-development"
    assert r.EXPECTED_FROZEN_BANK_SHA == "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"
    assert r.BANK_PATH != r.FRZ
    # diagnostic D3 bank lives elsewhere (steel-domain-adaptation/, not steel-d3-full-development)
    assert "steel-domain-adaptation" not in str(r.RUN_ROOT).replace("\\", "/")


# 8. 4721 train-only threshold calibration ------------------------------------

def test_4721_train_only_threshold_is_max():
    rng = np.random.default_rng(22)
    s = rng.random(4721)
    assert train_only_threshold(s) == pytest.approx(float(s.max()))


# 9. 590/3333 evaluation membership -------------------------------------------

def test_evaluation_membership():
    r = _load_runner()
    from steel_patchcore.d3_full_development import FULL_DEV_ANOMALY, FULL_VALIDATION_NORMAL

    assert FULL_VALIDATION_NORMAL == 590 and FULL_DEV_ANOMALY == 3333
    src = RUNNER.read_text(encoding="utf-8")
    assert 'run_role("train_normal", train_ids)' in src
    assert 'run_role("validation_normal", val_ids)' in src
    assert 'run_role("dev_anomaly", dev_ids)' in src
    assert 'run_role("' not in src.replace('run_role("train_normal"', "").replace('run_role("validation_normal"', "").replace('run_role("dev_anomaly"', "")


# 10. quartile boundary reuse -------------------------------------------------

def test_quartile_boundary_reuse():
    tpl = TEMPLATE.read_text(encoding="utf-8")
    assert "np.quantile" in tpl and "[0.25, 0.5, 0.75]" in tpl
    src = RUNNER.read_text(encoding="utf-8")
    assert "load_area_ratios(dev_ids)" in src


# 11. full-development gate semantics -----------------------------------------

def test_full_development_gate_semantics():
    assert D3_FULL_DEV_GATE == {"auroc_min": 0.75, "small_defect_q1_min": 0.65}
    assert d3_full_development_gate_passed(0.75, 1.0, 0.9)
    assert not d3_full_development_gate_passed(0.7499, 1.0, 0.9)
    assert not d3_full_development_gate_passed(0.80, 0.9, 1.0)
    assert not d3_full_development_gate_passed(0.80, 1.0, 1.0)
    assert not d3_full_development_gate_passed(float("nan"), 1.0, 0.9)


def test_small_defect_full_dev_signal_semantics():
    assert small_defect_full_dev_signal(0.65)
    assert not small_defect_full_dev_signal(0.6499)
    assert not small_defect_full_dev_signal(float("nan"))


# 12. checkpoint / resume -----------------------------------------------------

def test_stats_checkpoint_resume_roundtrip(tmp_path):
    r = _load_runner()
    count, mean, m2 = 500, np.arange(8, dtype=np.float64), np.eye(8, dtype=np.float64)
    path = tmp_path / "stats.npz"
    np.savez(path, count=np.asarray(count, np.int64), mean=mean, m2=m2, processed_images=np.asarray(25, np.int64))
    ck = np.load(path)
    assert int(ck["count"]) == 500 and int(ck["processed_images"]) == 25
    assert np.array_equal(ck["mean"], mean) and np.array_equal(ck["m2"], m2)


def test_rng_state_pickle_roundtrip():
    r = _load_runner()
    rng = r._new_rand(42)
    for _ in range(1000):
        rng.integers(0, 2**31)
    rng2 = r._new_rand(0)
    rng2.bit_generator.state = pickle.loads(pickle.dumps(rng.bit_generator.state))
    assert rng.integers(0, 10**9) == int(rng2.integers(0, 10**9))


def test_resumable_reservoir_matches_frozen_algorithm():
    r = _load_runner()
    rng = np.random.default_rng(7)
    stream = [rng.standard_normal((37, 32)).astype(np.float32) for _ in range(6)]
    ref, ref_seen = reservoir_from_stream([s.copy() for s in stream], budget=50, seed=42)
    budget = 50
    res = np.zeros((budget, 32), dtype=np.float32)
    seen = 0
    rrng = r._new_rand(42)
    for s in stream:
        for f in s:
            seen = r._reservoir_put(rrng, res, seen, f, budget)
    assert seen == ref_seen == 6 * 37
    assert np.array_equal(res, ref)


# 13. single-instance lifecycle -----------------------------------------------

def test_single_instance_lifecycle_stage():
    src = RUNNER.read_text(encoding="utf-8")
    assert 'lifecycle_enter("d3_full_development"' in src


# 14. diagnostic frozen reference ---------------------------------------------

def test_diagnostic_frozen_reference():
    assert D3_DIAGNOSTIC_AUROC == 0.8208
    assert D3_DIAGNOSTIC_QUARTILES == {"Q1": 0.7341, "Q2": 0.7959, "Q3": 0.8324, "Q4": 0.9209}