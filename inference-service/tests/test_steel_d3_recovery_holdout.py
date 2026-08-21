"""Pre-holdout freeze tests for the frozen D3 one-shot evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

import steel_patchcore.d3_recovery_holdout as holdout  # noqa: E402
from steel_patchcore.d3_recovery_holdout import (  # noqa: E402
    EXPECTED_LINEAGE,
    FROZEN_QUARTILE_BOUNDARIES,
    FROZEN_THRESHOLD,
    HOLDOUT_COUNTS,
    HoldoutBlocked,
    a0_global_max,
    assign_frozen_quartiles,
    assert_artifacts_unchanged,
    evaluate_holdout,
    gate_verdict,
    load_frozen_threshold,
    new_checkpoint,
    record_checkpoint_result,
    sha256_file,
    stratified_bootstrap_auroc,
    validate_checkpoint,
    validate_holdout_membership,
    verify_artifact_lineage,
)

DS = ROOT / "model-training/datasets/severstal-steel"
RUNNER = ROOT / "inference-service/scripts/run_steel_d3_recovery_holdout.py"
PROTOCOL = ROOT / "docs/steel-patchcore-d3-recovery-holdout-protocol.md"

ARTIFACT_PATHS = {
    "baseline_bank_sha256": ROOT / "inference-service/models/steel-patchcore/bank.npz",
    "source_split_sha256": DS / "split_manifest.json",
    "recovery_split_sha256": DS / "recovery_split_manifest.json",
    "evidence_manifest_sha256": DS / "recovery_evidence_manifest.json",
    "dino_weights_sha256": Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
    "whitening_sha256": ROOT / "model-training/runs/steel-d3-full-development/D3-full-development/whitening.npz",
    "d3_bank_sha256": ROOT / "model-training/runs/steel-d3-full-development/D3-full-development/bank.npz",
    "quartile_manifest_sha256": DS / "representation_diagnostic_manifest.json",
    "d3_results_sha256": ROOT / "docs/steel-patchcore-d3-full-development-results.json",
}


def _actual_manifests():
    source = json.loads((DS / "split_manifest.json").read_text(encoding="utf-8"))
    recovery = json.loads((DS / "recovery_split_manifest.json").read_text(encoding="utf-8"))
    return source, recovery


def _synthetic_manifests():
    train = [f"t{i}" for i in range(4721)]
    validation = [f"v{i}" for i in range(590)]
    test_normal = [f"n{i}" for i in range(591)]
    dev = [f"d{i}" for i in range(3333)]
    sealed = [f"h{i}" for i in range(3333)]
    source = {"splits": {
        "train_normal": train,
        "validation_normal": validation,
        "test_normal": test_normal,
        "test_anomaly": dev + sealed,
    }}
    recovery = {"recovery_dev_anomaly": dev, "recovery_holdout_anomaly": sealed}
    return source, recovery


def test_holdout_manifest_membership_and_counts():
    roles = validate_holdout_membership(*_actual_manifests())
    assert {role: len(ids) for role, ids in roles.items()} == HOLDOUT_COUNTS
    assert all(len(ids) == len(set(ids)) for ids in roles.values())


def test_development_and_holdout_are_disjoint():
    source, recovery = _actual_manifests()
    roles = validate_holdout_membership(source, recovery)
    development = (
        set(source["splits"]["train_normal"])
        | set(source["splits"]["validation_normal"])
        | set(recovery["recovery_dev_anomaly"])
    )
    assert not development.intersection(*roles.values())


def test_split_validation_fails_closed_on_overlap_and_duplicates():
    source, recovery = _synthetic_manifests()
    source["splits"]["validation_normal"][0] = recovery["recovery_holdout_anomaly"][0]
    with pytest.raises(HoldoutBlocked, match="DEVELOPMENT_HOLDOUT_INTERSECTION"):
        validate_holdout_membership(source, recovery)
    source, recovery = _synthetic_manifests()
    recovery["recovery_holdout_anomaly"][1] = recovery["recovery_holdout_anomaly"][0]
    with pytest.raises(HoldoutBlocked, match="HOLDOUT_DUPLICATE_IDS"):
        validate_holdout_membership(source, recovery)


def test_frozen_lineage_matches_every_local_artifact():
    assert verify_artifact_lineage(ARTIFACT_PATHS) == EXPECTED_LINEAGE


def test_lineage_verification_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"frozen")
    expected = {"artifact_sha256": sha256_file(path)}
    monkeypatch.setattr(holdout, "EXPECTED_LINEAGE", expected)
    assert verify_artifact_lineage({"artifact_sha256": path}) == expected
    path.write_bytes(b"changed")
    with pytest.raises(HoldoutBlocked, match="ARTIFACT_SHA_MISMATCH"):
        verify_artifact_lineage({"artifact_sha256": path})


def test_threshold_is_loaded_at_full_precision_only():
    results = json.loads((ROOT / "docs/steel-patchcore-d3-full-development-results.json").read_text(encoding="utf-8"))
    assert load_frozen_threshold(results) == FROZEN_THRESHOLD
    results["full"]["metrics"]["threshold"] = FROZEN_THRESHOLD + 1e-12
    with pytest.raises(HoldoutBlocked, match="FROZEN_THRESHOLD_MISMATCH"):
        load_frozen_threshold(results)


def test_runner_has_no_recalibration_or_candidate_search():
    source = RUNNER.read_text(encoding="utf-8")
    assert "load_frozen_threshold(d3_results)" in source
    assert "train_only_threshold" not in source
    assert "candidate_search" not in source
    assert "np.quantile" not in source
    assert "recovery_dev_anomaly\"]" not in source


def test_checkpoint_resume_roundtrip():
    roles = {"test_normal": ["n1", "n2"], "recovery_holdout_anomaly": ["a1"]}
    lineage = {"sha": "abc"}
    checkpoint = new_checkpoint(lineage, 0.5, "2026-01-01T00:00:00Z")
    record_checkpoint_result(checkpoint, "test_normal", "n1", 0.4)
    record_checkpoint_result(checkpoint, "recovery_holdout_anomaly", "a1", 0.8)
    assert validate_checkpoint(checkpoint, lineage, roles, 0.5) == {
        "test_normal": 1,
        "recovery_holdout_anomaly": 1,
    }
    assert checkpoint["completed"]["test_normal"]["n1"]["prediction"] == 0
    assert checkpoint["completed"]["recovery_holdout_anomaly"]["a1"]["prediction"] == 1


def test_checkpoint_duplicate_id_is_rejected():
    checkpoint = new_checkpoint({"sha": "abc"}, 0.5, "2026-01-01T00:00:00Z")
    record_checkpoint_result(checkpoint, "test_normal", "same", 0.4)
    with pytest.raises(HoldoutBlocked, match="CHECKPOINT_DUPLICATE_ID"):
        record_checkpoint_result(checkpoint, "recovery_holdout_anomaly", "same", 0.8)


def test_checkpoint_rejects_foreign_lineage_and_threshold():
    roles = {"test_normal": [], "recovery_holdout_anomaly": []}
    checkpoint = new_checkpoint({"sha": "abc"}, 0.5, "2026-01-01T00:00:00Z")
    with pytest.raises(HoldoutBlocked, match="CHECKPOINT_LINEAGE_MISMATCH"):
        validate_checkpoint(checkpoint, {"sha": "def"}, roles, 0.5)
    with pytest.raises(HoldoutBlocked, match="CHECKPOINT_THRESHOLD_MISMATCH"):
        validate_checkpoint(checkpoint, {"sha": "abc"}, roles, 0.6)


def test_a0_is_global_max_across_tiles_and_patches():
    distances = np.asarray([[[0.1, 0.2]], [[0.9, 0.3]], [[0.4, 0.5]]])
    assert a0_global_max(distances) == pytest.approx(0.9)


def test_metric_and_gate_semantics():
    normal = np.asarray([0.1, 0.2, 0.3, 0.4])
    anomaly = np.asarray([0.7, 0.8, 0.9, 1.0])
    metrics = evaluate_holdout(normal, anomaly, 0.75, np.asarray([1, 2, 3, 4]))
    assert metrics["image_auroc"] == 1.0
    assert metrics["normal_distribution"]["n"] == 4
    assert metrics["anomaly_distribution"]["p50"] == pytest.approx(0.85)
    assert metrics["operating_point"]["tp"] == 3
    assert metrics["operating_point"]["tn"] == 4
    assert gate_verdict(0.75, 0.8, 0.7) == "RECOVERY_HOLDOUT_PASS"
    assert gate_verdict(0.7499, 0.8, 0.7) == "RECOVERY_HOLDOUT_FAILED"
    assert gate_verdict(0.9, 0.7, 0.7) == "RECOVERY_HOLDOUT_FAILED"


def test_frozen_quartile_boundaries_are_reused_without_fitting():
    assert FROZEN_QUARTILE_BOUNDARIES == (
        0.010888671875,
        0.02656494140625,
        0.07214111328125,
    )
    q1, q2, q3 = FROZEN_QUARTILE_BOUNDARIES
    ratios = np.asarray([q1 - 1e-9, q1, q2, q3, q3 + 1e-9])
    assert assign_frozen_quartiles(ratios).tolist() == [1, 2, 3, 4, 4]


def test_bootstrap_is_deterministic_and_report_only():
    normal = np.asarray([0.1, 0.2, 0.3, 0.4])
    anomaly = np.asarray([0.5, 0.6, 0.7, 0.8])
    first = stratified_bootstrap_auroc(normal, anomaly, seed=42, iterations=50)
    second = stratified_bootstrap_auroc(normal, anomaly, seed=42, iterations=50)
    assert first == second
    assert first["median"] == 1.0
    assert first["percentile_95_ci"] == [1.0, 1.0]


def test_artifact_immutability_guard(tmp_path, monkeypatch):
    path = tmp_path / "frozen.bin"
    path.write_bytes(b"same")
    expected = {"artifact_sha256": sha256_file(path)}
    monkeypatch.setattr(holdout, "EXPECTED_LINEAGE", expected)
    before = verify_artifact_lineage({"artifact_sha256": path})
    assert_artifacts_unchanged({"artifact_sha256": path}, before)
    path.write_bytes(b"mutated")
    with pytest.raises(HoldoutBlocked):
        assert_artifacts_unchanged({"artifact_sha256": path}, before)


def test_frozen_method_and_checkpoint_are_explicit_in_runner():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'output["x_norm_patchtokens"]' in source
    assert "tuple(tokens.shape) != (18 * 18, 768)" in source
    assert "embedding @ bank.T" in source
    assert "a0_global_max(tile_maxima)" in source
    assert "CHECKPOINT_EVERY = 25" in source
    assert "validate_checkpoint(checkpoint, lineage, roles, threshold)" in source


def test_protocol_freezes_required_semantics():
    text = PROTOCOL.read_text(encoding="utf-8")
    for phrase in (
        "Frozen D3 definition",
        "Artifact lineage",
        "Metric semantics",
        "Holdout split semantics",
        "Gate definition",
        "Checkpoint resume rules",
        "No recalibration",
    ):
        assert phrase in text
