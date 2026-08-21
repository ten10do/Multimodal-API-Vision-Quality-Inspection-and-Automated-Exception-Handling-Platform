"""Representation-side localization and dual-objective isolation tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.d3_localization_representation import (  # noqa: E402
    D3_IMAGE_AUROC,
    LOCALIZATION_GATE,
    REPRESENTATION_SPECS,
    LocalizationRepresentationError,
    assert_image_branch_immutable,
    dual_objective_gate,
    fuse_dense_maps,
    score_delta_summary,
    validate_results_report,
)


def test_representation_set_and_frozen_d3_baseline_are_exact():
    assert set(REPRESENTATION_SPECS) == {"R-L1", "R-L2", "R-L3", "R-L4"}
    assert REPRESENTATION_SPECS["R-L1"]["grid"] == [18, 18]
    assert REPRESENTATION_SPECS["R-L2"]["grid"] == [32, 32]
    assert REPRESENTATION_SPECS["R-L4"]["encoder"] == "DINOv2-S/14 self-supervised"
    assert D3_IMAGE_AUROC == 0.8179071714278028
    assert LOCALIZATION_GATE == {"pixel_auroc_min": 0.75, "aupro_min": 0.50, "image_auroc_min": 0.75}


def test_dense_fusion_is_reproducible_read_only_and_equal_weighted():
    first = np.arange(12, dtype=np.float32).reshape(3, 4)
    second = np.flip(first, axis=1).copy()
    fused_a = fuse_dense_maps(first, second)
    fused_b = fuse_dense_maps(first, second)
    assert np.array_equal(fused_a, fused_b)
    assert np.array_equal(fused_a, (first + second) * 0.5)
    assert not fused_a.flags.writeable
    assert np.array_equal(first, np.arange(12, dtype=np.float32).reshape(3, 4))


def test_d3_image_branch_requires_byte_identical_scores():
    scores = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    assert_image_branch_immutable(scores, scores.copy())
    with pytest.raises(LocalizationRepresentationError, match="IMAGE_BRANCH_CHANGED"):
        assert_image_branch_immutable(scores, scores + np.asarray([0.0, 0.0, 1e-15]))


def test_score_delta_summary_records_signed_and_absolute_change():
    summary = score_delta_summary(np.asarray([0.2, 0.6]), np.asarray([0.1, 0.4]))
    assert summary["mean_signed"] == pytest.approx(0.15)
    assert summary["mean_absolute"] == pytest.approx(0.15)
    assert summary["max_absolute"] == pytest.approx(0.2)
    assert summary["p95_absolute"] == pytest.approx(0.195)


@pytest.mark.parametrize(
    ("pixel", "aupro", "image", "expected"),
    [(0.75, 0.50, 0.75, True), (0.7499, 0.50, 0.80, False), (0.80, 0.4999, 0.80, False), (0.80, 0.60, 0.7499, False)],
)
def test_dual_objective_gate_requires_all_three_metrics(pixel, aupro, image, expected):
    passed, checks = dual_objective_gate(pixel, aupro, image)
    assert passed is expected
    assert set(checks) == {"pixel_auroc", "aupro", "image_auroc"}


def test_result_schema_enforces_candidate_only_and_immutable_image_branch():
    rows = [
        {
            "candidate": name,
            "image_auroc": 0.8,
            "pixel_auroc": 0.76,
            "aupro": 0.51,
            "dual_objective": {"image_score_immutable": True},
        }
        for name in REPRESENTATION_SPECS
    ]
    report = {
        "schema_version": "steel_patchcore_d3_localization_representation_results_v1",
        "candidate_status": "CANDIDATE",
        "production_promotion": False,
        "threshold_changed": False,
        "artifact_unchanged": True,
        "representations": rows,
    }
    validate_results_report(report)
    rows[0]["dual_objective"]["image_score_immutable"] = False
    with pytest.raises(LocalizationRepresentationError, match="IMAGE_SCORE_NOT_IMMUTABLE"):
        validate_results_report(report)


def test_runner_is_representation_isolated_from_candidate_writes_and_training():
    source = (ROOT / "inference-service/scripts/investigate_steel_d3_localization_representation.py").read_text(
        encoding="utf-8"
    )
    assert "registry.register" not in source
    assert "train(" not in source
    assert "optimizer" not in source.lower()
    assert "threshold_changed\": False" in source
    assert "production_promotion\": False" in source
    assert 'RUN_ROOT / "R-L1"' in source and 'RUN_ROOT / "R-L2"' in source
    assert "manifest[\"threshold\"] =" not in source
