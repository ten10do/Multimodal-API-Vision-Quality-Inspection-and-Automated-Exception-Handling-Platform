"""D3 heatmap recovery must remain reproducible and score-isolated."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.d3_heatmap_recovery import (  # noqa: E402
    CANDIDATE_SPECS,
    HeatmapRecoveryError,
    assert_image_score_unchanged,
    frozen_a0_score,
    generate_heatmap_candidates,
    localization_gate,
    weighted_stitch,
)
from steel_patchcore.d3_operational import pixel_localization_metrics  # noqa: E402


def _grids() -> np.ndarray:
    base = np.linspace(0.1, 0.9, 18 * 18, dtype=np.float32).reshape(18, 18)
    return np.stack([base + index * 0.01 for index in range(7)]).astype(np.float32)


def test_all_frozen_candidates_are_reproducible_and_complete():
    grids = _grids()
    first = generate_heatmap_candidates(grids)
    second = generate_heatmap_candidates(grids)
    assert set(first) == set(second) == set(CANDIDATE_SPECS) == {"H0", "H1", "H2", "H3", "H4", "H5"}
    for name in CANDIDATE_SPECS:
        assert first[name].shape == (256, 1600)
        assert np.array_equal(first[name], second[name])
        assert not first[name].flags.writeable


def test_heatmap_candidates_cannot_mutate_frozen_a0_score_or_input():
    grids = _grids()
    original = grids.copy()
    score_before = frozen_a0_score(grids)
    generate_heatmap_candidates(grids)
    score_after = frozen_a0_score(grids)
    assert_image_score_unchanged(score_before, score_after)
    assert np.array_equal(grids, original)
    with pytest.raises(HeatmapRecoveryError, match="IMAGE_SCORE_CHANGED"):
        assert_image_score_unchanged(score_before, score_after + 0.01)


def test_h0_normalization_and_h2_raw_map_have_identical_localization_metrics():
    maps = generate_heatmap_candidates(_grids())
    mask = np.zeros((256, 1600), dtype=np.uint8)
    mask[128:, 800:] = 1
    h0 = pixel_localization_metrics(maps["H0"], mask)
    h2 = pixel_localization_metrics(maps["H2"], mask)
    assert h0["pixel_auroc"] == pytest.approx(h2["pixel_auroc"], abs=1e-12)
    assert h0["aupro"] == pytest.approx(h2["aupro"], abs=1e-12)


def test_h1_preserves_piecewise_constant_raw_patch_cells():
    maps = generate_heatmap_candidates(_grids())
    h1_values = np.unique(maps["H1"][:, :256])
    h2_values = np.unique(maps["H2"][:, :256])
    assert len(h1_values) <= 18 * 18
    assert len(h2_values) > len(h1_values)


def test_weighted_stitch_feathers_only_the_existing_overlap():
    tile_maps = [np.full((256, 256), index, dtype=np.float32) for index in range(7)]
    stitched = weighted_stitch(tile_maps)
    assert np.all(stitched[:, :256] == 0)
    assert np.all(stitched[:, 1024:1280] == 4)
    overlap = stitched[0, 1344:1536]
    assert overlap[0] > 5.0 and overlap[-1] < 6.0
    assert np.all(np.diff(overlap) > 0)
    assert np.all(stitched[:, 1536:] == 6)


def test_localization_gate_is_exactly_95_percent_of_both_baselines():
    passed, minimums = localization_gate(0.76, 0.57, 0.8, 0.6)
    assert passed is True
    assert minimums == {"pixel_auroc_min": pytest.approx(0.76), "aupro_min": pytest.approx(0.57)}
    assert localization_gate(0.759999, 0.57, 0.8, 0.6)[0] is False
    assert localization_gate(0.76, 0.569999, 0.8, 0.6)[0] is False


def test_recovery_source_is_artifact_isolated_and_has_no_training_or_threshold_write():
    module_source = (ROOT / "model-training/steel_patchcore/d3_heatmap_recovery.py").read_text(encoding="utf-8")
    runner_source = (ROOT / "inference-service/scripts/investigate_steel_d3_heatmaps.py").read_text(encoding="utf-8")
    assert "torch" not in module_source
    assert "train(" not in runner_source
    assert "optimizer" not in runner_source.lower()
    assert "threshold_changed\": False" in runner_source
    assert "production_promotion\": False" in runner_source
    assert "np.save" not in runner_source
