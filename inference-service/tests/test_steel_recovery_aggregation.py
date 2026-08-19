"""Deterministic offline A0-A6 aggregation primitives regression tests.

These cover image-level aggregation recovery semantics without touching any
raw evidence shard, model, or the sealed recovery holdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.aggregation import (  # noqa: E402
    CANDIDATE_IDS,
    DEVELOPMENT_GATE,
    auroc,
    candidate_scores_for_grids,
    distribution,
    gate_passed,
    normal_vs_quartile_auroc,
    operating_point,
    quartile_assign,
    select_best,
    top_percentage_count,
    train_only_threshold,
)
from steel_patchcore.recovery import (  # noqa: E402
    CANDIDATE_GRID,
    CAPTURE_ROLES,
    HOLDOUT_ROLES,
    candidate_score,
    stitch_raw_patch_grids,
)
from steel_patchcore.tile import IMG_W, TILE, TILE_X0  # noqa: E402


def test_candidate_grid_is_frozen_and_matches_recovery_definitions():
    assert CANDIDATE_IDS == tuple(candidate["id"] for candidate in CANDIDATE_GRID)
    assert CANDIDATE_IDS == ("A0", "A1", "A2", "A3", "A4", "A5", "A6")


def test_development_gate_is_frozen():
    assert DEVELOPMENT_GATE == {
        "image_auroc_min": 0.75,
        "normal_fpr_max": 0.10,
        "anomaly_recall_min": 0.60,
    }


def test_holdout_roles_are_excluded_from_capture_roles():
    assert set(CAPTURE_ROLES) == {"train_normal", "validation_normal", "recovery_dev_anomaly"}
    assert set(CAPTURE_ROLES).isdisjoint(HOLDOUT_ROLES)


def test_top_percentage_count_uses_ceil_with_minimum_one():
    # 32 x 200 = 6400 stitched cells.
    assert top_percentage_count(6400, 0.001) == 7
    assert top_percentage_count(6400, 0.005) == 32
    assert top_percentage_count(6400, 0.01) == 64
    assert top_percentage_count(0, 0.5) == 1
    assert top_percentage_count(3, 0.0001) == 1


def test_candidate_scores_for_grids_are_deterministic_and_match_frozen_semantics():
    raw = (np.arange(7 * 32 * 32, dtype=np.float32).reshape(7, 32, 32) / 1000.0)
    stitched, _ = stitch_raw_patch_grids(raw, TILE_X0, tile_size=TILE, original_width=IMG_W)

    first = candidate_scores_for_grids(raw[np.newaxis, ...], TILE_X0, tile_size=TILE, original_width=IMG_W)
    second = candidate_scores_for_grids(raw[np.newaxis, ...], TILE_X0, tile_size=TILE, original_width=IMG_W)

    assert first.keys() == set(CANDIDATE_IDS)
    for candidate_id in CANDIDATE_IDS:
        assert first[candidate_id][0] == pytest.approx(
            candidate_score(candidate_id, raw, stitched)
        )
        assert first[candidate_id][0] == second[candidate_id][0]
    assert first["A0"][0] == pytest.approx(float(raw.max()))


def test_train_only_threshold_is_the_maximum():
    assert train_only_threshold(np.asarray([0.1, 0.3, 0.2])) == 0.3
    with pytest.raises(ValueError):
        train_only_threshold(np.asarray([]))


def test_operating_point_counts_and_derived_metrics():
    op = operating_point(
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([0.5, 0.6]),
        threshold=0.4,
    )
    assert op == {
        "tp": 2, "fp": 0, "tn": 3, "fn": 0,
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "normal_fpr": 0.0, "anomaly_recall": 1.0,
        "confusion_matrix": [[3, 0], [0, 2]],
    }

    partial = operating_point(
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([0.5, 0.6]),
        threshold=0.55,
    )
    assert partial["tp"] == 1 and partial["fn"] == 1
    assert partial["precision"] == 1.0
    assert partial["recall"] == pytest.approx(0.5)
    assert partial["normal_fpr"] == 0.0


def test_gate_passed_enforces_all_three_thresholds():
    base = {"image_auroc": 0.75, "normal_fpr": 0.10, "anomaly_recall": 0.60}
    assert gate_passed(base)
    assert not gate_passed({**base, "image_auroc": 0.7499})
    assert not gate_passed({**base, "normal_fpr": 0.1001})
    assert not gate_passed({**base, "anomaly_recall": 0.5999})
    assert not gate_passed({**base, "image_auroc": float("nan")})


def test_select_best_is_deterministic_and_orders_correctly():
    def case(candidate_id, index, auroc_value, fpr, recall, f1):
        return {
            "_index": index,
            "candidate_id": candidate_id,
            "image_auroc": auroc_value,
            "normal_fpr": fpr,
            "anomaly_recall": recall,
            "f1": f1,
            "gate_passed": True,
        }

    # AUROC difference exactly 0.01 keeps only the max in the FPR comparison.
    distant = [case("A1", 1, 0.90, 0.05, 0.9, 0.9), case("A0", 0, 0.89, 0.0, 0.9, 0.9)]
    assert select_best(distant)["candidate_id"] == "A1"

    # Within 0.01, the lower Normal FPR wins even with lower recall.
    close = [case("A1", 1, 0.90, 0.08, 0.90, 0.8), case("A2", 2, 0.895, 0.02, 0.70, 0.7)]
    assert select_best(close)["candidate_id"] == "A2"

    # Tied FPR falls back to higher recall.
    tied_fpr = [case("A1", 1, 0.90, 0.02, 0.70, 0.7), case("A2", 2, 0.90, 0.02, 0.80, 0.8)]
    assert select_best(tied_fpr)["candidate_id"] == "A2"

    # Full tie falls back to the semantically simpler (lower-index) candidate.
    tied = [case("A2", 2, 0.90, 0.02, 0.80, 0.8), case("A0", 0, 0.90, 0.02, 0.80, 0.8)]
    assert select_best(tied)["candidate_id"] == "A0"

    # No candidate passes the gate -> None.
    none_pass = [dict(c, gate_passed=False) for c in tied]
    assert select_best(none_pass) is None


def test_quartile_assign_uses_area_ratio_and_is_monotonic():
    ratios = np.arange(100, dtype=np.float64) / 100.0
    quartiles, (q1, q2, q3) = quartile_assign(ratios)
    assert quartiles.shape == ratios.shape
    assert set(quartiles.tolist()) == {1, 2, 3, 4}
    assert q1 < q2 < q3
    # Balanced counts for an evenly spaced sample.
    assert (quartiles == 1).sum() == 25
    assert (quartiles == 2).sum() == 25
    assert (quartiles == 3).sum() == 25
    assert (quartiles == 4).sum() == 25
    # Smallest areas land in Q1 and largest in Q4.
    assert quartiles[0] == 1
    assert quartiles[-1] == 4


def test_distribution_returns_ordered_summary():
    values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    dist = distribution(values)
    assert dist["n"] == 10
    assert dist["min"] == 0.0
    assert dist["p50"] == 4.5
    assert dist["p95"] == pytest.approx(8.55)
    assert dist["p99"] == pytest.approx(8.91)
    assert dist["max"] == 9.0


def test_normal_vs_quartile_auroc_is_not_anomaly_only():
    normal = np.asarray([0.0, 0.1, 0.2, 0.3])
    anomaly = np.asarray([0.7, 0.8, 0.9, 1.0])
    assert normal_vs_quartile_auroc(normal, anomaly) == 1.0
    reversed_auroc = normal_vs_quartile_auroc(anomaly, normal)
    assert reversed_auroc == 0.0


def test_auroc_separation():
    assert auroc(np.asarray([0.0, 0.1, 1.0, 0.9]), np.asarray([0, 0, 1, 1])) == 1.0
    assert auroc(np.asarray([1.0, 0.9, 0.0, 0.1]), np.asarray([0, 0, 1, 1])) == 0.0