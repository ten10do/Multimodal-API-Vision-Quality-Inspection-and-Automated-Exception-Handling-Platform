"""Deterministic, offline A0-A6 aggregation evaluation for Steel PatchCore recovery.

This module contains the pure, CPU-only primitives used by the offline
evaluator. It never loads a model, never runs inference, and never touches the
sealed recovery holdout. All aggregation semantics are the frozen definitions
from ``steel_patchcore.recovery``.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from steel_patchcore.recovery import (
    CANDIDATE_GRID,
    baseline_score,
    candidate_score,
    stitch_raw_patch_grids,
)

CANDIDATE_IDS = tuple(candidate["id"] for candidate in CANDIDATE_GRID)

# Frozen development gate (not a production criterion).
DEVELOPMENT_GATE = {
    "image_auroc_min": 0.75,
    "normal_fpr_max": 0.10,
    "anomaly_recall_min": 0.60,
}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary image AUROC (0 = normal, 1 = anomaly). NaN when undefined."""
    from sklearn.metrics import roc_auc_score

    values = np.asarray(scores, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if values.size == 0 or len(np.unique(targets)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(targets, values))
    except ValueError:
        return float("nan")


def train_only_threshold(train_scores: np.ndarray) -> float:
    """Frozen threshold rule: max candidate score over train_normal only."""
    values = np.asarray(train_scores, dtype=float)
    if values.size == 0:
        raise ValueError("train_normal scores are empty")
    return float(np.max(values))


def operating_point(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> dict:
    """Confusion-based development operating point at a fixed threshold."""
    normal = np.asarray(normal_scores, dtype=float)
    anomaly = np.asarray(anomaly_scores, dtype=float)
    tp = int((anomaly >= threshold).sum())
    fp = int((normal >= threshold).sum())
    tn = int(normal.size) - fp
    fn = int(anomaly.size) - tp
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else 0.0
    )
    normal_fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normal_fpr": normal_fpr,
        "anomaly_recall": recall,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def distribution(values: np.ndarray) -> dict:
    """n/min/p50/p95/p99/max score distribution."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0, method="linear")),
        "p99": float(np.percentile(arr, 99.0, method="linear")),
        "max": float(arr.max()),
    }


def gate_passed(metrics: dict, gate: dict | None = None) -> bool:
    gate = gate or DEVELOPMENT_GATE
    if not np.isfinite(metrics["image_auroc"]):
        return False
    return (
        metrics["image_auroc"] >= gate["image_auroc_min"]
        and metrics["normal_fpr"] <= gate["normal_fpr_max"]
        and metrics["anomaly_recall"] >= gate["anomaly_recall_min"]
    )


def select_best(candidate_results: list[dict]) -> dict | None:
    """Deterministically freeze at most one development candidate.

    Ordering: max AUROC; candidates within 0.01 of that maximum are then ranked
    by lower Normal FPR, then higher Anomaly Recall, then higher F1, then the
    semantically simpler candidate (lower frozen-grid index).
    """
    passing = [result for result in candidate_results if result["gate_passed"]]
    if not passing:
        return None
    max_auroc = max(float(result["image_auroc"]) for result in passing)
    close = [result for result in passing if (max_auroc - float(result["image_auroc"])) < 0.01]
    close.sort(
        key=lambda result: (
            float(result["normal_fpr"]),
            -float(result["anomaly_recall"]),
            -float(result["f1"]),
            int(result["_index"]),
        )
    )
    return close[0]


def quartile_assign(area_ratios: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Assign fixed quartiles by defect area ratio (1=smallest, 4=largest)."""
    arr = np.asarray(area_ratios, dtype=float)
    q1, q2, q3 = (float(v) for v in np.quantile(arr, [0.25, 0.5, 0.75], method="linear"))
    quartiles = np.empty(arr.size, dtype=np.int8)
    quartiles[arr < q1] = 1
    quartiles[(arr >= q1) & (arr < q2)] = 2
    quartiles[(arr >= q2) & (arr < q3)] = 3
    quartiles[arr >= q3] = 4
    return quartiles, (q1, q2, q3)


def normal_vs_quartile_auroc(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """AUROC of validation-normal vs one anomaly quartile (not anomaly-only)."""
    normal = np.asarray(normal_scores, dtype=float)
    anomaly = np.asarray(anomaly_scores, dtype=float)
    values = np.concatenate([normal, anomaly])
    labels = np.concatenate([np.zeros(normal.size, dtype=int), np.ones(anomaly.size, dtype=int)])
    return auroc(values, labels)


def candidate_scores_for_grids(
    raw_grids: np.ndarray,
    tile_x_offsets: tuple[int, ...],
    *,
    tile_size: int,
    original_width: int,
) -> dict[str, np.ndarray]:
    """Compute all frozen A0-A6 scores for a batch of raw tile grids.

    ``raw_grids`` has shape (N, 7, H, W) float32. A0 uses the global maximum of
    the seven unstitched grids; A1-A6 use the mean-overlap stitched raw grid.
    """
    grids = np.asarray(raw_grids, dtype=np.float32)
    if grids.ndim != 4:
        raise ValueError("raw_grids must be (N, tiles, H, W)")
    scores: dict[str, np.ndarray] = {
        candidate["id"]: np.empty(grids.shape[0], dtype=np.float64)
        for candidate in CANDIDATE_GRID
    }
    for index in range(grids.shape[0]):
        raw = grids[index]
        stitched, _ = stitch_raw_patch_grids(
            raw, tile_x_offsets, tile_size=tile_size, original_width=original_width
        )
        for candidate in CANDIDATE_GRID:
            scores[candidate["id"]][index] = candidate_score(candidate["id"], raw, stitched)
    return scores


def top_percentage_count(total: int, fraction: float) -> int:
    """Frozen integer-k rule for top-percentage means: k = max(1, ceil(N * p))."""
    if total < 0:
        raise ValueError("total must be non-negative")
    return max(1, math.ceil(float(fraction) * total))