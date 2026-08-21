"""Pure contracts for D3 localization-aware representation experiments."""
from __future__ import annotations

from typing import Final

import numpy as np

D3_IMAGE_AUROC: Final = 0.8179071714278028
LOCALIZATION_GATE: Final = {"pixel_auroc_min": 0.75, "aupro_min": 0.50, "image_auroc_min": 0.75}
EXPERIMENTAL_BANK_BUDGET: Final = 50_000
EXPERIMENTAL_BANK_SEED: Final = 42
EXPERIMENTAL_TRAIN_COUNT: Final = 1_000
INTERMEDIATE_BLOCK_INDEX: Final = 7
HIGH_RESOLUTION_SIDE: Final = 448

REPRESENTATION_SPECS: Final[dict[str, dict]] = {
    "R-L1": {
        "encoder": "DINOv2-B/14",
        "feature": "normalized patch tokens after transformer block index 7",
        "input_side": 252,
        "grid": [18, 18],
        "dimension": 768,
        "distance": "cosine-1NN",
    },
    "R-L2": {
        "encoder": "DINOv2-B/14",
        "feature": "final normalized patch tokens at higher input resolution",
        "input_side": HIGH_RESOLUTION_SIDE,
        "grid": [32, 32],
        "dimension": 768,
        "distance": "cosine-1NN",
    },
    "R-L3": {
        "encoder": "DINOv2-B/14",
        "feature": "equal fusion of R-L1 and R-L2 dense cosine-distance maps",
        "input_side": [252, HIGH_RESOLUTION_SIDE],
        "grid": [256, 256],
        "dimension": None,
        "distance": "mean(R-L1 distance map, R-L2 distance map)",
    },
    "R-L4": {
        "encoder": "DINOv2-S/14 self-supervised",
        "feature": "final normalized dense patch tokens",
        "input_side": 252,
        "grid": [18, 18],
        "dimension": 384,
        "distance": "cosine-1NN",
    },
}


class LocalizationRepresentationError(RuntimeError):
    """Representation isolation, reproducibility, or dual-branch violation."""


def assert_image_branch_immutable(reference_scores: np.ndarray, branch_scores: np.ndarray) -> None:
    reference = np.asarray(reference_scores, dtype=np.float64)
    branch = np.asarray(branch_scores, dtype=np.float64)
    if reference.shape != branch.shape or reference.tobytes() != branch.tobytes():
        raise LocalizationRepresentationError("D3_IMAGE_BRANCH_CHANGED")


def fuse_dense_maps(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise LocalizationRepresentationError("DENSE_FUSION_INPUT_INVALID")
    fused = ((left.astype(np.float64) + right.astype(np.float64)) * 0.5).astype(np.float32)
    fused.flags.writeable = False
    return fused


def score_delta_summary(representation_scores: np.ndarray, d3_scores: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(representation_scores, dtype=np.float64)
    baseline = np.asarray(d3_scores, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or not len(candidate):
        raise LocalizationRepresentationError("SCORE_DELTA_INPUT_INVALID")
    delta = candidate - baseline
    return {
        "mean_signed": float(np.mean(delta)),
        "mean_absolute": float(np.mean(np.abs(delta))),
        "p95_absolute": float(np.percentile(np.abs(delta), 95)),
        "max_absolute": float(np.max(np.abs(delta))),
    }


def dual_objective_gate(pixel_auroc: float, aupro: float, image_auroc: float) -> tuple[bool, dict[str, bool]]:
    checks = {
        "pixel_auroc": float(pixel_auroc) >= LOCALIZATION_GATE["pixel_auroc_min"],
        "aupro": float(aupro) >= LOCALIZATION_GATE["aupro_min"],
        "image_auroc": float(image_auroc) >= LOCALIZATION_GATE["image_auroc_min"],
    }
    return all(checks.values()), checks


def validate_results_report(report: dict) -> None:
    if report.get("schema_version") != "steel_patchcore_d3_localization_representation_results_v1":
        raise LocalizationRepresentationError("RESULT_SCHEMA_MISMATCH")
    if report.get("candidate_status") != "CANDIDATE" or report.get("production_promotion") is not False:
        raise LocalizationRepresentationError("CANDIDATE_ONLY_REQUIRED")
    if report.get("threshold_changed") is not False or report.get("artifact_unchanged") is not True:
        raise LocalizationRepresentationError("FROZEN_D3_VIOLATION")
    rows = report.get("representations")
    if not isinstance(rows, list) or {row.get("candidate") for row in rows} != set(REPRESENTATION_SPECS):
        raise LocalizationRepresentationError("REPRESENTATION_SET_MISMATCH")
    for row in rows:
        for key in ("image_auroc", "pixel_auroc", "aupro"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise LocalizationRepresentationError(f"REPRESENTATION_METRIC_INVALID:{row.get('candidate')}:{key}")
        if row.get("dual_objective", {}).get("image_score_immutable") is not True:
            raise LocalizationRepresentationError(f"IMAGE_SCORE_NOT_IMMUTABLE:{row.get('candidate')}")


__all__ = [
    "D3_IMAGE_AUROC",
    "EXPERIMENTAL_BANK_BUDGET",
    "EXPERIMENTAL_BANK_SEED",
    "EXPERIMENTAL_TRAIN_COUNT",
    "HIGH_RESOLUTION_SIDE",
    "INTERMEDIATE_BLOCK_INDEX",
    "LOCALIZATION_GATE",
    "LocalizationRepresentationError",
    "REPRESENTATION_SPECS",
    "assert_image_branch_immutable",
    "dual_objective_gate",
    "fuse_dense_maps",
    "score_delta_summary",
    "validate_results_report",
]
