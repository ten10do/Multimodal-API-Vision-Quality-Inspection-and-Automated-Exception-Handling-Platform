"""Steel-domain representation adaptation primitives (Optimization 1.1).

Freezes the D0/D1/D2 references and the D3 candidate definition
(DINOv2 ViT-B/14 + train-normal ZCA covariance whitening). Pure, testable,
CPU-only: no model, no GPU, no holdout.

D0 = frozen S2 (WRN layer3 + 5x5). D1 = DINOv2 ViT-S/14. D2 = DINOv2 ViT-B/14.
D3 = DINOv2 ViT-B/14 whitened with TRAIN-NORMAL-ONLY mean + covariance (ZCA),
     then the identical frozen protocol (per-patch L2 + cosine 1-NN, A0 max).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from steel_patchcore.domain_representation import (  # noqa: E402
    D0_AUROC,
    D0_QUARTILES,
    reference_sha256,
    serialize_reference,
)
from steel_patchcore.domain_representation_capacity import (  # noqa: E402
    D1_AUROC,
    D1_QUARTILES,
    D2_REFERENCE,
)

ADAPTATION_PROTOCOL_VERSION = "domain_adaptation_protocol_v1"
ADAPTATION_SEED = 42
ADAPTATION_BANK_BUDGET = 50_000

# D2 = frozen DINOv2 ViT-B/14 (previous phase, never re-run).
D2_CANDIDATE_ID = "D2"
D2_AUROC = 0.6938
D2_QUARTILES = {"Q1": 0.6043, "Q2": 0.6318, "Q3": 0.7035, "Q4": 0.8358}

# D3 candidate identity: DINOv2 ViT-B/14 (unchanged backbone) + train-normal ZCA.
D3_CANDIDATE_ID = "D3"

# Frozen Adaptation Gate (stricter than prior phases; no goalpost lowering).
ADAPTATION_GATE = {
    "auroc_min": 0.75,
    "delta_vs_d2": 0.05,
    "strong_auroc": 0.80,
    "small_defect_q1_delta": 0.05,
}

# Numerical stabilization rule (NOT a tunable hyperparameter; no grid/search).
EPSILON_FACTOR = 1e-6


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


# --- streaming sufficient statistics (Chan et al. 1983 pairwise combine) -----

def chan_update_batch(count: int, mean: np.ndarray, m2: np.ndarray, x: np.ndarray):
    """Combine a batch `x` (m, d) into running count/mean/M2 statistics.

    Maintains M2 = sum_i (x_i - mean)^2 (sum of squared deviations, NOT divided),
    using the numerically stable pairwise (Chan-style) update in float64.
    """
    x = np.asarray(x, dtype=np.float64)
    m = int(x.shape[0])
    if m == 0:
        return int(count), mean, m2
    x_mean = x.mean(axis=0)
    x_m2 = (x - x_mean).T @ (x - x_mean)
    if count == 0:
        return m, x_mean.copy(), x_m2
    delta = x_mean - mean
    new_count = int(count) + m
    new_mean = mean + delta * (m / new_count)
    new_m2 = m2 + x_m2 + np.outer(delta, delta) * (int(count) * m / new_count)
    return new_count, new_mean, new_m2


def covariance_from_stats(m2: np.ndarray, count: int) -> np.ndarray:
    """Population covariance = M2 / n (fixed convention; n huge so n vs n-1 is negligible)."""
    if count <= 1:
        raise ValueError("need count > 1")
    return np.asarray(m2, dtype=np.float64) / float(count)


def epsilon_rule(covariance: np.ndarray) -> float:
    """Fixed numerical stabilization: epsilon = 1e-6 * trace(cov) / d."""
    d = int(covariance.shape[0])
    return EPSILON_FACTOR * float(np.trace(covariance)) / d


# --- ZCA whitening -----------------------------------------------------------

def zca_whitening_matrix(covariance_reg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric ZCA whitening W = Q Lambda^-1/2 Q^T from eigh(covariance_reg).

    Returns (W, eigenvalues) where eigenvalues are in ascending order.
    Raises ValueError on non-symmetric/non-finite/negative-definite input.
    """
    cov = np.asarray(covariance_reg, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite values")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise ValueError("eigendecomposition produced non-finite values")
    if np.any(eigenvalues <= 0.0):
        raise ValueError(f"non-positive eigenvalue after regularization: min={eigenvalues.min():.3e}")
    w = eigenvectors @ (np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T)
    return w, eigenvalues


def whiten(x: np.ndarray, mean: np.ndarray, whitening_matrix: np.ndarray) -> np.ndarray:
    """Apply train-normal centering + ZCA whitening: (x - mean) @ W."""
    return (np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) @ np.asarray(
        whitening_matrix, dtype=np.float64
    )


# --- numerical sanity gate ---------------------------------------------------

def whitening_sanity(whitened_sample: np.ndarray) -> dict:
    """Report whitened mean/covariance deviation from (0, I) over a sample.

    BLOCK triggers (checked by the caller): non-finite values; or the whitened
    covariance diagonal median outside [0.5, 1.5] (i.e. transform wrong by >~40%);
    or off-diagonal absolute max > 0.25 (gross non-identity). These bounds only
    fire on genuine numerical failure — a healthy ZCA shows diag ~1 and off-diag
    max ~0.02 on a ~45k-token sample.
    """
    x = np.asarray(whitened_sample, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("whitened_sample must be (n, d) with n > 0")
    n_finite = int(np.isfinite(x).all(axis=1).sum())
    col_mean = x.mean(axis=0)
    centered = x - col_mean
    cov = (centered.T @ centered) / x.shape[0]
    diag = np.diag(cov)
    off = np.abs(cov - np.diag(diag))
    off_flat = off[np.triu_indices_from(off, k=1)]
    return {
        "sample_size": int(x.shape[0]),
        "non_finite_count": int(x.shape[0] - n_finite),
        "max_abs_mean": float(np.max(np.abs(col_mean))),
        "mean_abs_mean": float(np.mean(np.abs(col_mean))),
        "cov_diag_min": float(diag.min()),
        "cov_diag_p50": float(np.median(diag)),
        "cov_diag_p95": float(np.percentile(diag, 95.0)),
        "cov_diag_max": float(diag.max()),
        "cov_offdiag_abs_mean": float(np.mean(off_flat)) if off_flat.size else 0.0,
        "cov_offdiag_abs_p95": float(np.percentile(off_flat, 95.0)) if off_flat.size else 0.0,
        "cov_offdiag_abs_p99": float(np.percentile(off_flat, 99.0)) if off_flat.size else 0.0,
        "cov_offdiag_abs_max": float(np.max(off_flat)) if off_flat.size else 0.0,
    }


def whitening_numerical_healthy(sanity: dict, eigenvalues: np.ndarray) -> tuple[bool, str]:
    """Return (healthy, reason). BLOCK on non-finite / non-positive eigenvalues /
    gross non-identity (documented bounds above)."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    if not np.all(np.isfinite(ev)) or ev.size == 0:
        return False, "non-finite or empty eigenvalues"
    if np.any(ev <= 0.0):
        return False, "non-positive eigenvalue after regularization"
    cond = float(ev.max() / ev.min())
    if not math.isfinite(cond):
        return False, "catastrophic (non-finite) condition number"
    for key in ("max_abs_mean", "cov_diag_p50", "cov_offdiag_abs_max"):
        if not math.isfinite(sanity.get(key, math.nan)):
            return False, f"non-finite sanity metric {key}"
    if not (0.5 <= sanity["cov_diag_p50"] <= 1.5):
        return False, "whitened covariance diagonal median outside [0.5, 1.5]"
    if sanity["cov_offdiag_abs_max"] > 0.25:
        return False, "whitened off-diagonal absolute max > 0.25"
    return True, "ok"


# --- adaptation gate ---------------------------------------------------------

def adaptation_gate_passed(d2_auroc: float, d3_auroc: float) -> bool:
    """D3 AUROC >= 0.75 AND D3 - D2 >= +0.05."""
    if not (_finite(d2_auroc) and _finite(d3_auroc)):
        return False
    return (
        d3_auroc >= ADAPTATION_GATE["auroc_min"]
        and (d3_auroc - d2_auroc) >= ADAPTATION_GATE["delta_vs_d2"]
    )


def small_defect_adaptation_signal(d3_q1: float, d2_q1: float) -> bool:
    """Secondary: D3 Q1 - D2 Q1 >= +0.05. Never substitutes the Primary Gate."""
    if not (_finite(d3_q1) and _finite(d2_q1)):
        return False
    return (d3_q1 - d2_q1) >= ADAPTATION_GATE["small_defect_q1_delta"]


def adaptation_strong_signal(d3_auroc: float) -> bool:
    return _finite(d3_auroc) and d3_auroc >= ADAPTATION_GATE["strong_auroc"]


__all__ = [
    "ADAPTATION_BANK_BUDGET",
    "ADAPTATION_GATE",
    "ADAPTATION_PROTOCOL_VERSION",
    "ADAPTATION_SEED",
    "D0_AUROC",
    "D0_QUARTILES",
    "D1_AUROC",
    "D1_QUARTILES",
    "D2_AUROC",
    "D2_CANDIDATE_ID",
    "D2_QUARTILES",
    "D2_REFERENCE",
    "D3_CANDIDATE_ID",
    "EPSILON_FACTOR",
    "adaptation_gate_passed",
    "adaptation_strong_signal",
    "chan_update_batch",
    "covariance_from_stats",
    "epsilon_rule",
    "serialize_reference",
    "small_defect_adaptation_signal",
    "whiten",
    "whitening_numerical_healthy",
    "whitening_sanity",
    "zca_whitening_matrix",
]