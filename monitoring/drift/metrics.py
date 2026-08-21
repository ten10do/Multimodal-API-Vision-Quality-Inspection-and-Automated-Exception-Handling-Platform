"""Drift metrics: PSI, cosine distribution shift, embedding mean distance."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

EPS = 1e-6


# --- Population Stability Index -------------------------------------------------


def _proportions_from_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges (ascending, interior only) -> (-inf, edges..., +inf) props."""
    lower = np.concatenate(([-np.inf], edges))
    upper = np.concatenate((edges, [np.inf]))
    return lower, upper


def psi_1d(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = EPS) -> float:
    """Empirical Population Stability Index between two 1-D samples.

    Bin edges are the expected sample's quantiles (deciles by default), the
    classic PSI construction. Returns a non-negative float; identical
    distributions give ~0.
    """
    expected = np.asarray(expected, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    if expected.size == 0 or actual.size == 0:
        raise ValueError("psi_1d requires non-empty samples")
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size == 0:  # degenerate constant sample
        edges = np.array([float(expected[0])])
    lower, upper = _proportions_from_edges(edges)
    e_props = np.array(
        [np.mean((expected >= lo) & (expected < hi)) for lo, hi in zip(lower, upper)]
    )
    a_props = np.array([np.mean((actual >= lo) & (actual < hi)) for lo, hi in zip(lower, upper)])
    e_props = np.clip(e_props / e_props.sum(), eps, None)
    a_props = np.clip(a_props / a_props.sum(), eps, None)
    return float(np.sum((a_props - e_props) * np.log(a_props / e_props)))


def psi_from_stats(
    expected_mean: float,
    expected_std: float,
    actual_mean: float,
    actual_std: float,
    bins: int = 10,
    eps: float = EPS,
) -> float:
    """Analytic PSI between two normal distributions N(mu, sigma).

    Equivalent to the empirical PSI for large samples, but computable purely
    from baseline/current statistics (mean embedding + variance + count).
    """
    if expected_std <= 0 or actual_std <= 0:
        raise ValueError("std must be positive")
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    edges = sps.norm.ppf(quantiles, loc=expected_mean, scale=expected_std)
    lower, upper = _proportions_from_edges(edges)
    e_props = sps.norm.cdf(upper, loc=expected_mean, scale=expected_std) - sps.norm.cdf(
        lower, loc=expected_mean, scale=expected_std
    )
    a_props = sps.norm.cdf(upper, loc=actual_mean, scale=actual_std) - sps.norm.cdf(
        lower, loc=actual_mean, scale=actual_std
    )
    e_props = np.clip(e_props / e_props.sum(), eps, None)
    a_props = np.clip(a_props / a_props.sum(), eps, None)
    return float(np.sum((a_props - e_props) * np.log(a_props / e_props)))


def psi_embedding(
    baseline: np.ndarray, current: np.ndarray, bins: int = 10
) -> dict[str, float | list]:
    """Per-dimension empirical PSI across an embedding matrix.

    Returns {"mean", "max", "per_dim"}; the detector evaluates both the mean
    (headline) and max (worst dimension) against configurable bands.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if baseline.ndim != 2 or current.ndim != 2:
        raise ValueError("embedding matrices must be 2-D")
    if baseline.shape[1] != current.shape[1]:
        raise ValueError("embedding dimension mismatch")
    per_dim = [
        psi_1d(baseline[:, d], current[:, d], bins=bins) for d in range(baseline.shape[1])
    ]
    arr = np.asarray(per_dim)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "per_dim": per_dim}


def psi_embedding_from_stats(
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    current_mean: np.ndarray,
    current_std: np.ndarray,
    bins: int = 10,
) -> dict[str, float]:
    """Vectorized analytic per-dimension PSI from summary statistics."""
    baseline_mean = np.asarray(baseline_mean, dtype=np.float64)
    baseline_std = np.asarray(baseline_std, dtype=np.float64)
    current_mean = np.asarray(current_mean, dtype=np.float64)
    current_std = np.asarray(current_std, dtype=np.float64)
    if not (baseline_mean.shape == baseline_std.shape == current_mean.shape == current_std.shape):
        raise ValueError("statistic vectors must share one shape")
    if np.any(baseline_std <= 0) or np.any(current_std <= 0):
        raise ValueError("all std values must be positive")
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    # (bins-1, D) interior edges from the baseline distribution
    edges = sps.norm.ppf(quantiles[:, None], loc=baseline_mean[None, :], scale=baseline_std[None, :])
    lower = np.concatenate((-np.inf * np.ones((1, baseline_mean.size)), edges), axis=0)
    upper = np.concatenate((edges, np.inf * np.ones((1, baseline_mean.size))), axis=0)
    e_cdf_hi = sps.norm.cdf(upper, loc=baseline_mean[None, :], scale=baseline_std[None, :])
    e_cdf_lo = sps.norm.cdf(lower, loc=baseline_mean[None, :], scale=baseline_std[None, :])
    a_cdf_hi = sps.norm.cdf(upper, loc=current_mean[None, :], scale=current_std[None, :])
    a_cdf_lo = sps.norm.cdf(lower, loc=current_mean[None, :], scale=current_std[None, :])
    e_props = np.clip(e_cdf_hi - e_cdf_lo, EPS, None)
    a_props = np.clip(a_cdf_hi - a_cdf_lo, EPS, None)
    psi_dims = np.sum((a_props - e_props) * np.log(a_props / e_props), axis=0)
    return {"mean": float(psi_dims.mean()), "max": float(psi_dims.max())}


# --- cosine distribution shift ----------------------------------------------------


def cosine_distribution_shift(baseline: np.ndarray, current: np.ndarray) -> float:
    """|coherence(reference) - coherence(current)| in [0, 2].

    Coherence = mean cos(x_i, mu). To avoid the in-sample bias (a mean fitted
    on the same rows it is evaluated against inflates coherence), the baseline
    is split in halves: mu is fitted on one half, the reference coherence is
    measured on the other, and the current batch is measured against the same
    mu - making the two estimates directly comparable.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if baseline.ndim != 2 or current.ndim != 2 or baseline.shape[1] != current.shape[1]:
        raise ValueError("embedding matrices must be 2-D with matching dims")
    if baseline.shape[0] < 2:
        raise ValueError("baseline needs >= 2 rows for a split-half reference")
    half = baseline.shape[0] // 2
    fit_half, ref_half = baseline[:half], baseline[half:]
    mu = fit_half.mean(axis=0)
    norm_mu = np.linalg.norm(mu)
    if norm_mu < EPS:
        raise ValueError("baseline mean is degenerate; cosine shift undefined")

    def _coherence(matrix: np.ndarray) -> float:
        norms = np.linalg.norm(matrix, axis=1)
        norms = np.where(norms < EPS, EPS, norms)
        return float(np.mean((matrix @ mu) / (norms * norm_mu)))

    return abs(_coherence(ref_half) - _coherence(current))


# --- embedding mean distance -------------------------------------------------------


def embedding_mean_distance(
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    current_mean: np.ndarray,
    current_std: np.ndarray | None = None,
) -> float:
    """RMS standardized shift of the mean embedding.

    Per dimension z_d = (mu_cur,d - mu_base,d) / sigma_base,d; the metric is
    sqrt(mean(z^2)). Pure resampling noise stays ~1/sqrt(n) (tiny), while a
    constant per-dimension shift of delta sigmas scores ~delta - stable and
    interpretable thresholds.
    """
    baseline_mean = np.asarray(baseline_mean, dtype=np.float64)
    baseline_std = np.asarray(baseline_std, dtype=np.float64)
    current_mean = np.asarray(current_mean, dtype=np.float64)
    if not (baseline_mean.shape == baseline_std.shape == current_mean.shape):
        raise ValueError("statistic vectors must share one shape")
    if np.any(baseline_std < 0):
        raise ValueError("std values must be non-negative")
    safe_std = np.where(baseline_std < EPS, 1.0, baseline_std)
    z = (current_mean - baseline_mean) / safe_std
    return float(np.sqrt(np.mean(z**2)))
