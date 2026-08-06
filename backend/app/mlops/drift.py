"""Drift detection (8I): simple, explainable distribution comparisons.

Baseline window vs current window on: confidence, defect type distribution,
anomaly score, review rate. Uses PSI / KS / distribution delta. Outputs
NORMAL / WARNING / CRITICAL.

Data drift is NOT model performance degradation: without ground truth we can
only call it drift. Quality degradation requires Human Review ground truth.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

DriftLevel = Literal["NORMAL", "WARNING", "CRITICAL"]


def _expected_bins(values: Sequence[float], bins: int, lo: float, hi: float) -> list[float]:
    """Distribution of `values` over fixed bins (with clamping)."""
    out = [0.0] * bins
    if not values:
        return out
    span = (hi - lo) or 1.0
    for v in values:
        if not isinstance(v, (int, float)) or math.isnan(v):
            continue
        idx = int((v - lo) / span * bins)
        idx = max(0, min(bins - 1, idx))
        out[idx] += 1.0
    total = sum(out) or 1.0
    return [c / total for c in out]


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10,
        lo: float = 0.0, hi: float = 1.0) -> float:
    """Population Stability Index between two distributions."""
    e = _expected_bins(expected, bins, lo, hi)
    a = _expected_bins(actual, bins, lo, hi)
    score = 0.0
    for ei, ai in zip(e, a):
        ei = min(max(ei, 1e-6), 1.0)
        ai = min(max(ai, 1e-6), 1.0)
        score += (ai - ei) * math.log(ai / ei)
    return score


def ks_statistic(expected: Sequence[float], actual: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap).

    Standard two-sample KS over the empirical CDFs; identical samples give
    ~0, completely separated samples give ~1.
    """
    if not expected or not actual:
        return 0.0
    a = sorted(expected)
    b = sorted(actual)
    i = j = 0
    n1, n2 = len(a), len(b)
    max_gap = 0.0
    while i < n1 and j < n2:
        # on ties advance BOTH pointers so equal distributions give ~0
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
        gap = abs(i / n1 - j / n2)
        max_gap = max(max_gap, gap)
    return max_gap


def review_rate_delta(baseline_rate: float, current_rate: float, threshold: float = 0.15) -> DriftLevel:
    d = abs(current_rate - baseline_rate)
    if d <= threshold:
        return "NORMAL"
    return "WARNING" if d <= 2 * threshold else "CRITICAL"


def classify_psi(score: float) -> DriftLevel:
    if score < 0.1:
        return "NORMAL"
    if score < 0.25:
        return "WARNING"
    return "CRITICAL"


def classify_ks(stat: float) -> DriftLevel:
    if stat < 0.1:
        return "NORMAL"
    if stat < 0.3:
        return "WARNING"
    return "CRITICAL"


def defect_distribution_delta(baseline: dict[str, float], current: dict[str, float],
                              threshold: float = 0.15) -> DriftLevel:
    """Max relative share change across defect types."""
    keys = set(baseline) | set(current)
    max_delta = 0.0
    for k in keys:
        b = baseline.get(k, 0.0)
        c = current.get(k, 0.0)
        max_delta = max(max_delta, abs(c - b))
    if max_delta <= threshold:
        return "NORMAL"
    return "WARNING" if max_delta <= 2 * threshold else "CRITICAL"
