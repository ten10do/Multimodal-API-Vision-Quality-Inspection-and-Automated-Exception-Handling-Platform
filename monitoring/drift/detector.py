"""Drift detector: NORMAL / WARNING / CRITICAL with configurable thresholds.

The detector NEVER modifies the model. Its only production action is the
fail-safe routing: CRITICAL drift must make subsequent inspections HOLD with
reason DATA_DISTRIBUTION_SHIFT (via the decision layer); WARNING keeps
production running and raises alerts.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from industrial_loop.events import utc_now_iso
from industrial_runtime.config import EdgeConfig

from .collector import BaselineStats, FeatureDriftCollector
from .metrics import (
    cosine_distribution_shift,
    embedding_mean_distance,
    psi_embedding_from_stats,
)

MIN_EVALUATION_SAMPLES = 30


class DriftState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


_SEVERITY = {DriftState.NORMAL: 0, DriftState.WARNING: 1, DriftState.CRITICAL: 2}


@dataclass(frozen=True)
class DriftThresholds:
    """Configurable bands (defaults follow the spec's PSI example)."""

    psi_warning: float = 0.10
    psi_critical: float = 0.25
    cosine_warning: float = 0.05
    cosine_critical: float = 0.20
    mean_dist_warning: float = 0.30
    mean_dist_critical: float = 1.00

    def __post_init__(self) -> None:
        for name in ("psi", "cosine", "mean_dist"):
            warning = getattr(self, f"{name}_warning")
            critical = getattr(self, f"{name}_critical")
            if not 0.0 < warning < critical:
                raise ValueError(f"{name} thresholds must satisfy 0 < warning < critical")

    @classmethod
    def from_config(cls, config: EdgeConfig) -> "DriftThresholds":
        return cls(
            psi_warning=config.psi_warning,
            psi_critical=config.psi_critical,
            cosine_warning=config.cosine_warning,
            cosine_critical=config.cosine_critical,
            mean_dist_warning=config.mean_dist_warning,
            mean_dist_critical=config.mean_dist_critical,
        )

    def as_dict(self) -> dict:
        return {
            "psi_warning": self.psi_warning,
            "psi_critical": self.psi_critical,
            "cosine_warning": self.cosine_warning,
            "cosine_critical": self.cosine_critical,
            "mean_dist_warning": self.mean_dist_warning,
            "mean_dist_critical": self.mean_dist_critical,
        }


@dataclass
class DriftReport:
    state: DriftState
    psi_mean: float
    psi_max: float
    cosine_shift: float
    mean_distance: float
    checks: dict[str, str]
    n_baseline: int
    n_current: int
    timestamp: str = field(default_factory=utc_now_iso)
    sufficient_data: bool = True
    alerts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "psi_mean": round(self.psi_mean, 6),
            "psi_max": round(self.psi_max, 6),
            "cosine_shift": round(self.cosine_shift, 6),
            "mean_distance": round(self.mean_distance, 6),
            "checks": dict(self.checks),
            "n_baseline": self.n_baseline,
            "n_current": self.n_current,
            "timestamp": self.timestamp,
            "sufficient_data": self.sufficient_data,
            "alerts": list(self.alerts),
        }


def _band(value: float, warning: float, critical: float) -> DriftState:
    if value >= critical:
        return DriftState.CRITICAL
    if value >= warning:
        return DriftState.WARNING
    return DriftState.NORMAL


def _worst(states: list[DriftState]) -> DriftState:
    return max(states, key=lambda s: _SEVERITY[s])


class DriftDetector:
    """Evaluates a collector's current window against its frozen baseline."""

    def __init__(
        self,
        thresholds: DriftThresholds | None = None,
        *,
        bins: int = 10,
        history_size: int = 50,
    ) -> None:
        self.thresholds = thresholds or DriftThresholds()
        self.bins = int(bins)
        self._history: deque[DriftReport] = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def evaluate(self, collector: FeatureDriftCollector) -> DriftReport:
        if not collector.has_baseline:
            raise ValueError("collector has no baseline; call set_baseline() first")
        baseline: BaselineStats = collector.baseline  # type: ignore[assignment]
        current = collector.current_stats()
        n_current = current.count if current else 0
        if current is None or n_current < MIN_EVALUATION_SAMPLES:
            report = DriftReport(
                state=DriftState.NORMAL,
                psi_mean=0.0,
                psi_max=0.0,
                cosine_shift=0.0,
                mean_distance=0.0,
                checks={"data": f"INSUFFICIENT({n_current}<{MIN_EVALUATION_SAMPLES})"},
                n_baseline=baseline.count,
                n_current=n_current,
                sufficient_data=False,
            )
            with self._lock:
                self._history.append(report)
            return report

        psi = psi_embedding_from_stats(
            baseline.mean,
            baseline.std,
            current.mean,
            current.std,
            bins=self.bins,
        )
        cosine_shift = cosine_distribution_shift(
            collector.baseline_samples(), collector.current_window()
        )
        mean_distance = embedding_mean_distance(baseline.mean, baseline.std, current.mean)

        checks = {
            "psi": _band(psi["mean"], self.thresholds.psi_warning, self.thresholds.psi_critical).value,
            "cosine_shift": _band(
                cosine_shift, self.thresholds.cosine_warning, self.thresholds.cosine_critical
            ).value,
            "mean_distance": _band(
                mean_distance,
                self.thresholds.mean_dist_warning,
                self.thresholds.mean_dist_critical,
            ).value,
        }
        states = [_band(psi["mean"], self.thresholds.psi_warning, self.thresholds.psi_critical)]
        states.append(
            _band(cosine_shift, self.thresholds.cosine_warning, self.thresholds.cosine_critical)
        )
        states.append(
            _band(mean_distance, self.thresholds.mean_dist_warning, self.thresholds.mean_dist_critical)
        )
        state = _worst(states)
        # psi_max is reported for transparency but excluded from the verdict:
        # a max over hundreds of noisy per-dimension estimates inflates far
        # beyond the mean and would make the bands dimension-count-dependent.
        values = {
            "psi": psi["mean"],
            "cosine_shift": cosine_shift,
            "mean_distance": mean_distance,
        }
        alerts = [
            f"{name}={values[name]:.4f} entered {band}"
            for name, band in checks.items()
            if band != DriftState.NORMAL.value
        ]
        report = DriftReport(
            state=state,
            psi_mean=psi["mean"],
            psi_max=psi["max"],
            cosine_shift=cosine_shift,
            mean_distance=mean_distance,
            checks=checks,
            n_baseline=baseline.count,
            n_current=n_current,
            alerts=alerts,
        )
        with self._lock:
            self._history.append(report)
        return report

    # -- history -----------------------------------------------------------------

    def history(self) -> list[DriftReport]:
        with self._lock:
            return list(self._history)

    def latest(self) -> DriftReport | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def reset_history(self) -> None:
        with self._lock:
            self._history.clear()

    def status(self) -> dict:
        latest = self.latest()
        return {
            "state": latest.state.value if latest else None,
            "thresholds": self.thresholds.as_dict(),
            "bins": self.bins,
            "evaluations": len(self._history),
            "latest": latest.as_dict() if latest else None,
        }
