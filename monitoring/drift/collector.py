"""Feature drift collector: baseline statistics + rolling current window.

The collector stores the frozen reference as raw capped samples plus summary
statistics (mean embedding, per-dimension variance, sample count) and keeps a
bounded rolling window of production embeddings for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass
class BaselineStats:
    """Summary statistics of an embedding distribution."""

    mean: np.ndarray  # (dim,)
    variance: np.ndarray  # (dim,), population variance
    count: int

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.variance.shape != self.mean.shape:
            raise ValueError("mean/variance must be 1-D vectors of equal shape")
        if self.count <= 0:
            raise ValueError("count must be positive")
        if np.any(self.variance < 0):
            raise ValueError("variance must be non-negative")

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> "BaselineStats":
        matrix = np.asarray(samples, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("samples must be a non-empty (n, dim) matrix")
        return cls(
            mean=matrix.mean(axis=0),
            variance=matrix.var(axis=0, ddof=0),
            count=int(matrix.shape[0]),
        )

    def as_dict(self) -> dict:
        return {
            "dim": self.dim,
            "count": self.count,
            "mean_norm": float(np.linalg.norm(self.mean)),
            "variance_mean": float(self.variance.mean()),
        }


class FeatureDriftCollector:
    """Collects production embeddings against a frozen baseline.

    ``window_size`` bounds memory on the edge device; when the window rolls
    over, the oldest embeddings are dropped (the baseline is never mutated).
    """

    def __init__(
        self,
        *,
        dim: int = 768,
        window_size: int = 4096,
        min_baseline_samples: int = 200,
    ) -> None:
        if dim < 1 or window_size < 1:
            raise ValueError("dim and window_size must be positive")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.min_baseline_samples = int(min_baseline_samples)
        self._baseline_samples: np.ndarray | None = None
        self.baseline: BaselineStats | None = None
        self._window: deque[np.ndarray] = deque(maxlen=window_size)
        self._total_seen = 0

    # -- baseline ----------------------------------------------------------------

    def set_baseline(self, samples: np.ndarray) -> BaselineStats:
        matrix = np.asarray(samples, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"baseline must be (n, {self.dim})")
        if matrix.shape[0] < self.min_baseline_samples:
            raise ValueError(
                f"baseline needs >= {self.min_baseline_samples} samples, got {matrix.shape[0]}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("baseline contains non-finite values")
        self._baseline_samples = matrix.copy()
        self.baseline = BaselineStats.from_samples(matrix)
        return self.baseline

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    def baseline_samples(self) -> np.ndarray:
        if self._baseline_samples is None:
            raise ValueError("baseline not set")
        return self._baseline_samples

    # -- production window ---------------------------------------------------------

    def add_sample(self, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float64).ravel()
        if vec.shape[0] != self.dim:
            raise ValueError(f"embedding dim {vec.shape[0]} != collector dim {self.dim}")
        if not np.isfinite(vec).all():
            raise ValueError("embedding contains non-finite values")
        self._window.append(vec)
        self._total_seen += 1

    def extend(self, matrix: np.ndarray) -> None:
        for row in np.asarray(matrix, dtype=np.float64):
            self.add_sample(row)

    def current_window(self) -> np.ndarray:
        if not self._window:
            return np.empty((0, self.dim), dtype=np.float64)
        return np.stack(list(self._window))

    def current_stats(self) -> BaselineStats | None:
        if not self._window:
            return None
        return BaselineStats.from_samples(self.current_window())

    @property
    def total_seen(self) -> int:
        return self._total_seen

    def reset_window(self) -> None:
        self._window.clear()

    def status(self) -> dict:
        return {
            "dim": self.dim,
            "window_size": self.window_size,
            "window_count": len(self._window),
            "total_seen": self._total_seen,
            "baseline": self.baseline.as_dict() if self.baseline else None,
            "has_baseline": self.has_baseline,
        }
