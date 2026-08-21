"""Edge resource monitoring: CPU / memory / GPU memory / latency / throughput.

``RuntimeMetrics`` is the fixed sampling record required by the spec; the
monitor keeps a bounded history and rolling request counters. GPU memory is
reported only when a CUDA device is actually available (CPU-only edge units
report ``null``), and every metric degrades gracefully when a probe is
unavailable - monitoring must never crash the runtime.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from industrial_loop.events import utc_now_iso


@dataclass(frozen=True)
class RuntimeMetrics:
    timestamp: str
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float | None
    latency_ms: float | None
    request_count: int
    error_count: int
    requests_per_second: float = 0.0

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 3),
            "memory_mb": round(self.memory_mb, 2),
            "gpu_memory_mb": round(self.gpu_memory_mb, 2) if self.gpu_memory_mb is not None else None,
            "latency_ms": round(self.latency_ms, 3) if self.latency_ms is not None else None,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "requests_per_second": round(self.requests_per_second, 4),
        }


@dataclass
class _RequestWindow:
    timestamps: deque = field(default_factory=deque)
    latencies: deque = field(default_factory=deque)
    errors: int = 0
    total: int = 0


class ResourceMonitor:
    def __init__(
        self,
        *,
        interval_seconds: float = 5.0,
        window_seconds: float = 60.0,
        max_history: int = 240,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self.interval_seconds = interval_seconds
        self.window_seconds = window_seconds
        self._history: deque[RuntimeMetrics] = deque(maxlen=max_history)
        self._window = _RequestWindow()
        self._lock = threading.Lock()
        self._process = None
        self._cpu_primed = False
        try:
            import psutil

            self._process = psutil.Process()
            psutil.cpu_percent(interval=None)  # prime the non-blocking sampler
            self._cpu_primed = True
        except Exception:  # noqa: BLE001 - monitoring is best-effort
            self._process = None

    # -- request accounting (called by the serving path) -----------------------

    def record_request(self, latency_ms: float | None = None, *, error: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            self._window.total += 1
            if error:
                self._window.errors += 1
            self._window.timestamps.append(now)
            if latency_ms is not None:
                self._window.latencies.append(float(latency_ms))
            cutoff = now - self.window_seconds
            while self._window.timestamps and self._window.timestamps[0] < cutoff:
                self._window.timestamps.popleft()
            while len(self._window.latencies) > max(1, len(self._window.timestamps)):
                self._window.latencies.popleft()

    # -- probes -----------------------------------------------------------------

    def _cpu_percent(self) -> float:
        if not self._cpu_primed:
            return 0.0
        try:
            import psutil

            return float(psutil.cpu_percent(interval=None))
        except Exception:  # noqa: BLE001
            return 0.0

    def _memory_mb(self) -> float:
        try:
            if self._process is not None:
                return float(self._process.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _gpu_memory_mb(self) -> float | None:
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                return float(total - free) / (1024.0 * 1024.0)
        except Exception:  # noqa: BLE001 - CPU-only edge unit or torch missing
            return None
        return None

    def _throughput(self) -> tuple[int, int, float]:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window_seconds
            recent = [t for t in self._window.timestamps if t >= cutoff]
            total = self._window.total
            errors = self._window.errors
        span = max(self.window_seconds, 1e-6)
        rps = len(recent) / span if recent else (total / span if total else 0.0)
        return total, errors, rps

    def _avg_latency(self) -> float | None:
        with self._lock:
            latencies = list(self._window.latencies)
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    # -- sampling ----------------------------------------------------------------

    def sample(self) -> RuntimeMetrics:
        total, errors, rps = self._throughput()
        metrics = RuntimeMetrics(
            timestamp=utc_now_iso(),
            cpu_percent=self._cpu_percent(),
            memory_mb=self._memory_mb(),
            gpu_memory_mb=self._gpu_memory_mb(),
            latency_ms=self._avg_latency(),
            request_count=total,
            error_count=errors,
            requests_per_second=rps,
        )
        self._history.append(metrics)
        return metrics

    def history(self) -> list[RuntimeMetrics]:
        return list(self._history)

    def latest(self) -> RuntimeMetrics | None:
        return self._history[-1] if self._history else None

    def snapshot(self) -> dict:
        latest = self.latest()
        return {
            "latest": latest.as_dict() if latest else None,
            "history_size": len(self._history),
            "interval_seconds": self.interval_seconds,
            "window_seconds": self.window_seconds,
        }
