"""In-process realtime metrics for the pipeline (3H).

Application-level statistics only. No Prometheus in Phase 3. Metrics are
computed from inspection completions (backend side) plus telemetry snapshots
pushed by the orchestrator (simulator side).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from statistics import median

logger = logging.getLogger(__name__)

MAX_LATENCY_SAMPLES = 1000
THROUGHPUT_WINDOW_SECONDS = 60.0


class RealtimeMetrics:
    """Backend-side canonical pipeline counters (Phase 4 semantics).

    completed_total is incremented on every completed inspection and must
    equal pass_total + review_total + fail_total. failed_total counts SYSTEM
    processing failures only and is shown separately from fail_total (product
    quality FAIL). captured_total / queued_current / processing_current are
    reported by the orchestrator via telemetry.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._processed = 0
        self._failed = 0
        self._pass_count = 0
        self._review_count = 0
        self._fail_count = 0
        self._latency_samples: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._inference_samples: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._completions: deque[float] = deque()
        self._started_at = time.monotonic()
        self._telemetry_updated_at: float | None = None
        # telemetry reported by the orchestrator (simulator side)
        self._telemetry: dict = {
            "captured_total": 0,
            "queued_current": 0,
            "processing_current": 0,
            "simulator_running": False,
            "simulator_interval_ms": None,
            "worker_count": None,
            "queue_size": None,
            "queue_peak_depth": 0,
        }

    async def record_completed(self, quality_result: str, latency_ms: float, inference_ms: float | None) -> None:
        async with self._lock:
            self._processed += 1
            self._latency_samples.append(latency_ms)
            if inference_ms is not None:
                self._inference_samples.append(inference_ms)
            self._completions.append(time.monotonic())
            if quality_result == "PASS":
                self._pass_count += 1
            elif quality_result == "REVIEW":
                self._review_count += 1
            else:
                self._fail_count += 1

    async def record_failed(self) -> None:
        async with self._lock:
            self._failed += 1

    async def update_telemetry(self, telemetry: dict) -> None:
        async with self._lock:
            self._telemetry.update(telemetry)
            self._telemetry_updated_at = time.time()  # wall clock for telemetry_at

    async def reset(self) -> None:
        async with self._lock:
            self._processed = 0
            self._failed = 0
            self._pass_count = 0
            self._review_count = 0
            self._fail_count = 0
            self._latency_samples.clear()
            self._inference_samples.clear()
            self._completions.clear()
            self._started_at = time.monotonic()
            self._telemetry.update({
                "captured_total": 0,
                "queued_current": 0,
                "processing_current": 0,
                "simulator_running": False,
                "simulator_interval_ms": None,
                "worker_count": None,
                "queue_size": None,
                "queue_peak_depth": 0,
            })

    async def snapshot(self) -> dict:
        from datetime import datetime, timezone

        async with self._lock:
            now = time.monotonic()
            samples = list(self._latency_samples)
            inference = list(self._inference_samples)
            recent = [t for t in self._completions if now - t <= THROUGHPUT_WINDOW_SECONDS]
            throughput = len(recent) / THROUGHPUT_WINDOW_SECONDS
            uptime = now - self._started_at
            snapshot_at = datetime.now(timezone.utc).isoformat()
            telemetry_at = (
                datetime.fromtimestamp(self._telemetry_updated_at, tz=timezone.utc).isoformat()
                if self._telemetry_updated_at is not None
                else None
            )

            # Quality / persisted facts (DB-owned, single coherent snapshot).
            # Invariant: pass + review + fail == completed; total == completed + failed.
            completed = self._processed
            failed = self._failed
            passed = self._pass_count
            reviewed = self._review_count
            failed_q = self._fail_count
            if passed + reviewed + failed_q != completed:
                logger.error(
                    "quality invariant broken: pass=%d review=%d fail=%d completed=%d",
                    passed, reviewed, failed_q, completed,
                )

            out = {
                # ---- quality / persisted facts ----
                "completed_total": completed,
                "failed_total": failed,
                "pass_total": passed,
                "review_total": reviewed,
                "fail_total": failed_q,
                "total_inspected": completed + failed,
                "yield_rate": round(passed / completed, 6) if completed else None,
                # ---- runtime telemetry (pipeline view) ----
                "captured_total": self._telemetry["captured_total"],
                "queued_current": self._telemetry["queued_current"],
                "processing_current": self._telemetry["processing_current"],
                "queue_depth": self._telemetry["queued_current"],
                "throughput": round(throughput, 3),
                "queue_peak_depth": self._telemetry["queue_peak_depth"],
                "simulator_running": self._telemetry["simulator_running"],
                "simulator_interval_ms": self._telemetry["simulator_interval_ms"],
                "worker_count": self._telemetry["worker_count"],
                "queue_size": self._telemetry["queue_size"],
                # ---- timestamps / freshness ----
                "snapshot_at": snapshot_at,
                "telemetry_at": telemetry_at,
                "current_throughput": round(throughput, 3),
                "average_processing_latency_ms": round(sum(samples) / len(samples), 2) if samples else None,
                "p50_latency_ms": round(median(samples), 2) if samples else None,
                "p95_latency_ms": round(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)], 2) if samples else None,
                "average_inference_latency_ms": round(sum(inference) / len(inference), 2) if inference else None,
                "uptime_seconds": round(uptime, 1),
                "ws_client_count": 0,  # patched by the status endpoint
            }
            return out


metrics = RealtimeMetrics()
