"""EdgeRuntimeManager: service lifecycle for the industrial edge runtime.

States: INIT -> STARTING -> READY -> RUNNING (or DEGRADED) -> STOPPED.

Services are registered as ``ServiceSpec`` callables (start/stop/health) so
the manager stays transport-agnostic: the camera session, the inference
client, the decision service and the PLC link are all just specs. A failing
service at start, or an unhealthy probe later, moves the runtime to DEGRADED
(production keeps running but alerts); stop()/restart() are idempotent.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .config import EdgeConfig
from .resource_monitor import ResourceMonitor


class RuntimeState(str, Enum):
    INIT = "INIT"
    STARTING = "STARTING"
    READY = "READY"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    start: Callable[[], None] | None = None
    stop: Callable[[], None] | None = None
    health: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("service name is required")


@dataclass
class _ServiceRecord:
    spec: ServiceSpec
    started: bool = False
    last_error: str | None = None


class EdgeRuntimeManager:
    def __init__(
        self,
        config: EdgeConfig | None = None,
        *,
        monitor: ResourceMonitor | None = None,
    ) -> None:
        self.config = config or EdgeConfig.load()
        self.monitor = monitor or ResourceMonitor(
            interval_seconds=self.config.monitoring_interval_seconds
        )
        self.state = RuntimeState.INIT
        self._specs: list[ServiceSpec] = []
        self._records: dict[str, _ServiceRecord] = {}
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._degraded_reasons: list[str] = []
        self._lock = threading.Lock()

    # -- registration -----------------------------------------------------------

    def register(self, spec: ServiceSpec) -> None:
        if self.state not in (RuntimeState.INIT, RuntimeState.STOPPED):
            raise RuntimeError("services must be registered before start()")
        with self._lock:
            if spec.name in self._records:
                raise ValueError(f"duplicate service name {spec.name}")
            self._specs.append(spec)
            self._records[spec.name] = _ServiceRecord(spec=spec)

    # -- lifecycle ----------------------------------------------------------------

    def start(self) -> dict:
        if self.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED):
            return self.get_status()  # idempotent
        if self.state is RuntimeState.STARTING:
            raise RuntimeError("start() already in progress")
        self.state = RuntimeState.STARTING
        self._degraded_reasons.clear()
        for spec in self._specs:
            record = self._records[spec.name]
            if spec.start is None:
                record.started = True
                continue
            try:
                spec.start()
                record.started = True
                record.last_error = None
            except Exception as exc:  # noqa: BLE001 - a failed service degrades, never crashes
                record.started = False
                record.last_error = f"{type(exc).__name__}: {exc}"
                self._degraded_reasons.append(f"{spec.name}:start_failed")
        self._started_at = time.monotonic()
        self._stopped_at = None
        health = self.health_check()
        self.state = (
            RuntimeState.RUNNING if health["overall"] == "healthy" else RuntimeState.DEGRADED
        )
        return self.get_status()

    def stop(self) -> dict:
        if self.state is RuntimeState.STOPPED:
            return self.get_status()  # idempotent
        for spec in reversed(self._specs):
            record = self._records[spec.name]
            if not record.started:
                continue
            if spec.stop is None:
                record.started = False
                continue
            try:
                spec.stop()
            except Exception as exc:  # noqa: BLE001 - shutdown continues regardless
                record.last_error = f"{type(exc).__name__}: {exc}"
            record.started = False
        self.state = RuntimeState.STOPPED
        self._stopped_at = time.monotonic()
        return self.get_status()

    def restart(self) -> dict:
        self.stop()
        return self.start()

    # -- health -------------------------------------------------------------------

    def health_check(self) -> dict:
        services: dict[str, str] = {}
        unhealthy: list[str] = []
        for spec in self._specs:
            record = self._records[spec.name]
            if not record.started:
                services[spec.name] = "stopped"
                unhealthy.append(spec.name)
                continue
            if spec.health is None:
                services[spec.name] = "healthy"
                continue
            try:
                healthy = bool(spec.health())
            except Exception as exc:  # noqa: BLE001 - probe failure == unhealthy
                healthy = False
                record.last_error = f"{type(exc).__name__}: {exc}"
            services[spec.name] = "healthy" if healthy else "unhealthy"
            if not healthy:
                unhealthy.append(spec.name)
        overall = "degraded" if unhealthy else "healthy"
        if self.state in (RuntimeState.RUNNING, RuntimeState.DEGRADED):
            self.state = (
                RuntimeState.RUNNING if overall == "healthy" else RuntimeState.DEGRADED
            )
        metrics = self.monitor.sample()  # one sample per health probe
        return {
            "overall": overall,
            "state": self.state.value,
            "services": services,
            "unhealthy": unhealthy,
            "degraded_reasons": list(self._degraded_reasons),
            "metrics": metrics.as_dict(),
        }

    # -- status ---------------------------------------------------------------------

    def get_status(self) -> dict:
        uptime = None
        if self._started_at is not None and self.state in (
            RuntimeState.RUNNING,
            RuntimeState.DEGRADED,
        ):
            uptime = round(time.monotonic() - self._started_at, 3)
        return {
            "state": self.state.value,
            "services": {
                name: {
                    "started": record.started,
                    "last_error": record.last_error,
                }
                for name, record in self._records.items()
            },
            "uptime_s": uptime,
            "config": self.config.summary(),
            "monitor": self.monitor.snapshot(),
        }

    def mark_degraded(self, reason: str) -> None:
        """External fail-safe hook (e.g. drift monitor raises an alert)."""
        if reason not in self._degraded_reasons:
            self._degraded_reasons.append(reason)
        if self.state is RuntimeState.RUNNING:
            self.state = RuntimeState.DEGRADED
