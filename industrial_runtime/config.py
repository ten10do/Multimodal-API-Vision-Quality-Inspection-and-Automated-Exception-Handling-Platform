"""Edge runtime configuration (config-driven, no hardcoded values).

Defaults live in the packaged ``edge_config.yaml``; a deployment overrides via
the ``INDUSTRIAL_EDGE_CONFIG`` environment variable or an explicit path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "edge_config.yaml"
ENV_CONFIG_PATH = "INDUSTRIAL_EDGE_CONFIG"

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
VALID_DEVICES = {"cuda", "cpu"}


@dataclass(frozen=True)
class EdgeConfig:
    # runtime
    device: str = "cuda"
    batch_size: int = 1
    timeout_ms: int = 3000
    # logging
    log_level: str = "INFO"
    # monitoring
    monitoring_interval_seconds: float = 5.0
    # drift thresholds (Phase 2)
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    cosine_warning: float = 0.05
    cosine_critical: float = 0.20
    mean_dist_warning: float = 0.30
    mean_dist_critical: float = 1.00
    min_baseline_samples: int = 200
    source_path: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {sorted(VALID_DEVICES)}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        if self.log_level not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(VALID_LOG_LEVELS)}")
        if self.monitoring_interval_seconds <= 0:
            raise ValueError("monitoring.interval_seconds must be > 0")
        for name in ("psi", "cosine", "mean_dist"):
            warning = getattr(self, f"{name}_warning")
            critical = getattr(self, f"{name}_critical")
            if not 0.0 < warning < critical:
                raise ValueError(f"{name} thresholds must satisfy 0 < warning < critical")

    @classmethod
    def from_mapping(cls, payload: dict, *, source_path: str | None = None) -> "EdgeConfig":
        runtime = dict(payload.get("runtime") or {})
        logging_ = dict(payload.get("logging") or {})
        monitoring = dict(payload.get("monitoring") or {})
        drift = dict(payload.get("drift") or {})
        return cls(
            device=str(runtime.get("device", cls.__dataclass_fields__["device"].default)),
            batch_size=int(runtime.get("batch_size", 1)),
            timeout_ms=int(runtime.get("timeout_ms", 3000)),
            log_level=str(logging_.get("level", "INFO")).upper(),
            monitoring_interval_seconds=float(monitoring.get("interval_seconds", 5.0)),
            psi_warning=float(drift.get("psi_warning", 0.10)),
            psi_critical=float(drift.get("psi_critical", 0.25)),
            cosine_warning=float(drift.get("cosine_warning", 0.05)),
            cosine_critical=float(drift.get("cosine_critical", 0.20)),
            mean_dist_warning=float(drift.get("mean_dist_warning", 0.30)),
            mean_dist_critical=float(drift.get("mean_dist_critical", 1.00)),
            min_baseline_samples=int(drift.get("min_baseline_samples", 200)),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EdgeConfig":
        resolved = Path(path or os.environ.get(ENV_CONFIG_PATH) or DEFAULT_CONFIG_PATH)
        payload = {}
        if resolved.exists():
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        return cls.from_mapping(payload, source_path=str(resolved))

    def summary(self) -> dict:
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "timeout_ms": self.timeout_ms,
            "log_level": self.log_level,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "drift_thresholds": {
                "psi_warning": self.psi_warning,
                "psi_critical": self.psi_critical,
                "cosine_warning": self.cosine_warning,
                "cosine_critical": self.cosine_critical,
                "mean_dist_warning": self.mean_dist_warning,
                "mean_dist_critical": self.mean_dist_critical,
                "min_baseline_samples": self.min_baseline_samples,
            },
            "source_path": self.source_path,
        }


def load_edge_config(path: str | Path | None = None) -> EdgeConfig:
    return EdgeConfig.load(path)
