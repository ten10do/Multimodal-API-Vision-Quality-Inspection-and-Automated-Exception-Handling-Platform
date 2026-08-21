"""Operational qualification primitives for the frozen D3 candidate.

This module deliberately contains no model training or threshold-selection code.
It provides report validation, the frozen pixel-localization metrics, fail-closed
monitoring, and candidate-only rollback state.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, TypeVar

import numpy as np

from steel_patchcore.candidate_registry import (
    ALLOWED_STATUS,
    CandidateRegistry,
    CandidateRegistryError,
    sha256_file,
    validate_manifest,
)

PERFORMANCE_SCHEMA_VERSION = "steel_patchcore_d3_performance_v1"
SHADOW_SCHEMA_VERSION = "steel_patchcore_d3_shadow_log_v1"
HEATMAP_SCHEMA_VERSION = "steel_patchcore_d3_heatmap_validation_v1"
MONITORING_SCHEMA_VERSION = "steel_patchcore_d3_monitoring_v1"
ROLLBACK_SCHEMA_VERSION = "steel_patchcore_d3_rollback_v1"
BENCHMARK_BATCH_SIZES = (1, 10, 100, 1000)
SHADOW_REQUIRED_FIELDS = {
    "timestamp",
    "image_id",
    "model_version",
    "artifact_version",
    "score",
    "heatmap_path",
}


class OperationalQualificationError(RuntimeError):
    """A fail-closed operational qualification violation."""


def atomic_write_json(path: Path, payload: Mapping | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def latency_percentiles(values_ms: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise OperationalQualificationError("INVALID_LATENCY_SAMPLES")
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def validate_performance_report(report: Mapping) -> None:
    if report.get("schema_version") != PERFORMANCE_SCHEMA_VERSION:
        raise OperationalQualificationError("PERFORMANCE_SCHEMA_MISMATCH")
    if report.get("candidate_status") != ALLOWED_STATUS or report.get("production_promotion") is not False:
        raise OperationalQualificationError("PERFORMANCE_CANDIDATE_ONLY_REQUIRED")
    batches = report.get("benchmarks")
    if not isinstance(batches, list) or [row.get("image_count") for row in batches] != list(BENCHMARK_BATCH_SIZES):
        raise OperationalQualificationError("PERFORMANCE_BATCH_SET_MISMATCH")
    for row in batches:
        latency = row.get("latency_ms", {})
        resources = row.get("resources", {})
        if set(latency) != {"p50_ms", "p95_ms", "p99_ms"}:
            raise OperationalQualificationError("PERFORMANCE_LATENCY_SCHEMA_MISMATCH")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in latency.values()):
            raise OperationalQualificationError("PERFORMANCE_LATENCY_INVALID")
        required_resources = {"cpu_memory_peak_mb", "gpu_memory_peak_mb", "gpu_utilization_percent"}
        if not required_resources <= set(resources):
            raise OperationalQualificationError("PERFORMANCE_RESOURCE_SCHEMA_MISMATCH")
        if not all(
            resources[key] is None
            or (isinstance(resources[key], (int, float)) and math.isfinite(resources[key]) and resources[key] >= 0)
            for key in required_resources
        ):
            raise OperationalQualificationError("PERFORMANCE_RESOURCE_INVALID")


def validate_shadow_record(record: Mapping) -> None:
    if not SHADOW_REQUIRED_FIELDS <= set(record):
        raise OperationalQualificationError("SHADOW_RECORD_FIELD_MISSING")
    if not all(isinstance(record[key], str) and record[key] for key in SHADOW_REQUIRED_FIELDS - {"score"}):
        raise OperationalQualificationError("SHADOW_RECORD_STRING_INVALID")
    if not isinstance(record["score"], (int, float)) or not math.isfinite(record["score"]):
        raise OperationalQualificationError("SHADOW_RECORD_SCORE_INVALID")


def pixel_localization_metrics(anomaly_map: np.ndarray, mask: np.ndarray, max_fpr: float = 0.3) -> dict[str, float]:
    """Frozen per-image metrics matching eval_steel_patchcore.py semantics."""
    scores = np.asarray(anomaly_map, dtype=np.float64)
    labels = np.asarray(mask) > 0
    if scores.shape != labels.shape or scores.ndim != 2 or not np.isfinite(scores).all():
        raise OperationalQualificationError("PIXEL_INPUT_INVALID")
    foreground = scores[labels]
    background = scores[~labels]
    if not foreground.size or not background.size:
        raise OperationalQualificationError("PIXEL_EVIDENCE_REQUIRES_FOREGROUND_AND_BACKGROUND")
    sorted_background = np.sort(background)
    below = np.searchsorted(sorted_background, foreground, side="left")
    at_or_below = np.searchsorted(sorted_background, foreground, side="right")
    pixel_auroc = float(np.sum(below + 0.5 * (at_or_below - below)) / (foreground.size * background.size))
    fprs = np.linspace(0.0, max_fpr, 101)
    positions = (sorted_background.size - 1) * (1.0 - fprs)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    fractions = positions - lower
    thresholds = sorted_background[lower] + fractions * (sorted_background[upper] - sorted_background[lower])
    recalls = np.asarray([(foreground > threshold).mean() for threshold in thresholds], dtype=np.float64)
    aupro = float(np.trapezoid(recalls, fprs) / max_fpr)
    return {"pixel_auroc": pixel_auroc, "aupro": aupro}


def validate_heatmap_report(report: Mapping) -> None:
    if report.get("schema_version") != HEATMAP_SCHEMA_VERSION:
        raise OperationalQualificationError("HEATMAP_SCHEMA_MISMATCH")
    if report.get("image_score_gate_changed") is not False:
        raise OperationalQualificationError("IMAGE_SCORE_GATE_CHANGE_FORBIDDEN")
    metrics = report.get("metrics", {})
    for key in ("pixel_auroc_mean_per_image", "aupro_mean_per_image"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise OperationalQualificationError(f"HEATMAP_METRIC_INVALID:{key}")
    comparison = report.get("paired_baseline_comparison", {})
    if comparison.get("sample_count") != report.get("sample_count"):
        raise OperationalQualificationError("HEATMAP_BASELINE_PAIRING_MISMATCH")


@dataclass
class InferenceMonitor:
    """In-process counters and a fail-closed readiness latch."""

    artifact_hashes: dict[str, str] = field(default_factory=dict)
    request_count: int = 0
    error_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    gpu_memory_mb: list[float] = field(default_factory=list)
    ready: bool = False
    closed_reason: str | None = None

    def start(
        self,
        verify_artifact: Callable[[], object],
        load_model: Callable[[], object],
    ) -> object:
        try:
            verification = verify_artifact()
            if not bool(getattr(verification, "passed", False)):
                errors = tuple(getattr(verification, "errors", ()))
                raise OperationalQualificationError(f"ARTIFACT_VERIFICATION_FAILED:{errors}")
            self.artifact_hashes = dict(getattr(verification, "hashes", {}))
            model = load_model()
        except Exception as exc:
            self.ready = False
            self.closed_reason = f"{type(exc).__name__}:{exc}"
            raise OperationalQualificationError(f"MONITOR_FAIL_CLOSED:{self.closed_reason}") from exc
        self.ready = True
        self.closed_reason = None
        return model

    def execute(self, operation: Callable[[], object], *, gpu_memory_mb: Callable[[], float | None] | None = None):
        if not self.ready:
            raise OperationalQualificationError(f"MONITOR_NOT_READY:{self.closed_reason or 'STARTUP_REQUIRED'}")
        self.request_count += 1
        started = time.perf_counter()
        try:
            result = operation()
        except Exception:
            self.error_count += 1
            raise
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
            value = gpu_memory_mb() if gpu_memory_mb is not None else None
            if value is not None and math.isfinite(value) and value >= 0:
                self.gpu_memory_mb.append(float(value))
        return result

    def snapshot(self) -> dict:
        latency = latency_percentiles(self.latencies_ms) if self.latencies_ms else None
        return {
            "schema_version": MONITORING_SCHEMA_VERSION,
            "ready": self.ready,
            "closed_reason": self.closed_reason,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count else 0.0,
            "latency_ms": latency,
            "gpu_memory_peak_mb": max(self.gpu_memory_mb) if self.gpu_memory_mb else None,
            "artifact_hashes": dict(self.artifact_hashes),
        }


def _verified_candidate_entry(manifest_path: Path, registry: CandidateRegistry) -> dict:
    path = manifest_path.resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationalQualificationError(f"ROLLBACK_MANIFEST_UNREADABLE:{path}") from exc
    try:
        validate_manifest(manifest)
        verification = registry.verify_artifact(manifest)
    except CandidateRegistryError as exc:
        raise OperationalQualificationError(f"ROLLBACK_MANIFEST_INVALID:{exc}") from exc
    if not verification.passed:
        raise OperationalQualificationError(f"ROLLBACK_ARTIFACT_INVALID:{verification.errors}")
    if manifest["status"] != ALLOWED_STATUS or manifest["production_promotion"] is not False:
        raise OperationalQualificationError("ROLLBACK_CANDIDATE_ONLY_REQUIRED")
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "model_name": manifest["model_name"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "artifact_hashes": verification.hashes,
    }


class CandidateRollbackManager:
    """Two-slot candidate pointer. It cannot represent production state."""

    def __init__(self, state_path: Path, registry: CandidateRegistry) -> None:
        self.state_path = state_path.resolve()
        self.registry = registry

    def load(self) -> dict:
        if not self.state_path.is_file():
            return {
                "schema_version": ROLLBACK_SCHEMA_VERSION,
                "status": "CANDIDATE_ONLY",
                "active_candidate": None,
                "previous_candidate": None,
                "automatic_production_upgrade": False,
            }
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OperationalQualificationError("ROLLBACK_STATE_UNREADABLE") from exc
        expected = {
            "schema_version": ROLLBACK_SCHEMA_VERSION,
            "status": "CANDIDATE_ONLY",
            "automatic_production_upgrade": False,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise OperationalQualificationError("ROLLBACK_STATE_INVALID")
        return state

    def activate(self, manifest_path: Path) -> dict:
        candidate = _verified_candidate_entry(manifest_path, self.registry)
        state = self.load()
        if state["active_candidate"] == candidate:
            return state
        state["previous_candidate"] = state["active_candidate"]
        state["active_candidate"] = candidate
        atomic_write_json(self.state_path, state)
        return state

    def rollback(self) -> dict:
        state = self.load()
        previous = state.get("previous_candidate")
        if not previous:
            raise OperationalQualificationError("ROLLBACK_PREVIOUS_CANDIDATE_MISSING")
        verified = _verified_candidate_entry(Path(previous["manifest_path"]), self.registry)
        if verified != previous:
            raise OperationalQualificationError("ROLLBACK_PREVIOUS_HASH_MISMATCH")
        state["active_candidate"], state["previous_candidate"] = previous, state["active_candidate"]
        atomic_write_json(self.state_path, state)
        return state


T = TypeVar("T")


def assert_reproducible(first: Mapping[str, T], second: Mapping[str, T], *, absolute_tolerance: float = 1e-6) -> None:
    if set(first) != set(second):
        raise OperationalQualificationError("REPRODUCIBILITY_ID_SET_MISMATCH")
    for image_id in first:
        a, b = first[image_id], second[image_id]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=absolute_tolerance):
                raise OperationalQualificationError(f"REPRODUCIBILITY_SCORE_MISMATCH:{image_id}")
        elif a != b:
            raise OperationalQualificationError(f"REPRODUCIBILITY_VALUE_MISMATCH:{image_id}")


__all__ = [
    "BENCHMARK_BATCH_SIZES",
    "CandidateRollbackManager",
    "HEATMAP_SCHEMA_VERSION",
    "InferenceMonitor",
    "MONITORING_SCHEMA_VERSION",
    "OperationalQualificationError",
    "PERFORMANCE_SCHEMA_VERSION",
    "ROLLBACK_SCHEMA_VERSION",
    "SHADOW_REQUIRED_FIELDS",
    "SHADOW_SCHEMA_VERSION",
    "assert_reproducible",
    "atomic_write_json",
    "latency_percentiles",
    "pixel_localization_metrics",
    "validate_heatmap_report",
    "validate_performance_report",
    "validate_shadow_record",
]
