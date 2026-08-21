"""Production-readiness primitives for the immutable D3 1.3 candidate."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from steel_patchcore.candidate_registry import (
    CandidateRegistry,
    CandidateRegistryError,
    sha256_file,
)
from steel_patchcore.d3_operational import OperationalQualificationError, atomic_write_json, latency_percentiles
from steel_patchcore.dual_candidate_registry import (
    SCHEMA_VERSION as DUAL_SCHEMA_VERSION,
    DualCandidateRegistry,
)

STABILITY_SCHEMA_VERSION = "steel_patchcore_d3_24h_stability_v1"
ROBUSTNESS_SCHEMA_VERSION = "steel_patchcore_d3_input_robustness_v1"
RUNTIME_MONITOR_SCHEMA_VERSION = "steel_patchcore_d3_runtime_monitor_v1"
ROLLBACK_DRILL_SCHEMA_VERSION = "steel_patchcore_d3_dual_rollback_v1"
REVIEW_WORKFLOW_SCHEMA_VERSION = "steel_patchcore_d3_human_review_v1"
READINESS_SCHEMA_VERSION = "steel_patchcore_d3_production_readiness_v1"

MEMORY_LIMITS = {
    "cpu_growth_mb_max": 128.0,
    "cpu_slope_mb_per_hour_max": 8.0,
    "gpu_growth_mb_max": 64.0,
    "gpu_slope_mb_per_hour_max": 4.0,
}
ROBUSTNESS_GATE = {"image_auroc_min": 0.75, "pixel_auroc_min": 0.75, "aupro_min": 0.50}


def threshold_margin_confidence(score: float, threshold: float) -> dict:
    """Return an explicitly uncalibrated confidence proxy; never a probability."""
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in (score, threshold)):
        raise OperationalQualificationError("CONFIDENCE_INPUT_INVALID")
    scale = max(float(threshold), 1e-12)
    return {
        "kind": "absolute_threshold_margin_ratio",
        "value": min(abs(float(score) - float(threshold)) / scale, 1.0),
        "calibrated_probability": False,
    }


@dataclass(frozen=True)
class HumanReviewPrediction:
    image_id: str
    image_score: float
    anomaly_label: str
    heatmap_ref: str
    confidence: dict
    model_version: str
    artifact_version: str

    def as_record(self) -> dict:
        record = {
            "schema_version": REVIEW_WORKFLOW_SCHEMA_VERSION,
            "image_id": self.image_id,
            "image_score": self.image_score,
            "anomaly_label": self.anomaly_label,
            "heatmap_ref": self.heatmap_ref,
            "confidence": dict(self.confidence),
            "model_version": self.model_version,
            "artifact_version": self.artifact_version,
        }
        validate_review_prediction(record)
        return record


def validate_review_prediction(record: Mapping) -> None:
    required = {
        "schema_version", "image_id", "image_score", "anomaly_label", "heatmap_ref",
        "confidence", "model_version", "artifact_version",
    }
    if set(record) != required or record.get("schema_version") != REVIEW_WORKFLOW_SCHEMA_VERSION:
        raise OperationalQualificationError("REVIEW_PREDICTION_SCHEMA_MISMATCH")
    if record.get("anomaly_label") not in {"ANOMALY", "NORMAL"}:
        raise OperationalQualificationError("REVIEW_PREDICTION_LABEL_INVALID")
    for key in ("image_id", "heatmap_ref", "model_version", "artifact_version"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise OperationalQualificationError(f"REVIEW_PREDICTION_STRING_INVALID:{key}")
    score = record.get("image_score")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise OperationalQualificationError("REVIEW_PREDICTION_SCORE_INVALID")
    confidence = record.get("confidence", {})
    if confidence.get("kind") != "absolute_threshold_margin_ratio" or confidence.get("calibrated_probability") is not False:
        raise OperationalQualificationError("REVIEW_CONFIDENCE_SEMANTICS_INVALID")
    if not isinstance(confidence.get("value"), (int, float)) or not 0 <= confidence["value"] <= 1:
        raise OperationalQualificationError("REVIEW_CONFIDENCE_VALUE_INVALID")


def create_feedback_record(
    prediction: Mapping,
    *,
    reviewer: str,
    feedback_type: str,
    reason: str,
    timestamp: str,
) -> dict:
    validate_review_prediction(prediction)
    allowed = {"human_confirmation", "false_positive", "false_negative"}
    if feedback_type not in allowed:
        raise OperationalQualificationError("REVIEW_FEEDBACK_TYPE_INVALID")
    if feedback_type == "false_positive" and prediction["anomaly_label"] != "ANOMALY":
        raise OperationalQualificationError("FALSE_POSITIVE_REQUIRES_ANOMALY_PREDICTION")
    if feedback_type == "false_negative" and prediction["anomaly_label"] != "NORMAL":
        raise OperationalQualificationError("FALSE_NEGATIVE_REQUIRES_NORMAL_PREDICTION")
    if not all(isinstance(value, str) and value for value in (reviewer, reason, timestamp)):
        raise OperationalQualificationError("REVIEW_FEEDBACK_STRING_INVALID")
    return {
        "schema_version": REVIEW_WORKFLOW_SCHEMA_VERSION,
        "image_id": prediction["image_id"],
        "model_version": prediction["model_version"],
        "artifact_version": prediction["artifact_version"],
        "reviewer": reviewer,
        "feedback_type": feedback_type,
        "reason": reason,
        "timestamp": timestamp,
        "prediction_snapshot": dict(prediction),
        "automatic_retraining": False,
        "automatic_threshold_change": False,
    }


@dataclass
class D3RuntimeMonitor:
    """Readiness latch and runtime telemetry bound to one immutable artifact set."""

    model_version: str
    artifact_version: str
    expected_artifact_hashes: dict[str, str]
    request_count: int = 0
    error_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    cpu_memory_mb: list[float] = field(default_factory=list)
    gpu_samples: list[dict] = field(default_factory=list)
    ready: bool = False
    failure_reason: str | None = None

    def _close(self, reason: str) -> None:
        self.ready = False
        self.failure_reason = reason

    def start(self, verify_artifacts: Callable[[], Mapping[str, str]], load_model: Callable[[], object]) -> object:
        try:
            actual = dict(verify_artifacts())
            if actual != self.expected_artifact_hashes:
                raise OperationalQualificationError("ARTIFACT_HASH_MISMATCH")
            model = load_model()
        except Exception as exc:
            self._close(f"{type(exc).__name__}:{exc}")
            raise OperationalQualificationError(f"RUNTIME_MONITOR_FAIL_CLOSED:{self.failure_reason}") from exc
        self.ready = True
        self.failure_reason = None
        return model

    def verify(self, verify_artifacts: Callable[[], Mapping[str, str]]) -> None:
        if not self.ready:
            raise OperationalQualificationError(f"RUNTIME_MONITOR_NOT_READY:{self.failure_reason or 'STARTUP_REQUIRED'}")
        try:
            actual = dict(verify_artifacts())
            if actual != self.expected_artifact_hashes:
                raise OperationalQualificationError("ARTIFACT_HASH_MISMATCH")
        except Exception as exc:
            self._close(f"{type(exc).__name__}:{exc}")
            raise OperationalQualificationError(f"RUNTIME_MONITOR_FAIL_CLOSED:{self.failure_reason}") from exc

    def execute(
        self,
        operation: Callable[[], object],
        *,
        cpu_memory_mb: Callable[[], float] | None = None,
        gpu_status: Callable[[], Mapping] | None = None,
    ):
        if not self.ready:
            raise OperationalQualificationError(f"RUNTIME_MONITOR_NOT_READY:{self.failure_reason or 'STARTUP_REQUIRED'}")
        self.request_count += 1
        started = time.perf_counter()
        try:
            result = operation()
        except Exception as exc:
            self.error_count += 1
            self._close(f"{type(exc).__name__}:{exc}")
            raise
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
            if cpu_memory_mb is not None:
                value = float(cpu_memory_mb())
                if math.isfinite(value) and value >= 0:
                    self.cpu_memory_mb.append(value)
            if gpu_status is not None:
                sample = dict(gpu_status())
                if sample:
                    self.gpu_samples.append(sample)
        return result

    def snapshot(self) -> dict:
        gpu_memory = [float(row["memory_allocated_mb"]) for row in self.gpu_samples if row.get("memory_allocated_mb") is not None]
        utilization = [float(row["utilization_percent"]) for row in self.gpu_samples if row.get("utilization_percent") is not None]
        return {
            "schema_version": RUNTIME_MONITOR_SCHEMA_VERSION,
            "model_version": self.model_version,
            "artifact_version": self.artifact_version,
            "artifact_hashes": dict(self.expected_artifact_hashes),
            "ready": self.ready,
            "failure_reason": self.failure_reason,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count else 0.0,
            "latency_ms": latency_percentiles(self.latencies_ms) if self.latencies_ms else None,
            "cpu_memory_peak_mb": max(self.cpu_memory_mb) if self.cpu_memory_mb else None,
            "gpu_memory_peak_mb": max(gpu_memory) if gpu_memory else None,
            "gpu_utilization_peak_percent": max(utilization) if utilization else None,
        }


def memory_leak_analysis(samples: list[Mapping]) -> dict:
    if len(samples) < 3:
        raise OperationalQualificationError("MEMORY_LEAK_SAMPLE_COUNT_INVALID")
    stable = [row for row in samples if float(row["virtual_hour"]) >= 1.0]
    if len(stable) < 2:
        raise OperationalQualificationError("MEMORY_LEAK_STABLE_WINDOW_MISSING")
    hours = np.asarray([row["virtual_hour"] for row in stable], dtype=np.float64)

    def trend(key: str) -> dict:
        values = np.asarray([row[key] for row in stable], dtype=np.float64)
        if not np.isfinite(values).all():
            raise OperationalQualificationError(f"MEMORY_LEAK_NONFINITE:{key}")
        slope = float(np.polyfit(hours, values, 1)[0])
        return {"start_mb": float(values[0]), "end_mb": float(values[-1]), "growth_mb": float(values[-1] - values[0]), "slope_mb_per_hour": slope, "peak_mb": float(values.max())}

    cpu = trend("cpu_memory_mb")
    gpu = trend("gpu_memory_mb")
    checks = {
        "cpu_growth": cpu["growth_mb"] <= MEMORY_LIMITS["cpu_growth_mb_max"],
        "cpu_slope": cpu["slope_mb_per_hour"] <= MEMORY_LIMITS["cpu_slope_mb_per_hour_max"],
        "gpu_growth": gpu["growth_mb"] <= MEMORY_LIMITS["gpu_growth_mb_max"],
        "gpu_slope": gpu["slope_mb_per_hour"] <= MEMORY_LIMITS["gpu_slope_mb_per_hour_max"],
    }
    return {"stable_window_start_hour": 1.0, "limits": MEMORY_LIMITS, "cpu": cpu, "gpu": gpu, "checks": checks, "passed": all(checks.values())}


def _verified_candidate_entry(manifest_path: Path, project_root: Path) -> dict:
    path = manifest_path.resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationalQualificationError(f"ROLLBACK_MANIFEST_UNREADABLE:{path}") from exc
    try:
        if manifest.get("schema_version") == DUAL_SCHEMA_VERSION:
            registry = DualCandidateRegistry(project_root)
            loaded = registry.load_manifest(path)
            _, hashes = registry.verify_artifacts(loaded)
        else:
            registry = CandidateRegistry(project_root / "model-training/registry", project_root)
            loaded = manifest
            from steel_patchcore.candidate_registry import validate_manifest

            validate_manifest(loaded)
            verification = registry.verify_artifact(loaded)
            if not verification.passed:
                raise OperationalQualificationError(f"ROLLBACK_ARTIFACT_INVALID:{verification.errors}")
            hashes = verification.hashes
    except (CandidateRegistryError, OperationalQualificationError) as exc:
        raise OperationalQualificationError(f"ROLLBACK_MANIFEST_INVALID:{exc}") from exc
    if loaded.get("status") != "CANDIDATE" or loaded.get("production_promotion") is not False:
        raise OperationalQualificationError("ROLLBACK_CANDIDATE_ONLY_REQUIRED")
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "model_name": loaded["model_name"],
        "model_version": loaded["model_version"],
        "artifact_version": loaded["artifact_version"],
        "artifact_hashes": hashes,
    }


class DualCandidateRollbackManager:
    """Two candidate slots with hash verification and no production state."""

    def __init__(self, state_path: Path, project_root: Path) -> None:
        self.state_path = state_path.resolve()
        self.project_root = project_root.resolve()

    def load(self) -> dict:
        if not self.state_path.is_file():
            return {
                "schema_version": ROLLBACK_DRILL_SCHEMA_VERSION,
                "status": "CANDIDATE_ONLY",
                "active_candidate": None,
                "previous_candidate": None,
                "automatic_production_upgrade": False,
            }
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OperationalQualificationError("ROLLBACK_STATE_UNREADABLE") from exc
        if state.get("schema_version") != ROLLBACK_DRILL_SCHEMA_VERSION or state.get("status") != "CANDIDATE_ONLY" or state.get("automatic_production_upgrade") is not False:
            raise OperationalQualificationError("ROLLBACK_STATE_INVALID")
        return state

    def activate(self, manifest_path: Path) -> dict:
        candidate = _verified_candidate_entry(manifest_path, self.project_root)
        state = self.load()
        if state["active_candidate"] != candidate:
            state["previous_candidate"] = state["active_candidate"]
            state["active_candidate"] = candidate
            atomic_write_json(self.state_path, state)
        return state

    def rollback(self) -> dict:
        state = self.load()
        previous = state.get("previous_candidate")
        if previous is None:
            raise OperationalQualificationError("ROLLBACK_PREVIOUS_CANDIDATE_MISSING")
        verified = _verified_candidate_entry(Path(previous["manifest_path"]), self.project_root)
        if verified != previous:
            raise OperationalQualificationError("ROLLBACK_PREVIOUS_HASH_MISMATCH")
        next_state = dict(state)
        next_state["active_candidate"], next_state["previous_candidate"] = previous, state["active_candidate"]
        atomic_write_json(self.state_path, next_state)
        return next_state


def validate_readiness_report(report: Mapping) -> None:
    if report.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise OperationalQualificationError("READINESS_SCHEMA_MISMATCH")
    if report.get("candidate_status") != "CANDIDATE" or report.get("production_promotion") is not False:
        raise OperationalQualificationError("READINESS_CANDIDATE_ONLY_REQUIRED")
    phases = report.get("phases", {})
    if set(phases) != {"stability", "robustness", "monitoring", "human_review", "rollback", "tests"}:
        raise OperationalQualificationError("READINESS_PHASE_SET_MISMATCH")
    expected = "PRODUCTION_CANDIDATE_QUALIFIED" if all(row.get("verdict") == "PASS" for row in phases.values()) else "NOT_QUALIFIED"
    if report.get("verdict") != expected:
        raise OperationalQualificationError("READINESS_VERDICT_MISMATCH")


__all__ = [
    "D3RuntimeMonitor", "DualCandidateRollbackManager", "HumanReviewPrediction", "MEMORY_LIMITS",
    "READINESS_SCHEMA_VERSION", "REVIEW_WORKFLOW_SCHEMA_VERSION", "ROBUSTNESS_GATE",
    "ROBUSTNESS_SCHEMA_VERSION", "ROLLBACK_DRILL_SCHEMA_VERSION", "RUNTIME_MONITOR_SCHEMA_VERSION",
    "STABILITY_SCHEMA_VERSION", "create_feedback_record", "memory_leak_analysis",
    "threshold_margin_confidence", "validate_readiness_report", "validate_review_prediction",
]
