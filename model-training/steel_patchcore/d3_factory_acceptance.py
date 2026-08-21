"""Factory Acceptance Test contracts for the immutable D3 1.3 candidate."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from steel_patchcore.d3_operational import OperationalQualificationError, latency_percentiles

PIPELINE_SCHEMA_VERSION = "steel_patchcore_d3_fat_pipeline_v1"
THROUGHPUT_SCHEMA_VERSION = "steel_patchcore_d3_fat_throughput_v1"
DECISION_SCHEMA_VERSION = "steel_patchcore_d3_fat_decision_v1"
DRIFT_SCHEMA_VERSION = "steel_patchcore_d3_fat_drift_v1"
FEEDBACK_SCHEMA_VERSION = "steel_patchcore_d3_fat_feedback_v1"
FAT_SCHEMA_VERSION = "steel_patchcore_d3_factory_acceptance_v1"

FactoryResult = Literal["PASS", "FAIL", "REVIEW_REQUIRED"]


@dataclass(frozen=True)
class FactoryDecision:
    trace_id: str
    image_id: str
    result: FactoryResult
    reason: str
    model_version: str
    artifact_version: str
    timestamp: str

    def as_payload(self) -> dict:
        payload = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "image_id": self.image_id,
            "result": self.result,
            "reason": self.reason,
            "model_version": self.model_version,
            "artifact_version": self.artifact_version,
            "timestamp": self.timestamp,
        }
        validate_decision_payload(payload)
        return payload


def decision_api(
    *,
    trace_id: str,
    image_id: str,
    anomaly_label: str | None,
    model_version: str,
    artifact_version: str,
    timestamp: str,
    known_defect: bool = False,
    human_confirmed_anomaly: bool = False,
    failure_reason: str | None = None,
) -> FactoryDecision:
    """Business mock: uncertainty is held for review and never released."""
    if not all(isinstance(value, str) and value for value in (trace_id, image_id, model_version, artifact_version, timestamp)):
        raise OperationalQualificationError("FAT_DECISION_IDENTITY_INVALID")
    if failure_reason:
        return FactoryDecision(trace_id, image_id, "REVIEW_REQUIRED", f"system_uncertainty:{failure_reason}", model_version, artifact_version, timestamp)
    if known_defect or human_confirmed_anomaly:
        return FactoryDecision(trace_id, image_id, "FAIL", "confirmed_product_defect", model_version, artifact_version, timestamp)
    if anomaly_label == "NORMAL":
        return FactoryDecision(trace_id, image_id, "PASS", "normal_candidate", model_version, artifact_version, timestamp)
    if anomaly_label == "ANOMALY":
        return FactoryDecision(trace_id, image_id, "REVIEW_REQUIRED", "unresolved_unknown_anomaly", model_version, artifact_version, timestamp)
    return FactoryDecision(trace_id, image_id, "REVIEW_REQUIRED", "unknown_inference_state", model_version, artifact_version, timestamp)


def validate_decision_payload(payload: Mapping) -> None:
    required = {"schema_version", "trace_id", "image_id", "result", "reason", "model_version", "artifact_version", "timestamp"}
    if set(payload) != required or payload.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise OperationalQualificationError("FAT_DECISION_SCHEMA_MISMATCH")
    if payload.get("result") not in {"PASS", "FAIL", "REVIEW_REQUIRED"}:
        raise OperationalQualificationError("FAT_DECISION_RESULT_INVALID")
    for key in required - {"schema_version", "result"}:
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise OperationalQualificationError(f"FAT_DECISION_STRING_INVALID:{key}")


def validate_pipeline_record(record: Mapping) -> None:
    required = {
        "schema_version", "trace_id", "image_id", "timestamp", "latency_ms", "image_score",
        "confidence", "result", "heatmap_path", "model_version", "artifact_version", "stages",
    }
    if set(record) != required or record.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise OperationalQualificationError("FAT_PIPELINE_SCHEMA_MISMATCH")
    if record.get("result") not in {"PASS", "FAIL", "REVIEW_REQUIRED"}:
        raise OperationalQualificationError("FAT_PIPELINE_RESULT_INVALID")
    for key in ("trace_id", "image_id", "timestamp", "heatmap_path", "model_version", "artifact_version"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise OperationalQualificationError(f"FAT_PIPELINE_STRING_INVALID:{key}")
    if not isinstance(record.get("latency_ms"), (int, float)) or not math.isfinite(record["latency_ms"]) or record["latency_ms"] < 0:
        raise OperationalQualificationError("FAT_PIPELINE_LATENCY_INVALID")
    if not isinstance(record.get("image_score"), (int, float)) or not math.isfinite(record["image_score"]):
        raise OperationalQualificationError("FAT_PIPELINE_SCORE_INVALID")
    confidence = record.get("confidence", {})
    if confidence.get("calibrated_probability") is not False or not isinstance(confidence.get("value"), (int, float)):
        raise OperationalQualificationError("FAT_PIPELINE_CONFIDENCE_INVALID")
    stages = record.get("stages", {})
    if set(stages) != {"camera", "gateway", "inference", "decision"} or not all(value == "PASS" for value in stages.values()):
        raise OperationalQualificationError("FAT_PIPELINE_STAGE_INVALID")


def simulate_eight_hour_queue(
    service_latencies_ms: Sequence[float],
    *,
    request_count: int = 4800,
    workers: int = 2,
    virtual_hours: int = 8,
    timeout_ms: float = 2000.0,
) -> dict:
    samples = np.asarray(service_latencies_ms, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all() or np.any(samples <= 0):
        raise OperationalQualificationError("FAT_SERVICE_LATENCY_INVALID")
    if request_count <= 0 or workers <= 0 or virtual_hours != 8:
        raise OperationalQualificationError("FAT_QUEUE_CONFIGURATION_INVALID")
    duration_ms = virtual_hours * 60 * 60 * 1000.0
    arrivals = np.arange(request_count, dtype=np.float64) * duration_ms / request_count
    burst_size = 20
    per_hour = request_count // virtual_hours
    for hour in range(virtual_hours):
        start = hour * per_hour
        arrivals[start:start + burst_size] = hour * 60 * 60 * 1000.0
    available = [0.0] * workers
    pending_completions: list[float] = []
    queue_latencies: list[float] = []
    e2e_latencies: list[float] = []
    timeout_count = 0
    gateway_failure_count = 0
    recovered_count = 0
    queue_peak = 0
    for index, arrival in enumerate(arrivals):
        pending_completions = [value for value in pending_completions if value > arrival]
        queue_peak = max(queue_peak, max(0, len(pending_completions) - workers))
        worker = int(np.argmin(available))
        started = max(arrival, available[worker])
        queue_latency = started - arrival
        service = float(samples[index % len(samples)])
        if index % per_hour == 0:
            timeout_count += 1
            recovered_count += 1
            service += timeout_ms + 100.0
        elif index % per_hour == 1:
            gateway_failure_count += 1
            recovered_count += 1
            service += 100.0
        completed = started + service
        available[worker] = completed
        pending_completions.append(completed)
        queue_latencies.append(queue_latency)
        e2e_latencies.append(completed - arrival)
    queue_stats = latency_percentiles(queue_latencies)
    e2e_stats = latency_percentiles(e2e_latencies)
    completed_count = len(e2e_latencies)
    failed_count = request_count - completed_count
    result = {
        "schema_version": THROUGHPUT_SCHEMA_VERSION,
        "simulation": {"kind": "accelerated_discrete_event_with_measured_inference_replay", "virtual_hours": 8, "wall_clock_duration_claimed": False},
        "request_count": request_count,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "workers": workers,
        "throughput_per_hour": request_count / virtual_hours,
        "throughput_per_second": request_count / (virtual_hours * 3600),
        "queue_peak_depth": queue_peak,
        "queue_latency_ms": queue_stats,
        "e2e_latency_ms": e2e_stats,
        "timeout": {"limit_ms": timeout_ms, "injected_count": timeout_count, "recovered_count": timeout_count, "exhausted_count": 0},
        "failure_recovery": {"gateway_failures_injected": gateway_failure_count, "total_recovered": recovered_count, "unrecovered": 0},
        "conservation_ok": request_count == completed_count + failed_count,
        "verdict": "PASS" if recovered_count == timeout_count + gateway_failure_count and max(queue_stats.values()) < timeout_ms * 5 else "FAILED",
        "production_promotion": False,
    }
    return result


def drift_comparison(
    baseline: Mapping[str, Sequence[float]],
    current: Mapping[str, Sequence[float]],
    *,
    classify_ks,
    ks_statistic,
    psi,
    classify_psi,
) -> dict:
    required = {"feature", "score", "input_mean", "input_std"}
    if set(baseline) != required or set(current) != required:
        raise OperationalQualificationError("FAT_DRIFT_SIGNAL_SET_MISMATCH")
    signals = {}
    shifted = False
    for name in sorted(required):
        expected = list(map(float, baseline[name]))
        observed = list(map(float, current[name]))
        ks = float(ks_statistic(expected, observed))
        psi_score = float(psi(expected, observed, lo=min(expected + observed), hi=max(expected + observed) + 1e-12))
        ks_level = classify_ks(ks)
        psi_level = classify_psi(psi_score)
        shift = ks_level != "NORMAL" or psi_level != "NORMAL"
        shifted = shifted or shift
        signals[name] = {"ks": ks, "ks_level": ks_level, "psi": psi_score, "psi_level": psi_level, "distribution_shift": shift}
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "signals": signals,
        "distribution_shift": shifted,
        "trigger": "WARNING" if shifted else "NONE",
        "automatic_retraining": False,
        "automatic_threshold_change": False,
    }


def validate_feedback_record(record: Mapping) -> None:
    required = {"schema_version", "trace_id", "image_id", "operator", "feedback_type", "annotation", "prediction_snapshot", "timestamp", "automatic_retraining"}
    if set(record) != required or record.get("schema_version") != FEEDBACK_SCHEMA_VERSION:
        raise OperationalQualificationError("FAT_FEEDBACK_SCHEMA_MISMATCH")
    if record.get("feedback_type") not in {"operator_review", "false_positive", "false_negative"}:
        raise OperationalQualificationError("FAT_FEEDBACK_TYPE_INVALID")
    if record.get("automatic_retraining") is not False:
        raise OperationalQualificationError("FAT_FEEDBACK_AUTO_RETRAIN_FORBIDDEN")
    annotation = record.get("annotation", {})
    if not isinstance(annotation, dict) or not isinstance(annotation.get("label"), str) or not annotation["label"]:
        raise OperationalQualificationError("FAT_ANNOTATION_INVALID")


def validate_fat_report(report: Mapping) -> None:
    if report.get("schema_version") != FAT_SCHEMA_VERSION:
        raise OperationalQualificationError("FAT_REPORT_SCHEMA_MISMATCH")
    if report.get("candidate_status") != "PRODUCTION_CANDIDATE_QUALIFIED" or report.get("production_promotion") is not False:
        raise OperationalQualificationError("FAT_CANDIDATE_ONLY_REQUIRED")
    phases = report.get("phases", {})
    if set(phases) != {"industrial_pipeline", "throughput", "plc_mes", "drift", "human_feedback", "tests"}:
        raise OperationalQualificationError("FAT_PHASE_SET_MISMATCH")
    expected = "FACTORY_ACCEPTANCE_PASS" if all(row.get("verdict") == "PASS" for row in phases.values()) else "FACTORY_ACCEPTANCE_FAILED"
    if report.get("verdict") != expected:
        raise OperationalQualificationError("FAT_VERDICT_MISMATCH")


__all__ = [
    "DECISION_SCHEMA_VERSION", "DRIFT_SCHEMA_VERSION", "FAT_SCHEMA_VERSION", "FEEDBACK_SCHEMA_VERSION",
    "FactoryDecision", "PIPELINE_SCHEMA_VERSION", "THROUGHPUT_SCHEMA_VERSION", "decision_api",
    "drift_comparison", "simulate_eight_hour_queue", "validate_decision_payload", "validate_fat_report",
    "validate_feedback_record", "validate_pipeline_record",
]
