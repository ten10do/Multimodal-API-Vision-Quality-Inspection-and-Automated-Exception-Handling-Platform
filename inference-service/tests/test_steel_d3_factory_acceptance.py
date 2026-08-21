"""Factory Acceptance Test contracts for D3 1.3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "backend"))

from app.mlops.drift import classify_ks, classify_psi, ks_statistic, psi  # noqa: E402
from steel_patchcore.d3_factory_acceptance import (  # noqa: E402
    FAT_SCHEMA_VERSION,
    FEEDBACK_SCHEMA_VERSION,
    PIPELINE_SCHEMA_VERSION,
    decision_api,
    drift_comparison,
    simulate_eight_hour_queue,
    validate_fat_report,
    validate_feedback_record,
    validate_pipeline_record,
)
from steel_patchcore.d3_operational import OperationalQualificationError  # noqa: E402


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"anomaly_label": "NORMAL"}, "PASS"),
        ({"anomaly_label": "ANOMALY"}, "REVIEW_REQUIRED"),
        ({"anomaly_label": "ANOMALY", "human_confirmed_anomaly": True}, "FAIL"),
        ({"anomaly_label": None, "failure_reason": "timeout"}, "REVIEW_REQUIRED"),
    ],
)
def test_decision_api_exposes_three_states_and_holds_uncertainty(kwargs, expected):
    decision = decision_api(
        trace_id="trace-1",
        image_id="image-1",
        model_version="1.3.0-candidate.1",
        artifact_version="d3-dual-rl3-0b148a6",
        timestamp="2026-08-21T00:00:00Z",
        **kwargs,
    ).as_payload()
    assert decision["result"] == expected
    if kwargs.get("failure_reason"):
        assert decision["result"] != "PASS"


def test_pipeline_record_requires_camera_gateway_inference_and_decision():
    record = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "trace_id": "trace-1",
        "image_id": "image-1",
        "timestamp": "2026-08-21T00:00:00Z",
        "latency_ms": 10.0,
        "image_score": 0.5,
        "confidence": {"kind": "absolute_threshold_margin_ratio", "value": 0.1, "calibrated_probability": False},
        "result": "PASS",
        "heatmap_path": "runtime/heatmap.png",
        "model_version": "1.3.0-candidate.1",
        "artifact_version": "artifact",
        "stages": {"camera": "PASS", "gateway": "PASS", "inference": "PASS", "decision": "PASS"},
    }
    validate_pipeline_record(record)
    record["stages"]["gateway"] = "FAILED"
    with pytest.raises(OperationalQualificationError, match="STAGE_INVALID"):
        validate_pipeline_record(record)


def test_eight_hour_queue_simulation_records_timeout_and_recovery():
    report = simulate_eight_hour_queue([500.0, 550.0, 525.0], request_count=4800, workers=2)
    assert report["verdict"] == "PASS"
    assert report["simulation"]["virtual_hours"] == 8
    assert report["simulation"]["wall_clock_duration_claimed"] is False
    assert report["throughput_per_hour"] == 600.0
    assert report["timeout"] == {"limit_ms": 2000.0, "injected_count": 8, "recovered_count": 8, "exhausted_count": 0}
    assert report["failure_recovery"]["gateway_failures_injected"] == 8
    assert report["failure_recovery"]["unrecovered"] == 0
    assert report["conservation_ok"] is True


def test_drift_monitor_warns_without_retraining():
    baseline = {name: [0.1, 0.2, 0.3, 0.4] for name in ("feature", "score", "input_mean", "input_std")}
    normal = drift_comparison(
        baseline, baseline, classify_ks=classify_ks, ks_statistic=ks_statistic, psi=psi, classify_psi=classify_psi
    )
    assert normal["trigger"] == "NONE"
    shifted = {name: [0.7, 0.8, 0.9, 1.0] for name in baseline}
    warning = drift_comparison(
        baseline, shifted, classify_ks=classify_ks, ks_statistic=ks_statistic, psi=psi, classify_psi=classify_psi
    )
    assert warning["trigger"] == "WARNING"
    assert warning["distribution_shift"] is True
    assert warning["automatic_retraining"] is False
    assert warning["automatic_threshold_change"] is False


def test_feedback_record_requires_annotation_and_forbids_auto_retraining():
    record = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "trace_id": "trace-1",
        "image_id": "image-1",
        "operator": "operator-a",
        "feedback_type": "false_positive",
        "annotation": {"label": "clean", "note": "surface accepted"},
        "prediction_snapshot": {"result": "REVIEW_REQUIRED"},
        "timestamp": "2026-08-21T00:00:00Z",
        "automatic_retraining": False,
    }
    validate_feedback_record(record)
    record["automatic_retraining"] = True
    with pytest.raises(OperationalQualificationError, match="AUTO_RETRAIN"):
        validate_feedback_record(record)


def test_fat_verdict_is_derived_from_all_six_phases():
    phases = {name: {"verdict": "PASS"} for name in ("industrial_pipeline", "throughput", "plc_mes", "drift", "human_feedback", "tests")}
    report = {
        "schema_version": FAT_SCHEMA_VERSION,
        "candidate_status": "PRODUCTION_CANDIDATE_QUALIFIED",
        "phases": phases,
        "verdict": "FACTORY_ACCEPTANCE_PASS",
        "production_promotion": False,
    }
    validate_fat_report(report)
    report["phases"]["drift"]["verdict"] = "FAILED"
    report["verdict"] = "FACTORY_ACCEPTANCE_FAILED"
    validate_fat_report(report)


def test_generated_fat_evidence_is_candidate_only_and_complete():
    paths = {
        "pipeline": ROOT / "docs/d3-fat-industrial-pipeline-report.json",
        "throughput": ROOT / "docs/d3-fat-throughput-report.json",
        "plc_mes": ROOT / "docs/d3-fat-plc-mes-report.json",
        "drift": ROOT / "docs/d3-fat-drift-report.json",
        "feedback": ROOT / "docs/d3-fat-human-feedback-report.json",
    }
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    assert all(report["verdict"] == "PASS" for report in reports.values())
    assert reports["pipeline"]["artifact_unchanged"] is True
    assert reports["pipeline"]["threshold"] == 0.8471092581748962
    assert reports["throughput"]["request_count"] == reports["throughput"]["completed_count"] == 4800
    assert reports["throughput"]["simulation"]["wall_clock_duration_claimed"] is False
    assert reports["plc_mes"]["command_mapping"] == {
        "PASS": "RELEASE", "REVIEW_REQUIRED": "HOLD", "FAIL": "REJECT"
    }
    assert reports["drift"]["brightness_shift_window"]["trigger"] == "WARNING"
    assert reports["drift"]["automatic_retraining"] is False
    assert reports["feedback"]["automatic_retraining"] is False
