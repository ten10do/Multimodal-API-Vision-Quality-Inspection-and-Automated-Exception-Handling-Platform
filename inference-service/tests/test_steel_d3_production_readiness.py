"""D3 1.3 production-readiness qualification contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.d3_operational import OperationalQualificationError  # noqa: E402
from steel_patchcore.d3_production_readiness import (  # noqa: E402
    READINESS_SCHEMA_VERSION,
    D3RuntimeMonitor,
    DualCandidateRollbackManager,
    HumanReviewPrediction,
    create_feedback_record,
    memory_leak_analysis,
    threshold_margin_confidence,
    validate_readiness_report,
)

LEGACY_MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/manifest.json"
DUAL_MANIFEST = ROOT / "model-training/registry/steel-patchcore-d3-candidate/1.3.0-candidate.1/manifest.json"
STABILITY_REPORT = ROOT / "docs/d3-24h-stability-report.json"
ROBUSTNESS_REPORT = ROOT / "docs/d3-input-robustness-report.json"
MONITORING_REPORT = ROOT / "docs/d3-runtime-monitoring-report.json"
ROLLBACK_REPORT = ROOT / "docs/d3-rollback-drill-report.json"
READINESS_REPORT = ROOT / "docs/d3-production-readiness-report.json"


def _hashes() -> dict[str, str]:
    return {"weights": "a" * 64, "bank": "b" * 64}


def test_runtime_monitor_records_identity_resources_latency_and_failures():
    monitor = D3RuntimeMonitor("1.3.0-candidate.1", "dual", _hashes())
    model = object()
    assert monitor.start(lambda: _hashes(), lambda: model) is model
    assert monitor.execute(
        lambda: "ok",
        cpu_memory_mb=lambda: 125.0,
        gpu_status=lambda: {"memory_allocated_mb": 256.0, "utilization_percent": 75.0},
    ) == "ok"
    snapshot = monitor.snapshot()
    assert snapshot["model_version"] == "1.3.0-candidate.1"
    assert snapshot["artifact_version"] == "dual"
    assert snapshot["request_count"] == 1 and snapshot["error_rate"] == 0.0
    assert snapshot["cpu_memory_peak_mb"] == 125.0
    assert snapshot["gpu_memory_peak_mb"] == 256.0
    assert snapshot["gpu_utilization_peak_percent"] == 75.0


@pytest.mark.parametrize("failure", ["hash mismatch", "artifact missing", "model load failure"])
def test_runtime_monitor_startup_is_fail_closed(failure):
    monitor = D3RuntimeMonitor("1.3", "dual", _hashes())
    if failure == "hash mismatch":
        verifier = lambda: {"weights": "c" * 64, "bank": "b" * 64}
        loader = lambda: object()
    elif failure == "artifact missing":
        verifier = lambda: (_ for _ in ()).throw(FileNotFoundError("artifact missing"))
        loader = lambda: object()
    else:
        verifier = lambda: _hashes()
        loader = lambda: (_ for _ in ()).throw(OSError("model load failure"))
    with pytest.raises(OperationalQualificationError, match="FAIL_CLOSED"):
        monitor.start(verifier, loader)
    assert monitor.ready is False and failure.split()[0] in monitor.failure_reason.lower()
    with pytest.raises(OperationalQualificationError, match="NOT_READY"):
        monitor.execute(lambda: None)


def test_runtime_monitor_periodic_hash_mismatch_and_inference_error_latch_closed():
    monitor = D3RuntimeMonitor("1.3", "dual", _hashes())
    monitor.start(lambda: _hashes(), lambda: object())
    with pytest.raises(OperationalQualificationError, match="FAIL_CLOSED"):
        monitor.verify(lambda: {"weights": "f" * 64})
    assert monitor.ready is False
    second = D3RuntimeMonitor("1.3", "dual", _hashes())
    second.start(lambda: _hashes(), lambda: object())
    with pytest.raises(ValueError, match="inference failure"):
        second.execute(lambda: (_ for _ in ()).throw(ValueError("inference failure")))
    assert second.ready is False and second.error_count == 1


def test_memory_leak_analysis_uses_post_warmup_24h_trend():
    stable = [
        {"virtual_hour": hour, "cpu_memory_mb": 500.0 + 0.2 * hour, "gpu_memory_mb": 1000.0}
        for hour in range(25)
    ]
    result = memory_leak_analysis(stable)
    assert result["passed"] is True
    leaking = [dict(row) for row in stable]
    for row in leaking:
        row["cpu_memory_mb"] = 500.0 + 10.0 * row["virtual_hour"]
    assert memory_leak_analysis(leaking)["passed"] is False


def test_review_prediction_exposes_uncalibrated_confidence_and_feedback_types():
    confidence = threshold_margin_confidence(0.9, 0.8)
    prediction = HumanReviewPrediction(
        image_id="sample",
        image_score=0.9,
        anomaly_label="ANOMALY",
        heatmap_ref="review/sample.png",
        confidence=confidence,
        model_version="1.3.0-candidate.1",
        artifact_version="d3-dual-rl3-0b148a6",
    ).as_record()
    assert prediction["confidence"]["calibrated_probability"] is False
    confirmed = create_feedback_record(
        prediction, reviewer="operator", feedback_type="human_confirmation", reason="confirmed", timestamp="2026-08-21T00:00:00Z"
    )
    false_positive = create_feedback_record(
        prediction, reviewer="operator", feedback_type="false_positive", reason="clean steel", timestamp="2026-08-21T00:00:00Z"
    )
    assert confirmed["automatic_retraining"] is False
    assert false_positive["automatic_threshold_change"] is False
    with pytest.raises(OperationalQualificationError, match="FALSE_NEGATIVE"):
        create_feedback_record(
            prediction, reviewer="operator", feedback_type="false_negative", reason="miss", timestamp="2026-08-21T00:00:00Z"
        )


@pytest.mark.artifact
def test_candidate_to_previous_rollback_drill_verifies_hashes_and_blocks_tampering(tmp_path):
    state_path = tmp_path / "rollback-state.json"
    manager = DualCandidateRollbackManager(state_path, ROOT)
    state = manager.activate(LEGACY_MANIFEST)
    assert state["active_candidate"]["model_version"] == "1.2.0-candidate.1"
    state = manager.activate(DUAL_MANIFEST)
    assert state["active_candidate"]["model_version"] == "1.3.0-candidate.1"
    assert state["previous_candidate"]["model_version"] == "1.2.0-candidate.1"
    rolled_back = manager.rollback()
    assert rolled_back["active_candidate"]["model_version"] == "1.2.0-candidate.1"
    assert rolled_back["automatic_production_upgrade"] is False

    manager.activate(DUAL_MANIFEST)
    tampered = manager.load()
    tampered["previous_candidate"]["manifest_sha256"] = "0" * 64
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    before = state_path.read_bytes()
    with pytest.raises(OperationalQualificationError, match="HASH_MISMATCH"):
        manager.rollback()
    assert state_path.read_bytes() == before


def test_readiness_report_verdict_is_derived_from_all_six_phases():
    phases = {name: {"verdict": "PASS"} for name in ("stability", "robustness", "monitoring", "human_review", "rollback", "tests")}
    report = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "candidate_status": "CANDIDATE",
        "phases": phases,
        "verdict": "PRODUCTION_CANDIDATE_QUALIFIED",
        "production_promotion": False,
    }
    validate_readiness_report(report)
    report["phases"]["robustness"]["verdict"] = "FAILED"
    report["verdict"] = "NOT_QUALIFIED"
    validate_readiness_report(report)


def test_actual_stability_robustness_monitoring_and_rollback_reports_pass():
    stability = json.loads(STABILITY_REPORT.read_text(encoding="utf-8"))
    assert stability["verdict"] == "PASS"
    assert stability["simulation"]["virtual_duration_hours"] == 24
    assert stability["simulation"]["wall_clock_duration_claimed"] is False
    assert stability["monitoring"]["request_count"] == 240
    assert stability["monitoring"]["error_rate"] == 0.0
    assert stability["score_drift_count"] == 0
    assert stability["memory_leak_analysis"]["passed"] is True
    assert stability["artifact_unchanged"] is True

    robustness = json.loads(ROBUSTNESS_REPORT.read_text(encoding="utf-8"))
    assert robustness["verdict"] == "PASS"
    assert {row["condition"] for row in robustness["conditions"]} == {
        "baseline", "brightness_shift", "contrast_shift", "noise", "compression", "small_resize"
    }
    assert all(row["verdict"] == "PASS" for row in robustness["conditions"])
    assert robustness["artifact_unchanged"] is True

    monitoring = json.loads(MONITORING_REPORT.read_text(encoding="utf-8"))
    assert monitoring["verdict"] == "PASS"
    assert {row["failure"] for row in monitoring["failure_drills"]} == {
        "artifact_hash_mismatch", "artifact_missing", "model_load_failure"
    }
    assert all(row["fail_closed"] for row in monitoring["failure_drills"])

    rollback = json.loads(ROLLBACK_REPORT.read_text(encoding="utf-8"))
    assert rollback["verdict"] == "PASS"
    assert rollback["rolled_back_active_candidate"]["model_version"] == "1.2.0-candidate.1"
    assert rollback["hash_mismatch_drill"] == {
        "blocked": True,
        "reason": "ROLLBACK_PREVIOUS_HASH_MISMATCH",
        "state_unchanged": True,
    }
    assert rollback["production_promotion"] is False

    readiness = json.loads(READINESS_REPORT.read_text(encoding="utf-8"))
    validate_readiness_report(readiness)
    assert readiness["verdict"] == "PRODUCTION_CANDIDATE_QUALIFIED"
    assert all(row["verdict"] == "PASS" for row in readiness["phases"].values())
    assert readiness["production_promotion"] is False
