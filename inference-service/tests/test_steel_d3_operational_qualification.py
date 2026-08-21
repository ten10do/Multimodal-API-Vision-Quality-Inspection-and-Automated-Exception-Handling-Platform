"""Regression coverage for D3 operational qualification."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.candidate_registry import canonical_sha256  # noqa: E402
from steel_patchcore.d3_operational import (  # noqa: E402
    HEATMAP_SCHEMA_VERSION,
    PERFORMANCE_SCHEMA_VERSION,
    CandidateRollbackManager,
    InferenceMonitor,
    OperationalQualificationError,
    assert_reproducible,
    pixel_localization_metrics,
    validate_heatmap_report,
    validate_performance_report,
    validate_shadow_record,
)

MANIFEST_PATH = ROOT / "model-training/registry/steel-patchcore-d3-candidate/manifest.json"


def test_performance_schema_requires_all_four_prefixes_and_resources():
    report = {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "candidate_status": "CANDIDATE",
        "production_promotion": False,
        "benchmarks": [
            {
                "image_count": size,
                "latency_ms": {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0},
                "resources": {
                    "cpu_memory_peak_mb": 100.0,
                    "gpu_memory_peak_mb": 200.0,
                    "gpu_utilization_percent": 50.0,
                },
            }
            for size in (1, 10, 100, 1000)
        ],
    }
    validate_performance_report(report)
    report["benchmarks"].pop()
    with pytest.raises(OperationalQualificationError, match="BATCH_SET"):
        validate_performance_report(report)


def test_shadow_schema_has_required_prediction_log_fields():
    record = {
        "timestamp": "2026-08-21T00:00:00Z",
        "image_id": "sample",
        "model_version": "1.2.0-candidate.1",
        "artifact_version": "d3-full-development-9b1ea19",
        "score": 0.75,
        "heatmap_path": "runs/sample.png",
    }
    validate_shadow_record(record)
    record.pop("artifact_version")
    with pytest.raises(OperationalQualificationError, match="FIELD_MISSING"):
        validate_shadow_record(record)


def test_heatmap_metrics_match_frozen_per_image_semantics():
    anomaly_map = np.asarray([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    mask = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
    metrics = pixel_localization_metrics(anomaly_map, mask)
    assert metrics["pixel_auroc"] == pytest.approx(1.0)
    assert metrics["aupro"] == pytest.approx(1.0)
    report = {
        "schema_version": HEATMAP_SCHEMA_VERSION,
        "sample_count": 1,
        "metrics": {
            "pixel_auroc_mean_per_image": metrics["pixel_auroc"],
            "aupro_mean_per_image": metrics["aupro"],
        },
        "paired_baseline_comparison": {"sample_count": 1},
        "image_score_gate_changed": False,
    }
    validate_heatmap_report(report)
    report["image_score_gate_changed"] = True
    with pytest.raises(OperationalQualificationError, match="GATE_CHANGE_FORBIDDEN"):
        validate_heatmap_report(report)


def test_monitoring_records_requests_latency_errors_gpu_and_artifact_hash():
    monitor = InferenceMonitor()
    model = object()
    assert monitor.start(
        lambda: SimpleNamespace(passed=True, hashes={"bank_sha256": "a" * 64}, errors=()),
        lambda: model,
    ) is model
    assert monitor.execute(lambda: "ok", gpu_memory_mb=lambda: 42.0) == "ok"
    with pytest.raises(ValueError):
        monitor.execute(lambda: (_ for _ in ()).throw(ValueError("bad request")))
    snapshot = monitor.snapshot()
    assert snapshot["ready"] is True
    assert snapshot["request_count"] == 2
    assert snapshot["error_count"] == 1
    assert snapshot["error_rate"] == 0.5
    assert snapshot["gpu_memory_peak_mb"] == 42.0
    assert snapshot["artifact_hashes"] == {"bank_sha256": "a" * 64}


@pytest.mark.parametrize("failure", ["missing file", "artifact mismatch", "model load failure"])
def test_monitoring_startup_failures_are_fail_closed(failure):
    monitor = InferenceMonitor()
    if failure == "model load failure":
        verifier = lambda: SimpleNamespace(passed=True, hashes={}, errors=())
        loader = lambda: (_ for _ in ()).throw(OSError(failure))
    else:
        verifier = lambda: SimpleNamespace(passed=False, hashes={}, errors=(failure,))
        loader = lambda: object()
    with pytest.raises(OperationalQualificationError, match="MONITOR_FAIL_CLOSED"):
        monitor.start(verifier, loader)
    assert monitor.ready is False
    with pytest.raises(OperationalQualificationError, match="MONITOR_NOT_READY"):
        monitor.execute(lambda: None)


class _FakeRegistry:
    def verify_artifact(self, manifest):
        return SimpleNamespace(
            passed=True,
            hashes={"bank_sha256": manifest["bank_sha256"]},
            errors=(),
        )


def _write_candidate(path: Path, version: str) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["model_version"] = version
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_candidate_rollback_has_active_previous_slots_and_verifies_hash(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_candidate(first, "1.2.0-candidate.1")
    _write_candidate(second, "1.2.0-candidate.2")
    manager = CandidateRollbackManager(tmp_path / "state.json", _FakeRegistry())
    first_state = manager.activate(first)
    assert first_state["active_candidate"]["model_version"] == "1.2.0-candidate.1"
    assert first_state["previous_candidate"] is None
    second_state = manager.activate(second)
    assert second_state["active_candidate"]["model_version"] == "1.2.0-candidate.2"
    assert second_state["previous_candidate"]["model_version"] == "1.2.0-candidate.1"
    rolled_back = manager.rollback()
    assert rolled_back["active_candidate"]["model_version"] == "1.2.0-candidate.1"
    assert rolled_back["status"] == "CANDIDATE_ONLY"
    assert rolled_back["automatic_production_upgrade"] is False


def test_candidate_rollback_fails_closed_if_previous_manifest_changes(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_candidate(first, "1.2.0-candidate.1")
    _write_candidate(second, "1.2.0-candidate.2")
    manager = CandidateRollbackManager(tmp_path / "state.json", _FakeRegistry())
    manager.activate(first)
    manager.activate(second)
    first.write_text(first.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(OperationalQualificationError, match="HASH_MISMATCH"):
        manager.rollback()


def test_inference_reproducibility_is_exact_with_frozen_tolerance():
    assert_reproducible({"a": 0.1, "b": 0.2}, {"a": 0.1000001, "b": 0.2})
    with pytest.raises(OperationalQualificationError, match="SCORE_MISMATCH"):
        assert_reproducible({"a": 0.1}, {"a": 0.10001})


def test_runner_preserves_threshold_artifact_and_candidate_only_guards():
    source = (ROOT / "inference-service/scripts/qualify_steel_d3_candidate.py").read_text(encoding="utf-8")
    assert "threshold_changed\": False" in source
    assert "artifact_changed\": False" in source
    assert "production_promotion\": False" in source
    assert "train(" not in source
    assert "optimizer" not in source.lower()
    assert "threshold =" not in source
