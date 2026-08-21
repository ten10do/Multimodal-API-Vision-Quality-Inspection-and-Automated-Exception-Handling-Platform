"""Final D3 deployment-readiness package contracts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.d3_deployment_readiness import SCHEMA_VERSION, validate_production_approval_report  # noqa: E402


def test_docker_review_image_is_pinned_non_root_and_health_checked():
    dockerfile = (ROOT / "inference-service/Dockerfile.d3-release-review").read_text(encoding="utf-8")
    assert "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:" in dockerfile
    assert "USER d3review:d3review" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "COPY model-training/runs" not in dockerfile


def test_api_contract_freezes_input_output_errors_trace_and_decision():
    text = (ROOT / "docs/release/api-contract.md").read_text(encoding="utf-8")
    for required in (
        "POST /v1/infer", "multipart/form-data", "VisionResult", "invalid_image",
        "model_unavailable", "vision_error", "X-Request-ID", "REVIEW_REQUIRED", "HOLD",
        "Known blocking deviation",
    ):
        assert required in text


def test_slo_defines_latency_availability_rollback_and_monitoring():
    text = (ROOT / "docs/release/service-level-objective.md").read_text(encoding="utf-8")
    for required in ("99.5%", "p95", "750 ms", "2000 ms", "15 minutes", "Artifact integrity", "Drift"):
        assert required in text


def test_production_approval_verdict_is_derived_and_never_promotes():
    gates = {name: {"verdict": "PASS"} for name in (
        "docker_clean_environment", "api_contract", "service_level_objective", "security", "tests"
    )}
    report = {
        "schema_version": SCHEMA_VERSION,
        "release": "steel-patchcore-d3-release@1.3.0",
        "package_status": "RELEASE_CANDIDATE_PACKAGE",
        "gates": gates,
        "verdict": "PASS",
        "remaining_risks": [],
        "production_promotion": False,
        "model_modified": False,
    }
    validate_production_approval_report(report)
    report["gates"]["security"]["verdict"] = "BLOCKED"
    report["verdict"] = "BLOCKED"
    validate_production_approval_report(report)


def test_current_per_request_anomaly_failure_is_detected_as_fail_open_risk():
    source = (ROOT / "inference-service/inference_app/api.py").read_text(encoding="utf-8")
    assert "the anomaly channel is best-effort" in source
    assert "anomaly_result = None" in source
