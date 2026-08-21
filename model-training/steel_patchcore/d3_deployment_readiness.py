"""Final deployment-readiness report contracts; no promotion operations."""
from __future__ import annotations

from typing import Mapping

from steel_patchcore.candidate_registry import CandidateRegistryError

SCHEMA_VERSION = "steel_patchcore_d3_production_approval_review_v1"


def validate_production_approval_report(report: Mapping) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise CandidateRegistryError("DEPLOYMENT_REVIEW_SCHEMA_MISMATCH")
    if report.get("release") != "steel-patchcore-d3-release@1.3.0":
        raise CandidateRegistryError("DEPLOYMENT_REVIEW_RELEASE_MISMATCH")
    if report.get("package_status") != "RELEASE_CANDIDATE_PACKAGE":
        raise CandidateRegistryError("DEPLOYMENT_REVIEW_STATUS_MISMATCH")
    if report.get("production_promotion") is not False or report.get("model_modified") is not False:
        raise CandidateRegistryError("DEPLOYMENT_REVIEW_FORBIDDEN_ACTION")
    gates = report.get("gates", {})
    expected_gates = {"docker_clean_environment", "api_contract", "service_level_objective", "security", "tests"}
    if set(gates) != expected_gates:
        raise CandidateRegistryError("DEPLOYMENT_REVIEW_GATE_SET_MISMATCH")
    expected = "PASS" if all(row.get("verdict") == "PASS" for row in gates.values()) else "BLOCKED"
    if report.get("verdict") != expected:
        raise CandidateRegistryError("DEPLOYMENT_REVIEW_VERDICT_MISMATCH")
    if not isinstance(report.get("remaining_risks"), list):
        raise CandidateRegistryError("DEPLOYMENT_REVIEW_RISK_SCHEMA_MISMATCH")


__all__ = ["SCHEMA_VERSION", "validate_production_approval_report"]
