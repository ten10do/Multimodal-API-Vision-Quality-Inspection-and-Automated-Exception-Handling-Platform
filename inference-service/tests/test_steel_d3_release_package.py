"""D3 release package freeze, isolation and readiness tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.candidate_registry import CandidateRegistryError, canonical_sha256  # noqa: E402
from steel_patchcore.d3_release_package import (  # noqa: E402
    ReleasePackageRegistry,
    validate_dependency_lock,
    validate_release_manifest,
    validate_release_report,
)

RELEASE_DIR = ROOT / "model-training/registry/steel-patchcore-d3-release/1.3.0"
MANIFEST = RELEASE_DIR / "manifest.json"
LOCK = RELEASE_DIR / "dependency-lock.json"


def test_dependency_lock_is_canonical_and_cuda_pinned():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_dependency_lock(lock)
    assert lock["python"] == "3.11"
    assert lock["cuda_wheel_index"].endswith("/cu128")
    assert "torch==2.11.0+cu128" in lock["declared_packages"]["inference"]


def test_release_manifest_freezes_candidate_without_promotion():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_release_manifest(manifest)
    assert manifest["status"] == "RELEASE_CANDIDATE_PACKAGE"
    assert manifest["candidate"]["model_version"] == "1.3.0-candidate.1"
    assert manifest["threshold"] == 0.8471092581748962
    assert manifest["production_promotion"] is False


def test_release_registry_verifies_all_lineage_and_artifacts():
    package = ReleasePackageRegistry(ROOT).load(MANIFEST)
    assert set(package.artifact_hashes) == {
        "legacy_manifest", "weights", "whitening", "image_bank", "R-L1", "R-L2", "protocol", "investigation_results"
    }


def test_release_manifest_tamper_fails_closed():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(manifest)
    tampered["threshold"] += 0.001
    payload = dict(tampered)
    payload.pop("manifest_payload_sha256")
    tampered["manifest_payload_sha256"] = canonical_sha256(payload)
    with pytest.raises(CandidateRegistryError, match="RELEASE_THRESHOLD_CHANGED"):
        validate_release_manifest(tampered)


def test_release_report_is_derived_and_candidate_only():
    report = {
        "schema_version": "steel_patchcore_d3_release_readiness_v1",
        "candidate_status": "PRODUCTION_CANDIDATE_QUALIFIED",
        "package_status": "RELEASE_CANDIDATE_PACKAGE",
        "gates": {name: {"verdict": "PASS"} for name in (
            "manifest_freeze", "documentation", "clean_environment", "security", "tests"
        )},
        "verdict": "PASS",
        "remaining_risks": [],
        "production_promotion": False,
        "automatic_retraining": False,
    }
    validate_release_report(report)
    assert report["verdict"] == "PASS"
    assert report["candidate_status"] == "PRODUCTION_CANDIDATE_QUALIFIED"
    assert report["package_status"] == "RELEASE_CANDIDATE_PACKAGE"
    assert report["production_promotion"] is False
    assert report["automatic_retraining"] is False


def test_release_documentation_package_is_complete():
    expected = {
        "system-architecture.md", "model-card.md", "deployment-guide.md",
        "operation-manual.md", "troubleshooting-guide.md", "rollback-procedure.md",
    }
    assert expected <= {path.name for path in (ROOT / "docs/release").glob("*.md")}
