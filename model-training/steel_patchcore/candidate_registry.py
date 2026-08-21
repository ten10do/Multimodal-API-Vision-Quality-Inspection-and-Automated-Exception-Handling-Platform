"""Fail-closed, candidate-only registry for the frozen steel D3 artifact."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

from steel_patchcore.d3_recovery_holdout import FROZEN_THRESHOLD

SCHEMA_VERSION = "steel_patchcore_d3_candidate_manifest_v1"
MODEL_NAME = "steel-patchcore-d3-candidate"
ALLOWED_STATUS = "CANDIDATE"
HASH_FIELDS = (
    "weights_sha256",
    "whitening_sha256",
    "bank_sha256",
    "source_split_sha256",
    "recovery_split_sha256",
    "full_development_results_sha256",
    "recovery_holdout_results_sha256",
)
URI_FOR_HASH = {
    "weights_sha256": "weights_uri",
    "whitening_sha256": "whitening_uri",
    "bank_sha256": "bank_uri",
    "source_split_sha256": "source_split_uri",
    "recovery_split_sha256": "recovery_split_uri",
    "full_development_results_sha256": "full_development_results_uri",
    "recovery_holdout_results_sha256": "recovery_holdout_results_uri",
}
REQUIRED_FIELDS = {
    "schema_version",
    "status",
    "model_name",
    "model_version",
    "artifact_version",
    "backbone",
    "weights_uri",
    "weights_sha256",
    "whitening_uri",
    "whitening_sha256",
    "bank_uri",
    "bank_sha256",
    "source_split_uri",
    "source_split_sha256",
    "recovery_split_uri",
    "recovery_split_sha256",
    "full_development_results_uri",
    "full_development_results_sha256",
    "recovery_holdout_results_uri",
    "recovery_holdout_results_sha256",
    "threshold",
    "aggregation",
    "distance",
    "tile_config",
    "embedding_dim",
    "created_at",
    "evaluation_evidence",
    "registration_evidence",
    "production_promotion",
    "manifest_payload_sha256",
}


class CandidateRegistryError(RuntimeError):
    """A schema, lineage, integrity, gate, or candidate-only violation."""


def canonical_sha256(payload: Mapping) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_manifest(manifest: Mapping) -> None:
    """Validate the exact v1 schema and all frozen D3 semantic constants."""
    if set(manifest) != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - set(manifest))
        extra = sorted(set(manifest) - REQUIRED_FIELDS)
        raise CandidateRegistryError(f"MANIFEST_FIELD_SET_MISMATCH:missing={missing}:extra={extra}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise CandidateRegistryError("MANIFEST_SCHEMA_MISMATCH")
    if manifest["status"] != ALLOWED_STATUS or manifest["production_promotion"] is not False:
        raise CandidateRegistryError("CANDIDATE_ONLY_STATUS_REQUIRED")
    if manifest["model_name"] != MODEL_NAME:
        raise CandidateRegistryError("MODEL_NAME_MISMATCH")
    if not isinstance(manifest["model_version"], str) or not manifest["model_version"]:
        raise CandidateRegistryError("MODEL_VERSION_INVALID")
    if not isinstance(manifest["artifact_version"], str) or not manifest["artifact_version"]:
        raise CandidateRegistryError("ARTIFACT_VERSION_INVALID")
    if manifest["backbone"] != "dinov2_vitb14":
        raise CandidateRegistryError("BACKBONE_MISMATCH")
    if manifest["aggregation"] != "A0" or manifest["distance"] != "cosine-1NN":
        raise CandidateRegistryError("SCORING_SEMANTICS_MISMATCH")
    if manifest["embedding_dim"] != 768:
        raise CandidateRegistryError("EMBEDDING_DIM_MISMATCH")
    if manifest["threshold"] != FROZEN_THRESHOLD:
        raise CandidateRegistryError("THRESHOLD_MISMATCH")
    expected_tiles = {
        "original_size": [256, 1600],
        "tile_size": 256,
        "model_input_size": 252,
        "x_offsets": [0, 256, 512, 768, 1024, 1280, 1344],
        "patch_grid": [18, 18],
        "heatmap_overlap": "mean",
    }
    if manifest["tile_config"] != expected_tiles:
        raise CandidateRegistryError("TILE_CONFIG_MISMATCH")
    if any(not _is_sha256(manifest[field]) for field in HASH_FIELDS):
        raise CandidateRegistryError("ARTIFACT_SHA256_INVALID")
    for field in URI_FOR_HASH.values():
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise CandidateRegistryError(f"ARTIFACT_URI_INVALID:{field}")
    try:
        datetime.fromisoformat(str(manifest["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidateRegistryError("CREATED_AT_INVALID") from exc
    evidence = manifest["evaluation_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"full_development", "recovery_holdout"}:
        raise CandidateRegistryError("EVALUATION_EVIDENCE_SCHEMA_MISMATCH")
    full = evidence["full_development"]
    holdout = evidence["recovery_holdout"]
    if not isinstance(full, dict) or set(full) != {"verdict", "image_auroc", "q1_q4"}:
        raise CandidateRegistryError("FULL_DEVELOPMENT_EVIDENCE_SCHEMA_MISMATCH")
    if not isinstance(holdout, dict) or set(holdout) != {"verdict", "image_auroc", "bootstrap_95_ci", "q1_q4"}:
        raise CandidateRegistryError("RECOVERY_HOLDOUT_EVIDENCE_SCHEMA_MISMATCH")
    if not isinstance(full["q1_q4"], list) or len(full["q1_q4"]) != 4:
        raise CandidateRegistryError("FULL_DEVELOPMENT_QUARTILES_SCHEMA_MISMATCH")
    if not isinstance(holdout["q1_q4"], list) or len(holdout["q1_q4"]) != 4:
        raise CandidateRegistryError("RECOVERY_HOLDOUT_QUARTILES_SCHEMA_MISMATCH")
    if not isinstance(holdout["bootstrap_95_ci"], list) or len(holdout["bootstrap_95_ci"]) != 2:
        raise CandidateRegistryError("RECOVERY_HOLDOUT_BOOTSTRAP_SCHEMA_MISMATCH")
    evidence_numbers = [full["image_auroc"], *full["q1_q4"], holdout["image_auroc"], *holdout["bootstrap_95_ci"], *holdout["q1_q4"]]
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in evidence_numbers):
        raise CandidateRegistryError("EVALUATION_EVIDENCE_NONFINITE")
    registration = manifest["registration_evidence"]
    if not isinstance(registration, dict) or set(registration) != {"candidate_gate", "tests", "test_command"}:
        raise CandidateRegistryError("REGISTRATION_EVIDENCE_SCHEMA_MISMATCH")
    if registration["candidate_gate"] != "PASS" or registration["tests"] != "PASS":
        raise CandidateRegistryError("REGISTRATION_EVIDENCE_NOT_PASS")
    if not isinstance(registration["test_command"], str) or not registration["test_command"]:
        raise CandidateRegistryError("REGISTRATION_TEST_COMMAND_INVALID")
    payload = dict(manifest)
    recorded = payload.pop("manifest_payload_sha256")
    if not _is_sha256(recorded) or canonical_sha256(payload) != recorded:
        raise CandidateRegistryError("MANIFEST_PAYLOAD_SHA256_MISMATCH")


def resolve_uri(project_root: Path, uri: str) -> Path:
    """Resolve project-relative or home-relative URIs without path traversal."""
    if uri.startswith("~/") or uri.startswith("~\\"):
        return Path(uri).expanduser().resolve()
    raw = Path(uri)
    if raw.is_absolute():
        raise CandidateRegistryError(f"ABSOLUTE_ARTIFACT_URI_FORBIDDEN:{uri}")
    root = project_root.resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CandidateRegistryError(f"ARTIFACT_URI_ESCAPES_PROJECT:{uri}") from exc
    return resolved


@dataclass(frozen=True)
class ArtifactVerification:
    passed: bool
    hashes: dict[str, str]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class CandidateGateResult:
    passed: bool
    checks: tuple[dict, ...]
    blocked: tuple[str, ...]


@dataclass(frozen=True)
class LoadedCandidate:
    manifest: dict
    paths: dict[str, Path]
    artifact_hashes: dict[str, str]
    bank: np.ndarray
    whitening_mean: np.ndarray
    whitening_matrix: np.ndarray

    @property
    def threshold(self) -> float:
        return float(self.manifest["threshold"])


def evaluate_candidate_gate(
    manifest: Mapping,
    verification: ArtifactVerification,
    *,
    tests_passed: bool,
) -> CandidateGateResult:
    """D3 CANDIDATE gate. It has no production transition by construction."""
    full = manifest.get("evaluation_evidence", {}).get("full_development", {})
    holdout = manifest.get("evaluation_evidence", {}).get("recovery_holdout", {})
    checks = (
        {"check": "artifact_integrity", "passed": verification.passed},
        {
            "check": "full_development_auroc",
            "passed": isinstance(full.get("image_auroc"), (int, float)) and full["image_auroc"] >= 0.75,
            "got": full.get("image_auroc"),
            "required": ">=0.75",
        },
        {
            "check": "recovery_holdout",
            "passed": holdout.get("verdict") == "RECOVERY_HOLDOUT_PASS",
            "got": holdout.get("verdict"),
            "required": "RECOVERY_HOLDOUT_PASS",
        },
        {"check": "lineage", "passed": verification.passed and len(verification.hashes) == len(HASH_FIELDS)},
        {"check": "tests", "passed": bool(tests_passed)},
        {
            "check": "candidate_only",
            "passed": manifest.get("status") == ALLOWED_STATUS and manifest.get("production_promotion") is False,
        },
    )
    blocked = tuple(str(row["check"]) for row in checks if not row["passed"])
    return CandidateGateResult(not blocked, checks, blocked)


class CandidateRegistry:
    def __init__(self, registry_root: Path, project_root: Path) -> None:
        self.registry_root = registry_root.resolve()
        self.project_root = project_root.resolve()

    def manifest_path(self, model_name: str = MODEL_NAME) -> Path:
        if model_name != MODEL_NAME:
            raise CandidateRegistryError("UNKNOWN_CANDIDATE")
        return self.registry_root / model_name / "manifest.json"

    def load_manifest(self, model_name: str = MODEL_NAME) -> dict:
        path = self.manifest_path(model_name)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CandidateRegistryError(f"MANIFEST_UNREADABLE:{path}") from exc
        validate_manifest(manifest)
        return manifest

    def verify_artifact(self, manifest: Mapping) -> ArtifactVerification:
        validate_manifest(manifest)
        hashes: dict[str, str] = {}
        errors: list[str] = []
        for hash_field, uri_field in URI_FOR_HASH.items():
            path = resolve_uri(self.project_root, str(manifest[uri_field]))
            if not path.is_file():
                errors.append(f"ARTIFACT_MISSING:{uri_field}:{path}")
                continue
            actual = sha256_file(path)
            hashes[hash_field] = actual
            if actual != manifest[hash_field]:
                errors.append(f"ARTIFACT_SHA_MISMATCH:{hash_field}:{actual}")
        if not errors:
            errors.extend(self._lineage_errors(manifest))
        return ArtifactVerification(not errors, hashes, tuple(errors))

    def _lineage_errors(self, manifest: Mapping) -> list[str]:
        """Bind model artifacts, splits, threshold, evidence, and verdicts."""
        try:
            full_path = resolve_uri(self.project_root, str(manifest["full_development_results_uri"]))
            holdout_path = resolve_uri(self.project_root, str(manifest["recovery_holdout_results_uri"]))
            full = json.loads(full_path.read_text(encoding="utf-8"))
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, CandidateRegistryError) as exc:
            return [f"EVIDENCE_LINEAGE_UNREADABLE:{type(exc).__name__}"]
        whitening = full.get("full", {}).get("whitening_manifest", {})
        checks = {
            "full_verdict": full.get("verdict") == "D3_FULL_DEVELOPMENT_CONFIRMED",
            "full_auroc": full.get("full", {}).get("metrics", {}).get("image_auroc")
            == manifest["evaluation_evidence"]["full_development"]["image_auroc"],
            "full_threshold": full.get("full", {}).get("metrics", {}).get("threshold") == manifest["threshold"],
            "full_bank": full.get("full", {}).get("bank_sha256") == manifest["bank_sha256"],
            "whitening": whitening.get("whitening_sha256") == manifest["whitening_sha256"],
            "weights": whitening.get("dino_weights_sha256") == manifest["weights_sha256"],
            "source_split": whitening.get("source_split_sha256") == manifest["source_split_sha256"],
            "recovery_split": whitening.get("recovery_split_sha256") == manifest["recovery_split_sha256"],
            "holdout_verdict": holdout.get("verdict") == "RECOVERY_HOLDOUT_PASS",
            "holdout_auroc": holdout.get("metrics", {}).get("image_auroc")
            == manifest["evaluation_evidence"]["recovery_holdout"]["image_auroc"],
            "holdout_threshold": holdout.get("metrics", {}).get("frozen_threshold") == manifest["threshold"],
            "holdout_bank": holdout.get("artifact_lineage", {}).get("d3_bank_sha256") == manifest["bank_sha256"],
            "holdout_whitening": holdout.get("artifact_lineage", {}).get("whitening_sha256")
            == manifest["whitening_sha256"],
            "holdout_weights": holdout.get("artifact_lineage", {}).get("dino_weights_sha256")
            == manifest["weights_sha256"],
            "holdout_source_split": holdout.get("artifact_lineage", {}).get("source_split_sha256")
            == manifest["source_split_sha256"],
            "holdout_recovery_split": holdout.get("artifact_lineage", {}).get("recovery_split_sha256")
            == manifest["recovery_split_sha256"],
        }
        return [f"LINEAGE_MISMATCH:{name}" for name, passed in checks.items() if not passed]

    def register(self, manifest: Mapping, *, tests_passed: bool) -> Path:
        """Register exactly one verified CANDIDATE; PRODUCTION is impossible."""
        validate_manifest(manifest)
        verification = self.verify_artifact(manifest)
        gate = evaluate_candidate_gate(manifest, verification, tests_passed=tests_passed)
        if not gate.passed:
            raise CandidateRegistryError(f"CANDIDATE_GATE_BLOCKED:{','.join(gate.blocked)}")
        destination = self.manifest_path(str(manifest["model_name"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != dict(manifest):
                raise CandidateRegistryError("CANDIDATE_ALREADY_REGISTERED_DIFFERENT_MANIFEST")
            return destination
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, destination)
        return destination

    def load_artifact(self, model_name: str = MODEL_NAME) -> LoadedCandidate:
        """Verify gate and hashes before loading immutable arrays into memory."""
        manifest = self.load_manifest(model_name)
        verification = self.verify_artifact(manifest)
        tests_passed = manifest["registration_evidence"]["tests"] == "PASS"
        gate = evaluate_candidate_gate(manifest, verification, tests_passed=tests_passed)
        if not gate.passed:
            raise CandidateRegistryError(f"CANDIDATE_LOAD_BLOCKED:{','.join(gate.blocked)}")
        paths = {uri_field: resolve_uri(self.project_root, str(manifest[uri_field])) for uri_field in URI_FOR_HASH.values()}
        with np.load(paths["whitening_uri"], allow_pickle=False) as whitening:
            mean = whitening["mean"].astype(np.float32)
            matrix = whitening["whitening_matrix"].astype(np.float32)
        with np.load(paths["bank_uri"], allow_pickle=False) as bank_file:
            bank = bank_file["features"].astype(np.float32)
        if mean.shape != (768,) or matrix.shape != (768, 768):
            raise CandidateRegistryError(f"WHITENING_SHAPE_MISMATCH:{mean.shape}:{matrix.shape}")
        if bank.shape != (50000, 768):
            raise CandidateRegistryError(f"BANK_SHAPE_MISMATCH:{bank.shape}")
        if not np.isfinite(mean).all() or not np.isfinite(matrix).all() or not np.isfinite(bank).all():
            raise CandidateRegistryError("ARTIFACT_NONFINITE")
        for array in (mean, matrix, bank):
            array.flags.writeable = False
        after = self.verify_artifact(manifest)
        if not after.passed or after.hashes != verification.hashes:
            raise CandidateRegistryError("ARTIFACT_MUTATED_DURING_LOAD")
        return LoadedCandidate(dict(manifest), paths, verification.hashes, bank, mean, matrix)


__all__ = [
    "ALLOWED_STATUS",
    "ArtifactVerification",
    "CandidateGateResult",
    "CandidateRegistry",
    "CandidateRegistryError",
    "HASH_FIELDS",
    "LoadedCandidate",
    "MODEL_NAME",
    "SCHEMA_VERSION",
    "canonical_sha256",
    "evaluate_candidate_gate",
    "resolve_uri",
    "sha256_file",
    "validate_manifest",
]
