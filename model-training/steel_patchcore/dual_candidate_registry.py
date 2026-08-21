"""Fail-closed loader for the versioned D3 dual-branch candidate."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from steel_patchcore.candidate_registry import (
    MODEL_NAME,
    CandidateRegistry,
    CandidateRegistryError,
    LoadedCandidate,
    canonical_sha256,
    resolve_uri,
    sha256_file,
)
from steel_patchcore.d3_localization_representation import REPRESENTATION_SPECS
from steel_patchcore.d3_recovery_holdout import FROZEN_THRESHOLD

SCHEMA_VERSION = "steel_patchcore_d3_dual_candidate_manifest_v1"
MODEL_VERSION = "1.3.0-candidate.1"
ALLOWED_STATUS = "CANDIDATE"
LOCALIZATION_BRANCH = "R-L3"


@dataclass(frozen=True)
class LoadedDualCandidate:
    manifest: dict
    image_candidate: LoadedCandidate
    localization_banks: dict[str, np.ndarray]
    paths: dict[str, Path]
    artifact_hashes: dict[str, str]

    @property
    def threshold(self) -> float:
        return float(self.image_candidate.threshold)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def localization_feature_payload() -> dict:
    return {
        "branch": LOCALIZATION_BRANCH,
        "fusion": "equal arithmetic mean of resized raw cosine-distance maps",
        "components": {name: REPRESENTATION_SPECS[name] for name in ("R-L1", "R-L2")},
        "tile_stitching": "mean",
    }


def localization_bank_bundle_sha256(r1_sha256: str, r2_sha256: str) -> str:
    return canonical_sha256({"R-L1": r1_sha256, "R-L2": r2_sha256})


def validate_dual_manifest(manifest: Mapping) -> None:
    required = {
        "schema_version", "status", "model_name", "model_version", "artifact_version",
        "created_at", "production_promotion", "image_branch", "localization_branch",
        "hashes", "evaluation_gate", "manifest_payload_sha256",
    }
    if set(manifest) != required:
        raise CandidateRegistryError("DUAL_MANIFEST_FIELD_SET_MISMATCH")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise CandidateRegistryError("DUAL_MANIFEST_SCHEMA_MISMATCH")
    if manifest["status"] != ALLOWED_STATUS or manifest["production_promotion"] is not False:
        raise CandidateRegistryError("DUAL_CANDIDATE_ONLY_STATUS_REQUIRED")
    if manifest["model_name"] != MODEL_NAME or manifest["model_version"] != MODEL_VERSION:
        raise CandidateRegistryError("DUAL_CANDIDATE_IDENTITY_MISMATCH")

    image = manifest["image_branch"]
    if not isinstance(image, dict) or image.get("branch") != "D3-ZCA" or image.get("aggregation") != "A0":
        raise CandidateRegistryError("DUAL_IMAGE_BRANCH_MISMATCH")
    if image.get("distance") != "cosine-1NN" or image.get("threshold") != FROZEN_THRESHOLD:
        raise CandidateRegistryError("DUAL_IMAGE_SCORING_CHANGED")
    localization = manifest["localization_branch"]
    if not isinstance(localization, dict) or localization.get("branch") != LOCALIZATION_BRANCH:
        raise CandidateRegistryError("DUAL_LOCALIZATION_BRANCH_MISMATCH")
    if localization.get("feature_protocol") != localization_feature_payload():
        raise CandidateRegistryError("DUAL_LOCALIZATION_FEATURE_MISMATCH")

    hashes = manifest["hashes"]
    required_hashes = {
        "model_sha256", "feature_sha256", "image_bank_sha256", "localization_bank_sha256",
        "whitening_sha256", "protocol_sha256",
    }
    if not isinstance(hashes, dict) or set(hashes) != required_hashes or not all(_is_sha256(v) for v in hashes.values()):
        raise CandidateRegistryError("DUAL_HASH_SCHEMA_MISMATCH")
    if hashes["feature_sha256"] != canonical_sha256(localization_feature_payload()):
        raise CandidateRegistryError("DUAL_FEATURE_HASH_MISMATCH")
    if hashes["model_sha256"] != image.get("weights_sha256"):
        raise CandidateRegistryError("DUAL_MODEL_HASH_LINEAGE_MISMATCH")
    if hashes["image_bank_sha256"] != image.get("bank_sha256"):
        raise CandidateRegistryError("DUAL_IMAGE_BANK_HASH_LINEAGE_MISMATCH")
    if hashes["whitening_sha256"] != image.get("whitening_sha256"):
        raise CandidateRegistryError("DUAL_WHITENING_HASH_LINEAGE_MISMATCH")
    banks = localization.get("banks", {})
    if set(banks) != {"R-L1", "R-L2"}:
        raise CandidateRegistryError("DUAL_LOCALIZATION_BANK_SET_MISMATCH")
    for name, expected_dim in (("R-L1", 768), ("R-L2", 768)):
        entry = banks[name]
        if set(entry) != {"uri", "sha256", "rows", "dimension"} or entry["rows"] != 50000 or entry["dimension"] != expected_dim:
            raise CandidateRegistryError(f"DUAL_LOCALIZATION_BANK_SCHEMA_MISMATCH:{name}")
        if not isinstance(entry["uri"], str) or not _is_sha256(entry["sha256"]):
            raise CandidateRegistryError(f"DUAL_LOCALIZATION_BANK_REFERENCE_INVALID:{name}")
    if hashes["localization_bank_sha256"] != localization_bank_bundle_sha256(
        banks["R-L1"]["sha256"], banks["R-L2"]["sha256"]
    ):
        raise CandidateRegistryError("DUAL_LOCALIZATION_BANK_BUNDLE_MISMATCH")
    gate = manifest["evaluation_gate"]
    if gate != {
        "image_auroc": 0.8179071714278028,
        "pixel_auroc": 0.9241393857425543,
        "aupro": 0.7993981069909584,
        "verdict": "PASS",
    }:
        raise CandidateRegistryError("DUAL_EVALUATION_GATE_MISMATCH")
    payload = dict(manifest)
    recorded = payload.pop("manifest_payload_sha256")
    if not _is_sha256(recorded) or canonical_sha256(payload) != recorded:
        raise CandidateRegistryError("DUAL_MANIFEST_PAYLOAD_SHA256_MISMATCH")


class DualCandidateRegistry:
    """Loads an explicitly selected dual manifest; it has no promotion operation."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def load_manifest(self, manifest_path: str | Path) -> dict:
        path = Path(manifest_path).resolve()
        try:
            path.relative_to(self.project_root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise CandidateRegistryError(f"DUAL_MANIFEST_UNREADABLE:{path}") from exc
        validate_dual_manifest(manifest)
        return manifest

    def verify_artifacts(self, manifest: Mapping) -> tuple[dict[str, Path], dict[str, str]]:
        validate_dual_manifest(manifest)
        image = manifest["image_branch"]
        localization = manifest["localization_branch"]
        references = {
            "legacy_manifest": (image["manifest_uri"], image["manifest_sha256"]),
            "weights": (image["weights_uri"], image["weights_sha256"]),
            "whitening": (image["whitening_uri"], image["whitening_sha256"]),
            "image_bank": (image["bank_uri"], image["bank_sha256"]),
            "R-L1": (localization["banks"]["R-L1"]["uri"], localization["banks"]["R-L1"]["sha256"]),
            "R-L2": (localization["banks"]["R-L2"]["uri"], localization["banks"]["R-L2"]["sha256"]),
            "protocol": (localization["protocol_uri"], manifest["hashes"]["protocol_sha256"]),
            "investigation_results": (localization["evidence_uri"], localization["evidence_sha256"]),
        }
        paths: dict[str, Path] = {}
        actual: dict[str, str] = {}
        for role, (uri, expected) in references.items():
            if not isinstance(uri, str) or not _is_sha256(expected):
                raise CandidateRegistryError(f"DUAL_ARTIFACT_REFERENCE_INVALID:{role}")
            path = resolve_uri(self.project_root, uri)
            if not path.is_file():
                raise CandidateRegistryError(f"DUAL_ARTIFACT_MISSING:{role}:{path}")
            digest = sha256_file(path)
            if digest != expected:
                raise CandidateRegistryError(f"DUAL_ARTIFACT_SHA_MISMATCH:{role}:{digest}")
            paths[role] = path
            actual[role] = digest
        return paths, actual

    def load_artifact(self, manifest_path: str | Path) -> LoadedDualCandidate:
        manifest = self.load_manifest(manifest_path)
        paths, before = self.verify_artifacts(manifest)
        legacy = json.loads(paths["legacy_manifest"].read_text(encoding="utf-8"))
        image = manifest["image_branch"]
        lineage = {
            "model_version": legacy.get("model_version") == image["model_version"],
            "artifact_version": legacy.get("artifact_version") == image["artifact_version"],
            "threshold": legacy.get("threshold") == image["threshold"],
            "weights": legacy.get("weights_sha256") == image["weights_sha256"],
            "whitening": legacy.get("whitening_sha256") == image["whitening_sha256"],
            "bank": legacy.get("bank_sha256") == image["bank_sha256"],
        }
        if not all(lineage.values()):
            failed = ",".join(name for name, passed in lineage.items() if not passed)
            raise CandidateRegistryError(f"DUAL_IMAGE_LINEAGE_MISMATCH:{failed}")
        image_candidate = CandidateRegistry(
            self.project_root / "model-training/registry", self.project_root
        ).load_artifact(MODEL_NAME)
        banks: dict[str, np.ndarray] = {}
        for name in ("R-L1", "R-L2"):
            with np.load(paths[name], allow_pickle=False) as payload:
                bank = payload["features"].astype(np.float32)
            if bank.shape != (50000, 768) or not np.isfinite(bank).all():
                raise CandidateRegistryError(f"DUAL_LOCALIZATION_BANK_INVALID:{name}:{bank.shape}")
            bank.flags.writeable = False
            banks[name] = bank
        after_paths, after = self.verify_artifacts(manifest)
        if after != before or after_paths != paths:
            raise CandidateRegistryError("DUAL_ARTIFACT_MUTATED_DURING_LOAD")
        return LoadedDualCandidate(dict(manifest), image_candidate, banks, paths, before)


__all__ = [
    "ALLOWED_STATUS", "DualCandidateRegistry", "LOCALIZATION_BRANCH", "LoadedDualCandidate",
    "MODEL_VERSION", "SCHEMA_VERSION", "localization_bank_bundle_sha256",
    "localization_feature_payload", "validate_dual_manifest",
]
