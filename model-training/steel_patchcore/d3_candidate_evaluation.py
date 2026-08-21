"""Reproducible, score-only evaluation pipeline for a verified D3 candidate."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from steel_patchcore.candidate_registry import HASH_FIELDS, canonical_sha256, validate_manifest
from steel_patchcore.d3_recovery_holdout import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    evaluate_holdout,
    gate_verdict,
    stratified_bootstrap_auroc,
)

EVALUATION_SCHEMA_VERSION = "steel_patchcore_d3_candidate_evaluation_v1"
ROLES = ("test_normal", "recovery_holdout_anomaly")


class CandidateEvaluationError(RuntimeError):
    pass


def _validated_scores(dataset_manifest: Mapping, score_records: Sequence[Mapping]):
    if set(dataset_manifest) != set(ROLES):
        raise CandidateEvaluationError("DATASET_MANIFEST_ROLE_MISMATCH")
    expected: dict[str, list[str]] = {role: list(dataset_manifest[role]) for role in ROLES}
    if any(len(ids) != len(set(ids)) for ids in expected.values()):
        raise CandidateEvaluationError("DATASET_MANIFEST_DUPLICATE_ID")
    if set(expected[ROLES[0]]) & set(expected[ROLES[1]]):
        raise CandidateEvaluationError("DATASET_MANIFEST_ROLE_OVERLAP")
    allowed = {role: set(ids) for role, ids in expected.items()}
    rows: dict[str, dict[str, Mapping]] = {role: {} for role in ROLES}
    for record in score_records:
        role = record.get("split_role")
        image_id = record.get("image_id")
        score = record.get("score")
        if role not in ROLES or not isinstance(image_id, str) or not image_id:
            raise CandidateEvaluationError("SCORE_RECORD_IDENTITY_INVALID")
        if image_id in rows[role] or image_id not in allowed[role]:
            raise CandidateEvaluationError(f"SCORE_RECORD_MEMBERSHIP_INVALID:{role}:{image_id}")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise CandidateEvaluationError(f"SCORE_RECORD_NONFINITE:{image_id}")
        if role == "recovery_holdout_anomaly" and record.get("quartile") not in (1, 2, 3, 4):
            raise CandidateEvaluationError(f"SCORE_RECORD_QUARTILE_INVALID:{image_id}")
        rows[role][image_id] = record
    for role in ROLES:
        if set(rows[role]) != allowed[role]:
            raise CandidateEvaluationError(f"SCORE_RECORD_INCOMPLETE:{role}")
    normal = np.asarray([rows["test_normal"][image_id]["score"] for image_id in expected["test_normal"]], dtype=np.float64)
    anomalies = [rows["recovery_holdout_anomaly"][image_id] for image_id in expected["recovery_holdout_anomaly"]]
    anomaly = np.asarray([row["score"] for row in anomalies], dtype=np.float64)
    quartiles = np.asarray([row["quartile"] for row in anomalies], dtype=np.int8)
    if set(int(value) for value in quartiles) != {1, 2, 3, 4}:
        raise CandidateEvaluationError("SCORE_RECORD_QUARTILES_INCOMPLETE")
    return normal, anomaly, quartiles


def generate_evaluation_report(
    *,
    dataset_manifest: Mapping,
    score_records: Sequence[Mapping],
    candidate_manifest: Mapping,
    artifact_hashes: Mapping[str, str],
    timestamp: str,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict:
    """Generate metrics from frozen scores; never loads a model or fits artifacts."""
    validate_manifest(candidate_manifest)
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise CandidateEvaluationError("EVALUATION_TIMESTAMP_INVALID") from exc
    if set(artifact_hashes) != set(HASH_FIELDS):
        raise CandidateEvaluationError("ARTIFACT_HASH_SET_MISMATCH")
    for field in HASH_FIELDS:
        if artifact_hashes[field] != candidate_manifest[field]:
            raise CandidateEvaluationError(f"ARTIFACT_HASH_MISMATCH:{field}")
    normal, anomaly, quartiles = _validated_scores(dataset_manifest, score_records)
    threshold = float(candidate_manifest["threshold"])
    metrics = evaluate_holdout(normal, anomaly, threshold, quartiles)
    bootstrap = stratified_bootstrap_auroc(
        normal,
        anomaly,
        seed=BOOTSTRAP_SEED,
        iterations=bootstrap_iterations,
    )
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "model_name": candidate_manifest["model_name"],
        "model_version": candidate_manifest["model_version"],
        "artifact_version": candidate_manifest["artifact_version"],
        "metrics": metrics,
        "bootstrap_ci": bootstrap,
        "verdict": gate_verdict(metrics["image_auroc"], metrics["anomaly_median"], metrics["normal_median"]),
        "lineage": {
            "source_split_sha256": candidate_manifest["source_split_sha256"],
            "recovery_split_sha256": candidate_manifest["recovery_split_sha256"],
        },
        "artifact_hashes": dict(artifact_hashes),
        "timestamp": timestamp,
    }
    reproducible = dict(report)
    reproducible.pop("timestamp")
    report["evaluation_fingerprint"] = canonical_sha256(reproducible)
    return report


def write_evaluation_report(path: Path, report: Mapping) -> None:
    """Atomically persist the JSON evaluation envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


__all__ = [
    "CandidateEvaluationError",
    "EVALUATION_SCHEMA_VERSION",
    "generate_evaluation_report",
    "write_evaluation_report",
]
