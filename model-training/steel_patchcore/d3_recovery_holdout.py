"""Frozen D3 one-shot recovery-holdout evaluation primitives.

This module is CPU-only.  It freezes lineage, holdout membership, checkpoint,
metric, bootstrap, quartile, and gate semantics before any holdout image is
opened by the GPU runner.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from steel_patchcore.aggregation import auroc, distribution, normal_vs_quartile_auroc, operating_point

PROTOCOL_VERSION = "d3_recovery_holdout_protocol_v1"
CHECKPOINT_SCHEMA_VERSION = "steel_patchcore_d3_recovery_holdout_checkpoint_v1"
RESULTS_SCHEMA_VERSION = "steel_patchcore_d3_recovery_holdout_results_v1"

EXPECTED_LINEAGE = {
    "baseline_bank_sha256": "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda",
    "source_split_sha256": "64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07",
    "recovery_split_sha256": "f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448",
    "evidence_manifest_sha256": "7067bbc6da747ffa90db9532729f579e19ee90fbcdf3867c96f2dd56a0d10303",
    "dino_weights_sha256": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    "whitening_sha256": "c8d9d2ed39fb7ba6d0013a27beba81e8d7b70c66da0e38b7d19e15ea7cae8c3a",
    "d3_bank_sha256": "40fe43331885422c8a32364a48fc403b766f807f69faafee775a2eb2403cbbda",
    "quartile_manifest_sha256": "8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075",
    "d3_results_sha256": "1d511ccd20a6c007f6f0b298de8e7b5bd649ff6c229a183b60c06dda1bbca35c",
}

FROZEN_THRESHOLD = 0.8471092581748962
FROZEN_QUARTILE_BOUNDARIES = (0.010888671875, 0.02656494140625, 0.07214111328125)
HOLDOUT_COUNTS = {"test_normal": 591, "recovery_holdout_anomaly": 3333}
HOLDOUT_GATE = {"image_auroc_min": 0.75, "require_anomaly_median_gt_normal_median": True}
BOOTSTRAP_SEED = 42
BOOTSTRAP_ITERATIONS = 2000
HOLDOUT_ROLES = tuple(HOLDOUT_COUNTS)


class HoldoutBlocked(RuntimeError):
    """A fail-closed precondition or frozen-protocol violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_lineage(paths: Mapping[str, Path]) -> dict[str, str]:
    """Hash every frozen artifact and fail closed on a missing/mismatched item."""
    if set(paths) != set(EXPECTED_LINEAGE):
        missing = sorted(set(EXPECTED_LINEAGE) - set(paths))
        extra = sorted(set(paths) - set(EXPECTED_LINEAGE))
        raise HoldoutBlocked(f"LINEAGE_PATH_SET_MISMATCH:missing={missing}:extra={extra}")
    actual: dict[str, str] = {}
    for name, expected in EXPECTED_LINEAGE.items():
        path = Path(paths[name])
        if not path.is_file():
            raise HoldoutBlocked(f"ARTIFACT_MISSING:{name}:{path}")
        actual[name] = sha256_file(path)
        if actual[name] != expected:
            raise HoldoutBlocked(f"ARTIFACT_SHA_MISMATCH:{name}:{actual[name]}")
    return actual


def assert_artifacts_unchanged(paths: Mapping[str, Path], before: Mapping[str, str]) -> None:
    after = verify_artifact_lineage(paths)
    if dict(before) != after:
        raise HoldoutBlocked("FROZEN_ARTIFACT_MUTATED_DURING_EVALUATION")


def load_frozen_threshold(results: Mapping) -> float:
    """Load, never calculate, the full-precision threshold from committed D3 results."""
    try:
        verdict = results["verdict"]
        threshold = float(results["full"]["metrics"]["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HoldoutBlocked("FROZEN_THRESHOLD_MISSING") from exc
    if verdict != "D3_FULL_DEVELOPMENT_CONFIRMED":
        raise HoldoutBlocked(f"FROZEN_D3_NOT_CONFIRMED:{verdict}")
    if threshold != FROZEN_THRESHOLD:
        raise HoldoutBlocked(f"FROZEN_THRESHOLD_MISMATCH:{threshold!r}")
    return threshold


def validate_holdout_membership(source: Mapping, recovery: Mapping) -> dict[str, list[str]]:
    """Validate exact holdout membership, uniqueness, and dev/holdout isolation."""
    try:
        splits = source["splits"]
        roles = {
            "test_normal": list(splits["test_normal"]),
            "recovery_holdout_anomaly": list(recovery["recovery_holdout_anomaly"]),
        }
        development = (
            list(splits["train_normal"])
            + list(splits["validation_normal"])
            + list(recovery["recovery_dev_anomaly"])
        )
        source_anomaly = list(splits["test_anomaly"])
    except (KeyError, TypeError) as exc:
        raise HoldoutBlocked("HOLDOUT_SPLIT_MALFORMED") from exc

    for role, expected_count in HOLDOUT_COUNTS.items():
        ids = roles[role]
        if len(ids) != expected_count:
            raise HoldoutBlocked(f"HOLDOUT_COUNT_MISMATCH:{role}:{len(ids)}")
        if len(ids) != len(set(ids)):
            raise HoldoutBlocked(f"HOLDOUT_DUPLICATE_IDS:{role}")
    if len(development) != len(set(development)):
        raise HoldoutBlocked("DEVELOPMENT_DUPLICATE_OR_OVERLAPPING_IDS")
    holdout_all = roles["test_normal"] + roles["recovery_holdout_anomaly"]
    if len(holdout_all) != len(set(holdout_all)):
        raise HoldoutBlocked("HOLDOUT_ROLE_INTERSECTION")
    overlap = set(development) & set(holdout_all)
    if overlap:
        raise HoldoutBlocked(f"DEVELOPMENT_HOLDOUT_INTERSECTION:{sorted(overlap)[:5]}")
    dev_anomaly = set(recovery["recovery_dev_anomaly"])
    holdout_anomaly = set(roles["recovery_holdout_anomaly"])
    if dev_anomaly | holdout_anomaly != set(source_anomaly) or dev_anomaly & holdout_anomaly:
        raise HoldoutBlocked("RECOVERY_ANOMALY_PARTITION_MISMATCH")
    return roles


def a0_global_max(raw_patch_distances) -> float:
    """Frozen A0: the maximum patch distance over all seven tiles."""
    values = np.asarray(raw_patch_distances, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise HoldoutBlocked("A0_SCORE_INPUT_INVALID")
    return float(values.max())


def assign_frozen_quartiles(
    area_ratios, boundaries: tuple[float, float, float] = FROZEN_QUARTILE_BOUNDARIES
) -> np.ndarray:
    """Assign Q1-Q4 using development-frozen boundaries; never fit on holdout."""
    ratios = np.asarray(area_ratios, dtype=np.float64)
    if ratios.ndim != 1 or not np.isfinite(ratios).all():
        raise HoldoutBlocked("AREA_RATIO_INPUT_INVALID")
    q1, q2, q3 = boundaries
    if not (q1 < q2 < q3):
        raise HoldoutBlocked("FROZEN_QUARTILE_BOUNDARIES_INVALID")
    quartiles = np.empty(ratios.size, dtype=np.int8)
    quartiles[ratios < q1] = 1
    quartiles[(ratios >= q1) & (ratios < q2)] = 2
    quartiles[(ratios >= q2) & (ratios < q3)] = 3
    quartiles[ratios >= q3] = 4
    return quartiles


def stratified_bootstrap_auroc(
    normal_scores,
    anomaly_scores,
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict:
    """Report-only class-stratified bootstrap AUROC percentile interval."""
    normal = np.asarray(normal_scores, dtype=np.float64)
    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    if normal.size == 0 or anomaly.size == 0 or iterations <= 0:
        raise HoldoutBlocked("BOOTSTRAP_INPUT_INVALID")
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    labels = np.concatenate([np.zeros(normal.size, dtype=np.int8), np.ones(anomaly.size, dtype=np.int8)])
    for index in range(iterations):
        sampled_normal = normal[rng.integers(0, normal.size, size=normal.size)]
        sampled_anomaly = anomaly[rng.integers(0, anomaly.size, size=anomaly.size)]
        values[index] = auroc(np.concatenate([sampled_normal, sampled_anomaly]), labels)
    return {
        "seed": int(seed),
        "iterations": int(iterations),
        "median": float(np.median(values)),
        "percentile_95_ci": [
            float(np.percentile(values, 2.5, method="linear")),
            float(np.percentile(values, 97.5, method="linear")),
        ],
    }


def evaluate_holdout(normal_scores, anomaly_scores, threshold: float, quartiles) -> dict:
    normal = np.asarray(normal_scores, dtype=np.float64)
    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    qarr = np.asarray(quartiles, dtype=np.int8)
    if normal.size == 0 or anomaly.size == 0 or qarr.shape != anomaly.shape:
        raise HoldoutBlocked("METRIC_INPUT_SHAPE_MISMATCH")
    if not np.isfinite(normal).all() or not np.isfinite(anomaly).all() or not math.isfinite(threshold):
        raise HoldoutBlocked("METRIC_INPUT_NONFINITE")
    image_auroc = auroc(
        np.concatenate([normal, anomaly]),
        np.concatenate([np.zeros(normal.size, dtype=np.int8), np.ones(anomaly.size, dtype=np.int8)]),
    )
    qrows = []
    for q in (1, 2, 3, 4):
        selected = anomaly[qarr == q]
        qrows.append(
            {
                "quartile": q,
                "count": int(selected.size),
                "normal_vs_quartile_auroc": normal_vs_quartile_auroc(normal, selected),
            }
        )
    normal_median = float(np.median(normal))
    anomaly_median = float(np.median(anomaly))
    return {
        "image_auroc": image_auroc,
        "normal_distribution": distribution(normal),
        "anomaly_distribution": distribution(anomaly),
        "normal_median": normal_median,
        "anomaly_median": anomaly_median,
        "anomaly_minus_normal_median": anomaly_median - normal_median,
        "frozen_threshold": float(threshold),
        "operating_point": operating_point(normal, anomaly, threshold),
        "quartiles": qrows,
    }


def gate_verdict(image_auroc: float, anomaly_median: float, normal_median: float) -> str:
    finite = all(math.isfinite(float(v)) for v in (image_auroc, anomaly_median, normal_median))
    passed = finite and image_auroc >= HOLDOUT_GATE["image_auroc_min"] and anomaly_median > normal_median
    return "RECOVERY_HOLDOUT_PASS" if passed else "RECOVERY_HOLDOUT_FAILED"


def new_checkpoint(lineage: Mapping[str, str], threshold: float, timestamp: str) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_lineage": dict(lineage),
        "threshold": float(threshold),
        "completed": {role: {} for role in HOLDOUT_ROLES},
        "updated_at": timestamp,
    }


def validate_checkpoint(
    checkpoint: Mapping,
    expected_lineage: Mapping[str, str],
    expected_roles: Mapping[str, list[str]],
    threshold: float,
) -> dict[str, int]:
    """Reject foreign, recalibrated, duplicate, or role-confused resume state."""
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise HoldoutBlocked("CHECKPOINT_SCHEMA_MISMATCH")
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise HoldoutBlocked("CHECKPOINT_PROTOCOL_MISMATCH")
    if checkpoint.get("artifact_lineage") != dict(expected_lineage):
        raise HoldoutBlocked("CHECKPOINT_LINEAGE_MISMATCH")
    if checkpoint.get("threshold") != float(threshold):
        raise HoldoutBlocked("CHECKPOINT_THRESHOLD_MISMATCH")
    completed = checkpoint.get("completed")
    if not isinstance(completed, dict) or set(completed) != set(HOLDOUT_ROLES):
        raise HoldoutBlocked("CHECKPOINT_ROLE_SET_MISMATCH")
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for role in HOLDOUT_ROLES:
        rows = completed[role]
        if not isinstance(rows, dict):
            raise HoldoutBlocked(f"CHECKPOINT_ROLE_MALFORMED:{role}")
        allowed = set(expected_roles[role])
        for image_id, row in rows.items():
            if image_id in seen:
                raise HoldoutBlocked(f"CHECKPOINT_DUPLICATE_ID:{image_id}")
            seen.add(image_id)
            if image_id not in allowed or not isinstance(row, dict):
                raise HoldoutBlocked(f"CHECKPOINT_MEMBERSHIP_MISMATCH:{role}:{image_id}")
            if row.get("image_id") != image_id or row.get("split_role") != role:
                raise HoldoutBlocked(f"CHECKPOINT_ROLE_BINDING_MISMATCH:{image_id}")
            score = row.get("score")
            prediction = row.get("prediction")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise HoldoutBlocked(f"CHECKPOINT_SCORE_INVALID:{image_id}")
            if prediction not in (0, 1) or prediction != int(float(score) >= threshold):
                raise HoldoutBlocked(f"CHECKPOINT_PREDICTION_INVALID:{image_id}")
        counts[role] = len(rows)
    return counts


def record_checkpoint_result(checkpoint: dict, role: str, image_id: str, score: float) -> None:
    if role not in HOLDOUT_ROLES:
        raise HoldoutBlocked(f"CHECKPOINT_UNKNOWN_ROLE:{role}")
    if any(image_id in checkpoint["completed"][other] for other in HOLDOUT_ROLES):
        raise HoldoutBlocked(f"CHECKPOINT_DUPLICATE_ID:{image_id}")
    if not math.isfinite(float(score)):
        raise HoldoutBlocked(f"CHECKPOINT_SCORE_INVALID:{image_id}")
    threshold = float(checkpoint["threshold"])
    checkpoint["completed"][role][image_id] = {
        "image_id": image_id,
        "split_role": role,
        "score": float(score),
        "prediction": int(float(score) >= threshold),
    }


__all__ = [
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_SCHEMA_VERSION",
    "EXPECTED_LINEAGE",
    "FROZEN_QUARTILE_BOUNDARIES",
    "FROZEN_THRESHOLD",
    "HOLDOUT_COUNTS",
    "HOLDOUT_GATE",
    "HOLDOUT_ROLES",
    "HoldoutBlocked",
    "PROTOCOL_VERSION",
    "RESULTS_SCHEMA_VERSION",
    "a0_global_max",
    "assert_artifacts_unchanged",
    "assign_frozen_quartiles",
    "evaluate_holdout",
    "gate_verdict",
    "load_frozen_threshold",
    "new_checkpoint",
    "record_checkpoint_result",
    "sha256_file",
    "stratified_bootstrap_auroc",
    "validate_checkpoint",
    "validate_holdout_membership",
    "verify_artifact_lineage",
]
