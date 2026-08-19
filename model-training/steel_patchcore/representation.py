"""Deterministic primitives for the Steel PatchCore representation investigation.

Pure, testable, CPU-only primitives: diagnostic-subset construction, frozen
candidate definitions, reservoir sampler, and diagnostic gate semantics. No
model, no GPU, no holdout.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from steel_patchcore.recovery import canonical_sha256

PROTOCOL_VERSION = "representation_protocol_v1"
REPRESENTATION_SEED = 42

SUBSET_SIZES = {
    "train_normal": 1000,
    "validation_normal": 300,
    "recovery_dev_anomaly": 1000,
}
ANOMALY_PER_QUARTILE = 250
ANOMALY_QUARTILES = (1, 2, 3, 4)

# Frozen feature-layer candidates (Stage R).
FEATURE_LAYER_CANDIDATES = (
    {"id": "R0", "layers": ["layer2", "layer3"], "dim": 1536, "note": "current"},
    {"id": "R1", "layers": ["layer2"], "dim": 512, "note": "layer2 only"},
    {"id": "R2", "layers": ["layer3"], "dim": 1024, "note": "layer3 (upsampled) only"},
)

# Frozen normalization candidates (Stage N, only if the R gate fails).
NORMALIZATION_CANDIDATES = (
    {"id": "N0", "kind": "current"},
    {"id": "N1", "kind": "per_patch_l2_cosine"},
    {"id": "N2", "kind": "per_layer_l2_before_concat"},
)

FEATURE_LAYER_GATE = {"auroc_min": 0.60, "delta_vs_r0": 0.10}
NORMALIZATION_GATE = {"auroc_min": 0.60, "delta_vs_n0": 0.10}


def reservoir_from_stream(
    feature_batches: Iterable[np.ndarray],
    budget: int,
    seed: int = REPRESENTATION_SEED,
) -> tuple[np.ndarray, int]:
    """Frozen reservoir (Algorithm R) sampler, matching the 1.0.0 trainer.

    ``feature_batches`` yields (N, D) float arrays; rows are drawn in stream
    order. Returns (reservoir [budget, D] float32, seen patches).
    """
    rng = np.random.default_rng(seed)
    reservoir: np.ndarray | None = None
    seen = 0
    for batch in feature_batches:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch[np.newaxis, :]
        if batch.ndim != 2:
            raise ValueError("feature batches must be (N, D)")
        if reservoir is None:
            reservoir = np.zeros((budget, batch.shape[1]), dtype=np.float32)
        elif batch.shape[1] != reservoir.shape[1]:
            raise ValueError("feature dimension changed mid-stream")
        for feature in batch:
            seen += 1
            if seen <= budget:
                reservoir[seen - 1] = feature
            else:
                j = int(rng.integers(0, seen))
                if j < budget:
                    reservoir[j] = feature
    if reservoir is None:
        raise ValueError("feature stream was empty")
    return reservoir, seen


def build_representation_subset_manifest(
    *,
    source_splits: Mapping[str, list[str]],
    recovery_dev_anomaly: list[str],
    recovery_holdout_anomaly: list[str],
    test_normal: list[str],
    area_ratios: Mapping[str, float],
    source_split_sha256: str,
    recovery_split_sha256: str,
    created_at: str,
) -> dict:
    """Deterministically build the frozen diagnostic subset manifest.

    - train_normal_subset = first 1000 of source order
    - validation_normal_subset = first 300 of source order
    - dev_anomaly_subset = 250 per defect-area quartile, in canonical order
    """
    train = list(source_splits["train_normal"])
    validation = list(source_splits["validation_normal"])
    if len(train) < SUBSET_SIZES["train_normal"]:
        raise ValueError("train_normal too small for subset")
    if len(validation) < SUBSET_SIZES["validation_normal"]:
        raise ValueError("validation_normal too small for subset")
    dev = list(recovery_dev_anomaly)
    if len(dev) < SUBSET_SIZES["recovery_dev_anomaly"]:
        raise ValueError("recovery_dev_anomaly too small for subset")
    if len(set(dev)) != len(dev):
        raise ValueError("recovery_dev_anomaly has duplicates")

    missing = [i for i in dev if i not in area_ratios]
    if missing:
        raise ValueError(f"missing area ratios for {len(missing)} dev anomalies")

    ratios = np.asarray([float(area_ratios[i]) for i in dev], dtype=np.float64)
    q1, q2, q3 = (float(v) for v in np.quantile(ratios, [0.25, 0.5, 0.75], method="linear"))
    quartile_of = np.empty(len(dev), dtype=np.int8)
    quartile_of[ratios < q1] = 1
    quartile_of[(ratios >= q1) & (ratios < q2)] = 2
    quartile_of[(ratios >= q2) & (ratios < q3)] = 3
    quartile_of[ratios >= q3] = 4

    stratified: list[str] = []
    quartile_counts: dict[int, int] = {}
    for q in ANOMALY_QUARTILES:
        group = [i for i, g in zip(dev, quartile_of) if g == q]
        if len(group) < ANOMALY_PER_QUARTILE:
            raise ValueError(f"quartile {q} too small for stratified subset")
        picked = group[:ANOMALY_PER_QUARTILE]
        stratified.extend(picked)
        quartile_counts[q] = len(picked)

    train_subset = train[: SUBSET_SIZES["train_normal"]]
    val_subset = validation[: SUBSET_SIZES["validation_normal"]]
    anomaly_subset = stratified

    all_dev_ids = train_subset + val_subset + anomaly_subset
    if len(all_dev_ids) != len(set(all_dev_ids)):
        raise ValueError("subset IDs are not unique")
    holdout = set(test_normal) | set(recovery_holdout_anomaly)
    if set(all_dev_ids) & holdout:
        raise ValueError("subset intersects sealed holdout")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": REPRESENTATION_SEED,
        "created_at": created_at,
        "source_split_sha256": source_split_sha256,
        "recovery_split_sha256": recovery_split_sha256,
        "subset_counts": dict(SUBSET_SIZES),
        "anomaly_per_quartile": ANOMALY_PER_QUARTILE,
        "anomaly_quartile_boundaries": {"q1": q1, "q2": q2, "q3": q3},
        "train_normal_subset": train_subset,
        "validation_normal_subset": val_subset,
        "recovery_dev_anomaly_subset": anomaly_subset,
        "anomaly_quartile_counts": quartile_counts,
        "holdout_access_count": 0,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def feature_layer_gate_passed(r0_auroc: float, candidate_auroc: float) -> bool:
    if not np.isfinite(r0_auroc) or not np.isfinite(candidate_auroc):
        return False
    return (
        candidate_auroc >= FEATURE_LAYER_GATE["auroc_min"]
        and (candidate_auroc - r0_auroc) >= FEATURE_LAYER_GATE["delta_vs_r0"]
    )


def normalization_gate_passed(n0_auroc: float, candidate_auroc: float) -> bool:
    if not np.isfinite(n0_auroc) or not np.isfinite(candidate_auroc):
        return False
    return (
        candidate_auroc >= NORMALIZATION_GATE["auroc_min"]
        and (candidate_auroc - n0_auroc) >= NORMALIZATION_GATE["delta_vs_n0"]
    )