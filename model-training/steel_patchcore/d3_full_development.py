"""D3 full-development confirmation primitives (Optimization 1.1).

Freezes the D3 method definition, the full authorized development data bounds
(train_normal 4721 / validation_normal 590 / recovery_dev_anomaly 3333), the
full-D3 weights + split lineage, the frozen confirmation gate, and holdout
fail-closed membership checks. Pure, testable, CPU-only.
"""
from __future__ import annotations

import math

from steel_patchcore.domain_adaptation import (  # noqa: E402
    D2_REFERENCE,
    ADAPTATION_BANK_BUDGET,
    ADAPTATION_SEED,
    EPSILON_FACTOR,
)

D3_FULL_DEV_PROTOCOL_VERSION = "d3_full_development_protocol_v1"

# Frozen D3 diagnostic reference (previous phase, never re-run).
D3_DIAGNOSTIC_CANDIDATE_ID = "D3"
D3_DIAGNOSTIC_AUROC = 0.8208
D3_DIAGNOSTIC_QUARTILES = {"Q1": 0.7341, "Q2": 0.7959, "Q3": 0.8324, "Q4": 0.9209}

# Full authorized development data bounds (frozen).
FULL_TRAIN_NORMAL = 4721
FULL_VALIDATION_NORMAL = 590
FULL_DEV_ANOMALY = 3333

# Full DINOv2 ViT-B/14 weights SHA256 (full, read from committed D3 results).
DINO_B_WEIGHTS_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
SOURCE_SPLIT_SHA256 = "64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07"
RECOVERY_SPLIT_SHA256 = "f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448"
DIAGNOSTIC_MANIFEST_SHA256 = "8d39aec04c35ac7a7edc931c6dfb3494c6ada6a0a6e13e77b89bc894f9919075"

# Frozen confirmation gate (NOT to be moved after seeing results).
D3_FULL_DEV_GATE = {
    "auroc_min": 0.75,
    "small_defect_q1_min": 0.65,
}

# D3 method identity (unchanged from diagnostic D3).
D3_METHOD = {
    "backbone": "DINOv2 ViT-B/14 (dinov2_vitb14)",
    "weights_sha256": DINO_B_WEIGHTS_SHA256,
    "extraction": "forward_features(x)['x_norm_patchtokens'] (CLS/register excluded)",
    "tile": "256x256 -> bilinear 252x252 (18x18 = 324 patch tokens)",
    "embed_dim": 768,
    "adaptation": "train-normal ZCA covariance whitening (768->768, no PCA truncation)",
    "epsilon_rule": "1e-6 * trace(cov) / d",
    "statistics": "streaming Chan-style float64 (never materialize full token matrix)",
    "distance": "per-patch L2 + cosine 1-NN (1 - cos-sim)",
    "bank": "reservoir Algorithm R, budget 50000, seed 42",
    "tiling": "7 tiles x0 in {0,256,512,768,1024,1280,1344}",
    "aggregation": "A0 global max patch distance over 7 tiles",
    "threshold": "max(train-normal image scores), train-only",
    "diagnostic_commit": "85d1457",
}


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def d3_full_development_gate_passed(auroc: float, anomaly_median: float, normal_median: float) -> bool:
    """AUROC >= 0.75 AND anomaly median > normal median (score ordering)."""
    if not all(_finite(v) for v in (auroc, anomaly_median, normal_median)):
        return False
    return auroc >= D3_FULL_DEV_GATE["auroc_min"] and anomaly_median > normal_median


def small_defect_full_dev_signal(q1_auroc: float) -> bool:
    """Secondary: full-dev Q1 AUROC >= 0.65 (never substitutes the primary gate)."""
    return _finite(q1_auroc) and q1_auroc >= D3_FULL_DEV_GATE["small_defect_q1_min"]


def fail_closed_membership(
    train_ids: list[str],
    validation_ids: list[str],
    dev_ids: list[str],
    test_normal_ids: list[str],
    holdout_anomaly_ids: list[str],
) -> tuple[bool, str]:
    """Fail-closed: the worked set must be exactly within the allowed dev sets and
    disjoint from the hard-sealed holdout sets (test_normal + recovery_holdout)."""
    train = list(train_ids)
    val = list(validation_ids)
    dev = list(dev_ids)
    forbidden = set(test_normal_ids) | set(holdout_anomaly_ids)
    worked = set(train) | set(val) | set(dev)
    if len(worked) != len(train) + len(val) + len(dev):
        return False, "dev sets overlap each other"
    if worked & forbidden:
        return False, f"holdout leaked into worked set: {sorted(worked & forbidden)[:5]}"
    if len(train) != FULL_TRAIN_NORMAL:
        return False, f"train_normal count {len(train)} != {FULL_TRAIN_NORMAL}"
    if len(val) != FULL_VALIDATION_NORMAL:
        return False, f"validation_normal count {len(val)} != {FULL_VALIDATION_NORMAL}"
    if len(dev) != FULL_DEV_ANOMALY:
        return False, f"recovery_dev_anomaly count {len(dev)} != {FULL_DEV_ANOMALY}"
    return True, "ok"


__all__ = [
    "ADAPTATION_BANK_BUDGET",
    "ADAPTATION_SEED",
    "D2_REFERENCE",
    "D3_DIAGNOSTIC_AUROC",
    "D3_DIAGNOSTIC_CANDIDATE_ID",
    "D3_DIAGNOSTIC_QUARTILES",
    "D3_FULL_DEV_GATE",
    "D3_FULL_DEV_PROTOCOL_VERSION",
    "D3_METHOD",
    "DIAGNOSTIC_MANIFEST_SHA256",
    "DINO_B_WEIGHTS_SHA256",
    "EPSILON_FACTOR",
    "FULL_DEV_ANOMALY",
    "FULL_TRAIN_NORMAL",
    "FULL_VALIDATION_NORMAL",
    "RECOVERY_SPLIT_SHA256",
    "SOURCE_SPLIT_SHA256",
    "d3_full_development_gate_passed",
    "fail_closed_membership",
    "small_defect_full_dev_signal",
]