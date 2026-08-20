"""Canonical PatchCore reference cross-check primitives.

Freezes C0/C1 candidate identities, the canonical Gate, tiling, and the
train-only calibration rule. Reference-algorithm behavior itself lives in the
installed anomalib 0.7.0 library; this module only holds our adapter-level
frozen semantics (testable without the reference library).
"""
from __future__ import annotations

import json
import math
from hashlib import sha256

CANONICAL_PROTOCOL_VERSION = "canonical_patchcore_protocol_v1"

# C0 = frozen S2 (layer3 + 5x5 avg context) from the spatial-context phase.
C0_AUROC = 0.6029

# C1 canonical configuration (documented; exact algorithm from anomalib source).
CANONICAL_REFERENCE = {
    "name": "anomalib",
    "version": "0.7.0",
    "model": "anomalib.models.patchcore.torch_model.PatchcoreModel",
    "backbone": "wide_resnet50_2",
    "pre_trained": True,
    "layers": ["layer2", "layer3"],
    "input_size": [256, 256],
    "coreset_sampling_ratio": 0.1,
    "num_neighbors": 9,
}

CANONICAL_GATE = {"auroc_min": 0.65, "delta_vs_c0": 0.05, "strong_auroc": 0.75}

# Frozen seven-tile scheme (unchanged across all phases).
TILE_X0 = (0, 256, 512, 768, 1024, 1280, 1344)
IMG_W = 1600
IMG_H = 256


def canonical_gate_passed(c0_auroc: float, c1_auroc: float) -> bool:
    if not (_finite(c0_auroc) and _finite(c1_auroc)):
        return False
    return (
        c1_auroc >= CANONICAL_GATE["auroc_min"]
        and (c1_auroc - c0_auroc) >= CANONICAL_GATE["delta_vs_c0"]
    )


def canonical_strong_signal(c1_auroc: float) -> bool:
    return _finite(c1_auroc) and c1_auroc >= CANONICAL_GATE["strong_auroc"]


def original_score_from_tiles(tile_scores: list[float]) -> float:
    """Original 256x1600 image score = max over the frozen seven tile scores."""
    if not tile_scores:
        raise ValueError("empty tile scores")
    if len(tile_scores) != len(TILE_X0):
        raise ValueError(f"expected {len(TILE_X0)} tile scores, got {len(tile_scores)}")
    return max(tile_scores)


def diagnostic_threshold(train_scores: list[float]) -> float:
    """Train-only calibration: max train-normal image score. No anomaly/val."""
    return max(train_scores)


def serialize_config(config: dict) -> str:
    return json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_sha256(config: dict) -> str:
    return sha256(serialize_config(config).encode("utf-8")).hexdigest()


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)