"""Spatial scale & local context primitives for Steel PatchCore (Stage S/P).

Frozen, deterministic, GPU-agnostic definitions for the spatial/context
experiments. No sampling strategy change, no backbone change, no holdout.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SPATIAL_CONTEXT_PROTOCOL_VERSION = "spatial_context_protocol_v1"
SPATIAL_SEED = 42
BANK_BUDGET = 50_000

# Stage S: local context on top of the strongest single-layer representation
# (layer3, current). S0 is the reference (= R2 from the representation phase).
SPATIAL_CONTEXT_CANDIDATES = [
    {"id": "S0", "layer": "layer3", "context": None, "dim": 1024, "desc": "layer3-only current (R2 reference)"},
    {"id": "S1", "layer": "layer3", "context": 3, "dim": 1024, "desc": "layer3 + 3x3 avg context"},
    {"id": "S2", "layer": "layer3", "context": 5, "dim": 1024, "desc": "layer3 + 5x5 avg context"},
]

# Stage P: patch-scale diagnostic (run only if the S gate fails).
PATCH_SCALE_CANDIDATES = [
    {"id": "P0", "layer": "layer3", "context": None, "dim": 1024, "desc": "S0 reference"},
    {"id": "P1", "layer": "layer2", "context": 3, "dim": 512, "desc": "layer2 + 3x3 avg context"},
]

SPATIAL_CONTEXT_GATE = {"auroc_min": 0.65, "delta_vs_s0": 0.10}
PATCH_SCALE_GATE = {"auroc_min": 0.60, "delta_vs_r1": 0.10, "q1_delta_vs_r1": 0.10}


def average_pool_same(feature_map: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """2D average pooling, stride 1, same spatial grid.

    Padding semantics: zero padding of `kernel_size // 2` on each side to
    preserve the exact input HxW; `count_include_pad=False` so the mean is over
    the valid (non-padded) neighbors in the kxk window. Feature channels are
    unchanged.
    """
    if kernel_size is None or kernel_size <= 1:
        return feature_map
    assert kernel_size % 2 == 1, "kernel_size must be odd for same-grid pooling"
    padding = kernel_size // 2
    return F.avg_pool2d(
        feature_map, kernel_size=kernel_size, stride=1, padding=padding,
        count_include_pad=False,
    )


def context_embed(feature_map: torch.Tensor, kernel_size) -> torch.Tensor:
    """Context (if any) then per-patch L2, returning (H*W, C) embeddings.

    Order is FIXED: average-pool (context) FIRST, then per-patch L2
    normalization. Only the pooled representation is emitted (no raw+pooled
    concatenation).
    """
    pooled = average_pool_same(feature_map, kernel_size)
    b, c, h, w = pooled.shape
    flat = pooled.permute(0, 2, 3, 1).reshape(b, h * w, c)[0]
    return F.normalize(flat, p=2, dim=1)


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def spatial_context_gate_passed(s0_auroc: float, cand_auroc: float) -> bool:
    """Stage S gate: candidate AUROC >= 0.65 AND delta vs S0 >= +0.10."""
    if not (_finite(s0_auroc) and _finite(cand_auroc)):
        return False
    return cand_auroc >= SPATIAL_CONTEXT_GATE["auroc_min"] and (cand_auroc - s0_auroc) >= SPATIAL_CONTEXT_GATE["delta_vs_s0"]


def patch_scale_gate_passed(r1_auroc: float, r1_q1_auroc: float,
                            p1_auroc: float, p1_q1_auroc: float) -> bool:
    """Stage P gate: AUROC >= 0.60, delta vs R1 >= +0.10, Q1 delta vs R1 >= +0.10."""
    if not all(_finite(v) for v in (r1_auroc, r1_q1_auroc, p1_auroc, p1_q1_auroc)):
        return False
    return (
        p1_auroc >= PATCH_SCALE_GATE["auroc_min"]
        and (p1_auroc - r1_auroc) >= PATCH_SCALE_GATE["delta_vs_r1"]
        and (p1_q1_auroc - r1_q1_auroc) >= PATCH_SCALE_GATE["q1_delta_vs_r1"]
    )


def select_best_spatial_candidate(results: dict[str, dict]) -> str | None:
    """Deterministic best-candidate selection among gate-passing candidates.

    Highest overall image AUROC wins. If two candidates differ by < 0.01, the
    one with the higher Q1+Q2 mean AUROC wins. Returns None if none pass.
    """
    passing = {cid: r for cid, r in results.items() if r.get("gate_passed")}
    if not passing:
        return None
    ordered = sorted(
        passing,
        key=lambda cid: (-passing[cid]["image_auroc"],
                         -((passing[cid]["q1_auroc"] + passing[cid]["q2_auroc"]) / 2.0),
                         cid),
    )
    best = ordered[0]
    for other in ordered[1:]:
        if abs(passing[best]["image_auroc"] - passing[other]["image_auroc"]) < 0.01:
            best_avg = (passing[best]["q1_auroc"] + passing[best]["q2_auroc"]) / 2.0
            other_avg = (passing[other]["q1_auroc"] + passing[other]["q2_auroc"]) / 2.0
            if other_avg > best_avg:
                best = other
    return best