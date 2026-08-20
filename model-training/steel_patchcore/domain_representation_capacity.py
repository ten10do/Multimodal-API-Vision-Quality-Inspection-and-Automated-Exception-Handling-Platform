"""DINOv2 capacity cross-check primitives (Optimization 1.1).

Freezes the D0/D1 references and the D2 candidate identity (DINOv2 ViT-B/14),
the unchanged Primary Domain Representation Gate, plus the capacity-specific
secondary metric (D2 - D1 >= +0.03). Pure, testable, CPU-only: no model, no GPU,
no holdout.

D0 = frozen S2 (WRN-50-2 layer3 + 5x5 avg context), read from frozen results.
D1 = frozen DINOv2 ViT-S/14 (GATE_FAILED, small-defect secondary signal TRUE).
D2 = DINOv2 ViT-B/14 spatial patch tokens (official facebookresearch/dinov2),
     built into a NEW 50k reservoir bank; frozen tiling; A0 aggregation.
"""
from __future__ import annotations

import math

from steel_patchcore.domain_representation import (  # noqa: E402
    D0_AUROC,
    D0_QUARTILES,
    reference_sha256,
    serialize_reference,
)

CAPACITY_PROTOCOL_VERSION = "domain_representation_capacity_protocol_v1"
CAPACITY_SEED = 42
CAPACITY_BANK_BUDGET = 50_000

# D1 = frozen DINOv2 ViT-S/14 (previous phase). Do NOT re-run.
D1_CANDIDATE_ID = "D1"
D1_AUROC = 0.6699
D1_QUARTILES = {"Q1": 0.5843, "Q2": 0.6086, "Q3": 0.6607, "Q4": 0.8261}
D1_SMALL_DEFECT_SIGNAL = True

# D2 = DINOv2 ViT-B/14 (official facebookresearch torch.hub implementation).
D2_REFERENCE = {
    "name": "DINOv2",
    "model_identifier": "dinov2_vitb14",
    "arch": "vit_base",
    "implementation": "facebookresearch/dinov2 (official torch.hub)",
    "license": "Apache-2.0",
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4,
    "patch_size": 14,
    "img_size": 518,
    "num_register_tokens": 0,
    "num_cls_tokens": 1,
    "pretraining": "self-supervised DINOv2 on LVD-142M",
    "weights_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "weights_sha256": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    "extraction": "model.forward_features(x)['x_norm_patchtokens'] (final LayerNorm patch tokens; CLS and any register tokens already excluded)",
    "input_normalization": "ImageNet mean/std (0.485,0.456,0.406)/(0.229,0.224,0.225) on [0,1]",
    "input_resize": "256x256 tile bilinearly resized to 252x252 (PatchEmbed requires H,W multiples of patch_size=14; 252=18*14 preserves all tile content)",
}

# Primary Gate unchanged from D1 (no moving goalposts). Capacity-delta is secondary.
CAPACITY_GATE = {
    "auroc_min": 0.70,
    "delta_vs_d0": 0.10,
    "strong_auroc": 0.80,
    "capacity_delta_vs_d1": 0.03,
}


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def capacity_gate_passed(d0_auroc: float, d2_auroc: float) -> bool:
    """D2 AUROC >= 0.70 AND D2 - D0 >= +0.10 (Primary Gate, unchanged)."""
    if not (_finite(d0_auroc) and _finite(d2_auroc)):
        return False
    return (
        d2_auroc >= CAPACITY_GATE["auroc_min"]
        and (d2_auroc - d0_auroc) >= CAPACITY_GATE["delta_vs_d0"]
    )


def capacity_gain(d2_auroc: float, d1_auroc: float) -> bool:
    """Secondary finding: D2 - D1 >= +0.03. Never substitutes the Primary Gate."""
    if not (_finite(d2_auroc) and _finite(d1_auroc)):
        return False
    return (d2_auroc - d1_auroc) >= CAPACITY_GATE["capacity_delta_vs_d1"]


def capacity_strong_signal(d2_auroc: float) -> bool:
    return _finite(d2_auroc) and d2_auroc >= CAPACITY_GATE["strong_auroc"]


def d2_reference_sha256() -> str:
    return reference_sha256(D2_REFERENCE)


__all__ = [
    "CAPACITY_BANK_BUDGET",
    "CAPACITY_GATE",
    "CAPACITY_PROTOCOL_VERSION",
    "CAPACITY_SEED",
    "D0_AUROC",
    "D0_QUARTILES",
    "D1_AUROC",
    "D1_CANDIDATE_ID",
    "D1_QUARTILES",
    "D1_SMALL_DEFECT_SIGNAL",
    "D2_REFERENCE",
    "capacity_gain",
    "capacity_gate_passed",
    "capacity_strong_signal",
    "d2_reference_sha256",
    "serialize_reference",
]