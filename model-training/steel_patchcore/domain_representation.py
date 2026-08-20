"""DINOv2 domain-representation cross-check primitives (Optimization 1.1).

Freezes the D0/D1 candidate identities, the Domain Representation Gate, the
patch-token extraction semantics (CLS/register exclusion), per-patch L2 +
cosine 1-NN distance, and identity serialization. Pure, testable, CPU-only:
no model, no GPU, no holdout.

D0 = frozen S2 (WRN-50-2 layer3 + 5x5 avg context), read from frozen results.
D1 = DINOv2 ViT-S/14 spatial patch tokens (official facebookresearch/dinov2),
     built into a NEW 50k reservoir bank; frozen tiling; A0 aggregation.
"""
from __future__ import annotations

import json
import math
from hashlib import sha256

import numpy as np

DOMAIN_REPRESENTATION_PROTOCOL_VERSION = "domain_representation_protocol_v1"
DOMAIN_REPRESENTATION_SEED = 42
DOMAIN_BANK_BUDGET = 50_000

# D0 = frozen S2 (WRN-50-2 layer3 + 5x5 avg context). Do NOT re-run.
D0_CANDIDATE_ID = "S2"
D0_AUROC = 0.6029
D0_QUARTILES = {"Q1": 0.4790, "Q2": 0.5305, "Q3": 0.6145, "Q4": 0.7876}

# D1 = DINOv2 ViT-S/14 (official facebookresearch torch.hub implementation).
DINO_REFERENCE = {
    "name": "DINOv2",
    "model_identifier": "dinov2_vits14",
    "arch": "vit_small",
    "implementation": "facebookresearch/dinov2 (official torch.hub)",
    "license": "Apache-2.0",
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4,
    "patch_size": 14,
    "img_size": 518,
    "num_register_tokens": 0,
    "num_cls_tokens": 1,
    "pretraining": "self-supervised DINOv2 on LVD-142M",
    "weights_url": "https://dl.fbaipublicfiles.com/dinov2/vit_small14/vit_small14_pretrain.pth",
    "weights_sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    "extraction": "model.forward_features(x)['x_norm_patchtokens'] (final LayerNorm patch tokens; CLS and any register tokens already excluded)",
    "input_normalization": "ImageNet mean/std (0.485,0.456,0.406)/(0.229,0.224,0.225) on [0,1]",
    "input_resize": "256x256 tile bilinearly resized to 252x252 (PatchEmbed requires H,W multiples of patch_size=14; 252=18*14 preserves all tile content)",
}

DOMAIN_REPRESENTATION_GATE = {
    "auroc_min": 0.70,
    "delta_vs_d0": 0.10,
    "strong_auroc": 0.80,
    "quartile_delta": 0.10,
}


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def domain_representation_gate_passed(d0_auroc: float, d1_auroc: float) -> bool:
    """D1 AUROC >= 0.70 AND D1 - D0 >= +0.10."""
    if not (_finite(d0_auroc) and _finite(d1_auroc)):
        return False
    return (
        d1_auroc >= DOMAIN_REPRESENTATION_GATE["auroc_min"]
        and (d1_auroc - d0_auroc) >= DOMAIN_REPRESENTATION_GATE["delta_vs_d0"]
    )


def domain_representation_strong_signal(d1_auroc: float) -> bool:
    return _finite(d1_auroc) and d1_auroc >= DOMAIN_REPRESENTATION_GATE["strong_auroc"]


def small_defect_signal(d1_q1: float, d1_q2: float, d0_q1: float, d0_q2: float) -> bool:
    """Secondary finding: Q1 or Q2 improvement >= +0.10. Never substitutes the gate."""
    if not all(_finite(v) for v in (d1_q1, d1_q2, d0_q1, d0_q2)):
        return False
    q_delta = DOMAIN_REPRESENTATION_GATE["quartile_delta"]
    return (d1_q1 - d0_q1) >= q_delta or (d1_q2 - d0_q2) >= q_delta


def serialize_reference(reference: dict) -> str:
    return json.dumps(reference, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def reference_sha256(reference: dict) -> str:
    return sha256(serialize_reference(reference).encode("utf-8")).hexdigest()


def adapted_input_side(side: int, patch_size: int) -> int:
    """Largest multiple of patch_size <= side (preserves content, never upscales).

    DINOv2's PatchEmbed asserts H,W are exact multiples of patch_size; a frozen
    256px tile is not a multiple of 14, so it is bilinearly resized to 252 (18*14).
    """
    if side < patch_size:
        raise ValueError("input side smaller than patch_size")
    return (side // patch_size) * patch_size


def expected_patch_grid(height: int, width: int, patch_size: int) -> tuple[int, int]:
    """Patch grid for an (already adapted) HxW input that is a multiple of patch_size."""
    if height % patch_size or width % patch_size:
        raise ValueError("height/width must be multiples of patch_size (adapt first)")
    return height // patch_size, width // patch_size


def strip_non_patch_tokens(tokens: np.ndarray, num_cls_tokens: int = 1, num_register_tokens: int = 0) -> np.ndarray:
    """Drop leading CLS (and optional register) tokens; keep spatial patch tokens only."""
    arr = np.asarray(tokens)
    skip = num_cls_tokens + num_register_tokens
    if arr.ndim == 2:
        return arr[skip:]
    if arr.ndim == 3:
        return arr[:, skip:]
    raise ValueError("tokens must be (L, D) or (B, L, D)")


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Per-patch L2 normalization (zero-rows stay zero)."""
    arr = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(arr, axis=axis, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return arr / n


def cosine_1nn_distance(embedding: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """1 - max(cosine similarity) per row: distance = 1 - max(emb @ bank.T)."""
    emb = l2_normalize(embedding)
    bnk = l2_normalize(bank)
    sim = emb @ bnk.T
    return 1.0 - sim.max(axis=-1)