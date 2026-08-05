"""Train the PatchCore memory bank on MVTec AD 'bottle' normal images.

No backprop: features are extracted once and a normal-distribution threshold is
fixed so the training set has zero false positives (image-level score max).

Usage:
  python inference-service/scripts/train_patchcore.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_app.patchcore_predictor import PatchCorePredictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "model-training/datasets/mvtec/bottle/train/good"
OUT = ROOT / "inference-service/models/patchcore-bottle"
BANK_PATCHES = 50_000  # random subsample of the full normal feature set
SEED = 42


def main() -> None:
    predictor = PatchCorePredictor()
    images = sorted(DATA.glob("*.png"))
    if not images:
        sys.exit(f"no normal images found at {DATA}")
    print(f"training on {len(images)} normal images")

    all_features: list[np.ndarray] = []
    for p in images:
        feats = predictor._embed(Image.open(p))
        all_features.append(feats)
    full = np.concatenate(all_features, axis=0).astype(np.float32)
    print(f"full bank: {full.shape} patches x {full.shape[1]} dims")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(full.shape[0], size=min(BANK_PATCHES, full.shape[0]), replace=False)
    bank = full[idx]

    # threshold: max image-level score over the training set (zero train FP)
    image_scores: list[float] = []
    for p in images:
        predictor._bank = bank
        predictor._threshold = 0.0
        _, score = predictor.score(Image.open(p))
        image_scores.append(score)
    threshold = float(max(image_scores))
    print(f"train image-level scores: min={min(image_scores):.4f} max={max(image_scores):.4f}")
    print(f"threshold (train max) = {threshold:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "bank.npz",
        features=bank,
        threshold=threshold,
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        train_images=len(images),
        bank_patches=bank.shape[0],
    )
    print("saved:", OUT / "bank.npz")


if __name__ == "__main__":
    main()
