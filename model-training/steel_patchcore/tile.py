"""7-tile preprocessing for Severstal 256x1600 images (N/O items).

Original: 256 (h) x 1600 (w). Tiles are 256x256 with deterministic offsets:
  0, 256, 512, 768, 1024, 1280, 1344
The last two tiles intentionally overlap (1280-1536 and 1344-1600) so the full
width [0, 1600) is covered with zero uncovered pixels. No resize, no right-edge
drop, no silent padding.

Produces tiles_manifest.json (per-tile metadata incl. source_sha256).
Provides mask tile/stitch helpers with pixel-equivalent verification (O item).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_patchcore.rle import rle_decode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets/severstal-steel"
RAW = DS / "raw"
IMG_DIR = RAW / "train_images"
CSV = RAW / "train.csv"
SPLIT = DS / "split_manifest.json"
OUT = DS / "tiles_manifest.json"

IMG_H, IMG_W = 256, 1600
TILE = 256
TILE_X0 = (0, 256, 512, 768, 1024, 1280, 1344)


def tile_coords(tile_id: int) -> tuple[int, int, int, int]:
    """Return (x0, y0, width, height) for tile_id 0..6."""
    assert 0 <= tile_id < len(TILE_X0)
    return (TILE_X0[tile_id], 0, TILE, TILE)


def coverage_check() -> tuple[bool, int]:
    """Verify full width coverage; returns (ok, uncovered_pixels)."""
    covered = np.zeros(IMG_W, dtype=bool)
    for x0 in TILE_X0:
        covered[x0:x0 + TILE] = True
    uncovered = int((~covered).sum())
    return uncovered == 0, uncovered


def tile_mask(mask: np.ndarray, tile_id: int) -> np.ndarray:
    """Crop a binary (256,1600) mask to the given tile using identical coords."""
    x0, y0, w, h = tile_coords(tile_id)
    return mask[y0:y0 + h, x0:x0 + w]


def stitch_binary(masks: list[np.ndarray]) -> np.ndarray:
    """Stitch 7 tile binary masks back to (256,1600); overlaps use OR."""
    full = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for tid, m in enumerate(masks):
        x0, y0, w, h = tile_coords(tid)
        region = full[y0:y0 + h, x0:x0 + w]
        full[y0:y0 + h, x0:x0 + w] = np.maximum(region, m)
    return full


def stitch_scores(maps: list[np.ndarray]) -> np.ndarray:
    """Stitch 7 tile anomaly maps to (256,1600); overlaps use MEAN (S item)."""
    acc = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    cnt = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    for tid, m in enumerate(maps):
        x0, y0, w, h = tile_coords(tid)
        acc[y0:y0 + h, x0:x0 + w] += m
        cnt[y0:y0 + h, x0:x0 + w] += 1.0
    cnt[cnt == 0] = 1.0
    return (acc / cnt).astype(np.float32)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> int:
    ok, uncovered = coverage_check()
    print(f"coverage ok={ok} uncovered_pixels={uncovered}")
    if not ok:
        print("TILING_BLOCKED: width not fully covered")
        return 2

    split_manifest = json.load(open(SPLIT, encoding="utf-8"))
    splits = split_manifest["splits"]

    rows = []
    for split_name, ids in splits.items():
        for img_id in ids:
            src = IMG_DIR / f"{img_id}.jpg"
            if not src.exists():
                print(f"MISSING_IMAGE {src}")
                return 2
            sha = sha256_file(src)
            for tid, x0 in enumerate(TILE_X0):
                rows.append({
                    "original_image_id": img_id,
                    "tile_id": tid,
                    "x0": x0,
                    "y0": 0,
                    "width": TILE,
                    "height": TILE,
                    "split": split_name,
                    "source_sha256": sha,
                })
            if len(rows) % 700 == 0:
                print(f"...{len(rows)} tiles", flush=True)

    manifest = {
        "dataset": "severstal-steel",
        "tiling_version": "v1",
        "original_size": [IMG_H, IMG_W],
        "tile_size": TILE,
        "tile_x0_offsets": list(TILE_X0),
        "full_width_covered": ok,
        "uncovered_pixels": uncovered,
        "tile_count": len(rows),
        "per_split_tile_counts": {k: sum(1 for r in rows if r["split"] == k) for k in splits},
        "overlap_regions": [[1280, 1536], [1344, 1600]],
        "stitch_rules": {"binary_mask_overlap": "OR", "score_map_overlap": "mean"},
        "tiles": rows,
    }
    json.dump(manifest, open(OUT, "w"), indent=1)
    print(f"tiles written: {len(rows)} -> {OUT}")
    print("TILING_OK")
    return 0


def verify_mask_stitch(n_sample: int = 25) -> int:
    """O item: mask tile -> stitch -> original mask pixel equivalence."""
    import pandas as pd

    df = pd.read_csv(CSV)
    samples = df["ImageId"].astype(str).unique()[:n_sample]
    fails = 0
    for img_id in samples:
        rows = df[df["ImageId"].astype(str) == img_id]
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for rle in rows["EncodedPixels"].astype(str):
            mask = np.maximum(mask, rle_decode(rle))
        tiles = [tile_mask(mask, tid) for tid in range(len(TILE_X0))]
        stitched = stitch_binary(tiles)
        if not np.array_equal(mask, stitched):
            print(f"STITCH_MISMATCH {img_id} diff={int((mask != stitched).sum())}")
            fails += 1
    print(f"mask stitch verify: {n_sample} images, fails={fails}")
    if fails == 0:
        print("MASK_STITCH_OK")
        return 0
    print("MASK_STITCH_FAILED")
    return 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "verify":
        sys.exit(verify_mask_stitch())
    sys.exit(build_manifest())
