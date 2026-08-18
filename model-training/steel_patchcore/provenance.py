"""Dataset provenance for Severstal steel (L item).

Records verifiable facts only (no invented license terms, no tokens).
Aggregates: raw archive hashes, audited counts, annotation schema,
preprocessing/tiling/split versions and manifest SHAs.

Output: model-training/datasets/severstal-steel/provenance.json
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets/severstal-steel"
RAW = DS / "raw"
CSV_ZIP = RAW / "train.csv.zip"
CSV = RAW / "train.csv"
IMG_DIR = RAW / "train_images"
AUDIT = DS / "audit.json"
SPLIT = DS / "split_manifest.json"
TILES = DS / "tiles_manifest.json"
OUT = DS / "provenance.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    audit = json.load(open(AUDIT, encoding="utf-8")) if AUDIT.exists() else {}
    split = json.load(open(SPLIT, encoding="utf-8")) if SPLIT.exists() else {}
    tiles = json.load(open(TILES, encoding="utf-8")) if TILES.exists() else {}

    raw_hashes = {}
    if CSV_ZIP.exists():
        raw_hashes["train.csv.zip"] = sha256_file(CSV_ZIP)
    if CSV.exists():
        raw_hashes["train.csv"] = sha256_file(CSV)

    img_count = len(list(IMG_DIR.glob("*.jpg"))) if IMG_DIR.exists() else 0
    img_total_bytes = sum(p.stat().st_size for p in IMG_DIR.glob("*.jpg")) if IMG_DIR.exists() else 0

    provenance = {
        "dataset_name": "Severstal Steel Defect Detection (train subset)",
        "source": "Kaggle competition 'severstal-steel-defect-detection' (Severstal PAO, 2019)",
        "competition": "https://www.kaggle.com/competitions/severstal-steel-defect-detection",
        "access_method": "official Kaggle CLI/API with user-provided API token (rules accepted)",
        "download_date": datetime.date.today().isoformat(),
        "license": "owned by PAO Severstal; distributed under Kaggle competition terms; no standard OSS license claimed",
        "raw_file_hashes_sha256": raw_hashes,
        "raw_train_images": {
            "file_count": img_count,
            "total_bytes": img_total_bytes,
        },
        "audited_counts": audit,
        "annotation_schema": ["ImageId", "ClassId", "EncodedPixels"],
        "rle_encoding": "1-indexed column-major runs; empty/None means no defect",
        "preprocessing": {
            "version": "v1",
            "original_size": [256, 1600],
            "tile_size": 256,
            "tile_x0_offsets": list(tiles.get("tile_x0_offsets", [])),
            "tiling_version": tiles.get("tiling_version", "n/a"),
        },
        "split_seed": split.get("split_seed", None),
        "split_counts": split.get("counts", {}),
        "manifest_sha256": {
            "audit.json": sha256_file(AUDIT) if AUDIT.exists() else None,
            "split_manifest.json": sha256_file(SPLIT) if SPLIT.exists() else None,
            "tiles_manifest.json": sha256_file(TILES) if TILES.exists() else None,
        },
        "notes": [
            "normal == image present in train_images but absent from train.csv (no defect RLE)",
            "anomaly == image present in train.csv with >=1 non-empty defect RLE",
            "test split of the original competition has no public ground truth and is not used",
        ],
    }
    json.dump(provenance, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in provenance.items() if k not in ("audited_counts",)}, indent=2, ensure_ascii=False))
    print("PROVENANCE_WRITTEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
