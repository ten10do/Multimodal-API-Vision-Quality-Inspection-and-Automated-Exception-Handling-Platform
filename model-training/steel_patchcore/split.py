"""Original-image split for Severstal steel (M item).

SPLIT FIRST, TILE SECOND. The split operates on ORIGINAL image ids only.
Classes are derived from the REAL data:
  * ANOMALY: image id present in train.csv (>=1 non-empty defect RLE)
  * NORMAL:  image file present in train_images/ but absent from train.csv

Splits:
  train_normal       (memory bank source ONLY)
  validation_normal
  test_normal
  test_anomaly

Fixed seed, non-overlapping by construction, verified programmatically.
Output: model-training/datasets/severstal-steel/split_manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets/severstal-steel"
RAW = DS / "raw"
CSV = RAW / "train.csv"
IMG_DIR = RAW / "train_images"
OUT = DS / "split_manifest.json"

SPLIT_SEED = 12345
NORMAL_VAL_RATIO = 0.10
NORMAL_TEST_RATIO = 0.10


def main() -> int:
    if not CSV.exists() or not IMG_DIR.exists():
        print("SPLIT_BLOCKED: train.csv or train_images missing")
        return 2

    df = pd.read_csv(CSV)
    # normalize both sides to bare stems (no .jpg suffix)
    anomaly_ids = {str(i)[:-4] if str(i).lower().endswith(".jpg") else str(i)
                   for i in df["ImageId"].unique()}
    physical = {p.stem for p in IMG_DIR.glob("*.jpg")}

    normal_ids = sorted(physical - anomaly_ids)
    anomaly_ids_sorted = sorted(anomaly_ids)

    print(f"physical images      : {len(physical)}")
    print(f"anomaly (in csv)     : {len(anomaly_ids_sorted)}")
    print(f"normal (not in csv)  : {len(normal_ids)}")

    # normal -> train / val / test
    train_n, rest_n = train_test_split(normal_ids, test_size=NORMAL_VAL_RATIO + NORMAL_TEST_RATIO,
                                       random_state=SPLIT_SEED)
    val_n, test_n = train_test_split(rest_n, test_size=NORMAL_TEST_RATIO / (NORMAL_VAL_RATIO + NORMAL_TEST_RATIO),
                                     random_state=SPLIT_SEED)

    splits = {
        "train_normal": sorted(train_n),
        "validation_normal": sorted(val_n),
        "test_normal": sorted(test_n),
        "test_anomaly": anomaly_ids_sorted,
    }

    # non-overlap verification
    sets = {k: set(v) for k, v in splits.items()}
    checks = {}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]] & sets[names[j]]
            checks[f"{names[i]}_n_{names[j]}"] = len(inter)
    all_ok = all(v == 0 for v in checks.values())

    manifest = {
        "dataset": "severstal-steel",
        "split_seed": SPLIT_SEED,
        "definition": {
            "normal": "image present in train_images but absent from train.csv (no defect RLE)",
            "anomaly": "image present in train.csv (>=1 non-empty defect RLE)",
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "non_overlap_checks": checks,
        "all_disjoint": all_ok,
        "splits": splits,
    }
    DS.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, indent=2))
    print("SPLIT_OK" if all_ok else "SPLIT_OVERLAP_FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
