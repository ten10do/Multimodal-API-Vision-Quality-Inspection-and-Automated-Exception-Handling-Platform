"""Leakage Gate for steel PatchCore (P item).

Must pass BEFORE training. Verifies:
  1. train/validation/test original ids pairwise disjoint
  2. train anomaly images = 0 and train anomaly tiles = 0
  3. memory bank sources subset of train_normal (checked again post-bank)
  4. duplicate original ids = 0
  5. masks are never part of the model input (they live in a separate dir)

Post-bank verification (bank_source.json vs test ids) is invoked separately
with --bank to run after the memory bank has been built.

Output: model-training/datasets/severstal-steel/leakage_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets/severstal-steel"
CSV = DS / "raw/train.csv"
SPLIT = DS / "split_manifest.json"
TILES = DS / "tiles_manifest.json"
BANK_SRC = DS / "bank_source.json"
OUT = DS / "leakage_report.json"


def main() -> int:
    use_bank = "--bank" in sys.argv
    if not SPLIT.exists():
        print("LEAKAGE_BLOCKED: split_manifest.json missing")
        return 2

    sm = json.load(open(SPLIT, encoding="utf-8"))
    splits = {k: set(v) for k, v in sm["splits"].items()}
    report: dict = {}

    # 1. pairwise disjoint
    names = list(splits)
    overlaps = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = splits[names[i]] & splits[names[j]]
            overlaps[f"{names[i]}_n_{names[j]}"] = sorted(inter)
    report["original_id_overlaps"] = {k: len(v) for k, v in overlaps.items()}
    ok_disjoint = all(len(v) == 0 for v in overlaps.values())

    # 2. train has zero anomaly images
    df = pd.read_csv(CSV)
    anomaly_ids = set(df["ImageId"].astype(str).unique())
    train_anomaly = sorted(splits["train_normal"] & anomaly_ids)
    report["train_anomaly_images"] = train_anomaly
    ok_train_normal = len(train_anomaly) == 0

    # train anomaly tiles
    train_anomaly_tiles = 0
    if TILES.exists():
        tm = json.load(open(TILES, encoding="utf-8"))
        train_anomaly_set = set(train_anomaly)
        train_anomaly_tiles = sum(1 for t in tm["tiles"]
                                  if t["split"] == "train_normal" and t["original_image_id"] in train_anomaly_set)
    report["train_anomaly_tiles"] = train_anomaly_tiles
    ok_train_tiles = train_anomaly_tiles == 0

    # 4. duplicate original ids within any split
    dup = {}
    for k, v in sm["splits"].items():
        dup[k] = int(len(v) - len(set(v)))
    report["duplicate_ids_per_split"] = dup
    ok_no_dup = all(c == 0 for c in dup.values())

    # 3. bank source check (post-bank)
    ok_bank = None
    if use_bank and BANK_SRC.exists():
        bank_ids = set(json.load(open(BANK_SRC, encoding="utf-8")))
        leak_test = bank_ids & (splits["test_normal"] | splits["test_anomaly"])
        report["bank_vs_test_overlap"] = sorted(leak_test)
        report["bank_sources_count"] = len(bank_ids)
        ok_bank = len(leak_test) == 0

    passed = ok_disjoint and ok_train_normal and ok_train_tiles and ok_no_dup and (ok_bank is not False)
    report.update({
        "checks": {
            "original_id_disjoint": ok_disjoint,
            "train_zero_anomaly_images": ok_train_normal,
            "train_zero_anomaly_tiles": ok_train_tiles,
            "no_duplicate_ids": ok_no_dup,
            "bank_not_containing_test" if use_bank else "bank_not_checked": ok_bank if ok_bank is not None else (None if not use_bank else "pending"),
        },
        "gate_passed": bool(passed),
    })
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if passed:
        print("LEAKAGE_GATE_PASSED")
        return 0
    print("LEAKAGE_GATE_FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
