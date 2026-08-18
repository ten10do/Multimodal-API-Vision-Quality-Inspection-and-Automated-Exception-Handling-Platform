"""Actual dataset audit for Severstal steel (J item).

All counts come from the REAL downloaded files, not from any prior report.
Audits:
  * CSV schema / rows
  * unique image ids, duplicate ids
  * per-class annotation counts, multi-class images
  * malformed RLE (odd tokens, out-of-bounds runs)
  * empty RLE -> NORMAL, non-empty -> ANOMALY
  * physical image file existence / size / dimensions (when images are present)

Output: model-training/datasets/severstal-steel/audit.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_patchcore.rle import rle_decode, rle_is_empty  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets/severstal-steel"
RAW = DS / "raw"
CSV = RAW / "train.csv"
IMG_DIR = RAW / "train_images"
OUT = DS / "audit.json"


def main() -> int:
    if not CSV.exists():
        print("AUDIT_BLOCKED: train.csv missing")
        return 2

    df = pd.read_csv(CSV)
    report: dict = {}
    report["csv"] = {
        "path": str(CSV),
        "rows_incl_header": int(len(df) + 1),
        "columns": list(df.columns),
    }

    # schema
    if list(df.columns) != ["ImageId", "ClassId", "EncodedPixels"]:
        report["schema_ok"] = False
    else:
        report["schema_ok"] = True

    # empty / non-empty
    empty_mask = df["EncodedPixels"].isna() | df["EncodedPixels"].astype(str).str.strip().eq("")
    report["annotation_rows_nonempty"] = int((~empty_mask).sum())
    report["annotation_rows_empty"] = int(empty_mask.sum())

    # unique images
    all_ids = df["ImageId"].astype(str)
    unique_ids = all_ids.unique()
    report["unique_image_ids_in_csv"] = int(len(unique_ids))

    # duplicate (same image+class listed twice)
    dup = df.duplicated(subset=["ImageId", "ClassId"]).sum()
    report["duplicate_image_class_pairs"] = int(dup)

    # per class counts (non-empty annotations)
    nonempty = df.loc[~empty_mask]
    cls_counts = Counter(nonempty["ClassId"])
    report["per_class_annotation_count"] = {str(k): int(v) for k, v in sorted(cls_counts.items())}

    # multi-class images
    img_classes = nonempty.groupby("ImageId")["ClassId"].nunique()
    report["multi_class_image_count"] = int((img_classes > 1).sum())
    report["single_class_image_count"] = int((img_classes == 1).sum())
    report["defect_image_count_unique"] = int(img_classes.shape[0])

    # RLE integrity
    malformed = 0
    out_of_bounds = 0
    bad_mask = 0
    for rle in nonempty["EncodedPixels"].astype(str):
        try:
            m = rle_decode(rle)
            if m.shape != (256, 1600) or int(m.sum()) == 0:
                bad_mask += 1
        except ValueError as e:
            if "exceeds image bounds" in str(e):
                out_of_bounds += 1
            else:
                malformed += 1
    report["rle_malformed"] = malformed
    report["rle_out_of_bounds"] = out_of_bounds
    report["rle_empty_mask_from_nonempty"] = bad_mask

    # image-level normal/anomaly (final, computed from PHYSICAL files)
    # 6666 is strictly "annotated/anomalous unique images", NOT the full count.
    report["annotated_anomaly_unique_images"] = report["defect_image_count_unique"]
    report["anomaly_images"] = report["defect_image_count_unique"]

    # physical files
    if IMG_DIR.exists():
        phys = list(IMG_DIR.glob("*.jpg"))
        phys_ids = {p.name for p in phys}  # includes .jpg suffix, matches CSV ImageId
        report["physical_image_files"] = len(phys)
        report["csv_ids_missing_on_disk"] = len(set(unique_ids) - phys_ids)
        report["orphan_images_not_in_csv"] = len(phys_ids - set(unique_ids))
        # full validation when --full: every jpg must decode with expected dims
        full = "--full" in sys.argv
        valid_ids = set()
        corrupted = 0
        dims: Counter = Counter()
        sizes_total = 0
        check_list = phys if full else phys[:200]
        for i, p in enumerate(check_list):
            try:
                with Image.open(p) as im:
                    im.load()
                    dims[im.size] += 1
                    sizes_total += p.stat().st_size
                    if im.size == (1600, 256):
                        valid_ids.add(p.stem)
            except Exception:  # noqa: BLE001
                corrupted += 1
        report["sample_image_dimensions"] = {f"{w}x{h}": int(c) for (w, h), c in dims.items()}
        report["corrupted_images"] = corrupted if full else None
        report["physical_valid_images"] = len(valid_ids) if full else None
        report["sample_mean_bytes"] = round(sizes_total / max(len(check_list), 1), 1)
        if full:
            # normal = physical valid train images - unique anomaly (never hardcoded)
            report["normal_images"] = len(valid_ids) - report["defect_image_count_unique"]
            report["full_audit_ok"] = len(valid_ids) == 12568 and corrupted == 0
    else:
        report["physical_image_files"] = "images not downloaded yet"

    DS.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nAUDIT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
