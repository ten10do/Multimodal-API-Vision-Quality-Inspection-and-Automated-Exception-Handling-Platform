"""Build the frozen representation diagnostic subset manifest (no GPU).

Deterministic subset of authorized development data: 1000 train_normal,
300 validation_normal, 1000 recovery_dev_anomaly (250 per defect-area quartile).
Outputs a small JSON plus its SHA256; eligible for Git (no tensors/features).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))

from steel_patchcore.recovery import sha256_file, validate_recovery_split_manifest  # noqa: E402
from steel_patchcore.representation import build_representation_subset_manifest  # noqa: E402
from steel_patchcore.rle import rle_decode  # noqa: E402

DS = ROOT / "model-training/datasets/severstal-steel"
SOURCE_SPLIT = DS / "split_manifest.json"
RECOVERY_SPLIT = DS / "recovery_split_manifest.json"
CSV = DS / "raw/train.csv"
OUT = DS / "representation_diagnostic_manifest.json"
OUT_SHA = DS / "representation_diagnostic_manifest.sha256"

EXPECTED_SOURCE_SPLIT_SHA = "64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07"
EXPECTED_RECOVERY_SPLIT_SHA = "f60ce0a6aa012ad1d9a9636800c1e94f365940500f20b8deb2f7fd9ffbc55448"


def main() -> int:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    validate_recovery_split_manifest(recovery, source, EXPECTED_SOURCE_SPLIT_SHA)
    source_sha = sha256_file(SOURCE_SPLIT)
    recovery_sha = sha256_file(RECOVERY_SPLIT)
    if source_sha != EXPECTED_SOURCE_SPLIT_SHA or recovery_sha != EXPECTED_RECOVERY_SPLIT_SHA:
        raise RuntimeError("FROZEN_SPLIT_SHA_MISMATCH")

    # area ratios for development anomalies only (from frozen GT RLE)
    dev_ids = list(recovery["recovery_dev_anomaly"])
    wanted = set(dev_ids)
    df = pd.read_csv(CSV, keep_default_na=True)
    area_ratios: dict[str, float] = {}
    for img_id, group in df.groupby(df["ImageId"].astype(str)):
        norm = Path(str(img_id)).stem
        if norm not in wanted:
            continue
        mask = np.zeros((256, 1600), dtype=np.uint8)
        for value in group["EncodedPixels"]:
            if pd.isna(value):
                continue
            mask = np.maximum(mask, rle_decode(str(value)))
        area_ratios[norm] = float(mask.sum()) / (256 * 1600)
    missing = wanted - set(area_ratios)
    if missing:
        raise RuntimeError(f"MISSING_ANOMALY_MASK:{sorted(missing)[:5]}")

    manifest = build_representation_subset_manifest(
        source_splits=source["splits"],
        recovery_dev_anomaly=dev_ids,
        recovery_holdout_anomaly=list(recovery["recovery_holdout_anomaly"]),
        test_normal=list(source["splits"]["test_normal"]),
        area_ratios=area_ratios,
        source_split_sha256=source_sha,
        recovery_split_sha256=recovery_sha,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_sha = sha256_file(OUT)
    OUT_SHA.write_text(manifest_sha + "\n", encoding="ascii")
    print(json.dumps({
        "train_normal": len(manifest["train_normal_subset"]),
        "validation_normal": len(manifest["validation_normal_subset"]),
        "dev_anomaly": len(manifest["recovery_dev_anomaly_subset"]),
        "quartile_counts": manifest["anomaly_quartile_counts"],
        "quartile_boundaries": manifest["anomaly_quartile_boundaries"],
        "manifest_sha256": manifest_sha,
    }))
    print("REPRESENTATION_MANIFEST_WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())