"""Download MVTec AD 'bottle' via HuggingFace parquet files.

The official MVTec mydrive links have been unreliable since late 2025, so we
fetch the `TheoM55/mvtec_all_objects_split` parquet shards directly (much
faster than the datasets library's resolution) and materialize the standard
MVTec layout: train/good, test/<defect>, ground_truth/<defect>.

Usage:
  python scripts/fetch_mvtec_bottle.py
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model-training/datasets/mvtec" / "bottle"
BASE = "https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split/resolve/main/data"


def _download(url: str, dest: Path) -> None:
    print("downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done/1024/1024:.1f}/{total/1024/1024:.1f} MB", end="\r")
    print()


def _save_images(df: pd.DataFrame, split: str, base: Path) -> None:
    counts: dict[str, int] = {}
    n_masks = 0
    for _, row in df.iterrows():
        defect = row["defect"] if "defect" in df.columns else "good"
        idx = counts.get(defect, 0) + 1
        counts[defect] = idx
        img = Image.open(io.BytesIO(row["image_path"]["bytes"])) if isinstance(row["image_path"], dict) else row["image_path"]
        if isinstance(img, Image.Image) is False:
            img = Image.open(io.BytesIO(img))
        out_dir = base / split / defect
        out_dir.mkdir(parents=True, exist_ok=True)
        img.save(out_dir / f"{idx:03d}.png")
        label = row.get("label", 0)
        mask = row.get("mask_path")
        if label == 1 and mask is not None and isinstance(mask, dict) and mask.get("bytes"):
            mask_dir = base / "ground_truth" / defect
            mask_dir.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(mask["bytes"])).convert("L").save(mask_dir / f"{idx:03d}_mask.png")
            n_masks += 1
    print(f"{split}: {counts}, masks={n_masks}")


def main() -> None:
    tmp = ROOT / "model-training/datasets/mvtec" / "_parquet"
    tmp.mkdir(parents=True, exist_ok=True)
    train_parquet = tmp / "bottle.train.parquet"
    test_parquet = tmp / "bottle.test.parquet"
    if not train_parquet.exists():
        _download(f"{BASE}/bottle.train-00000-of-00001.parquet", train_parquet)
    if not test_parquet.exists():
        _download(f"{BASE}/bottle.test-00000-of-00001.parquet", test_parquet)

    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)
    print("train rows:", len(train_df), "| test rows:", len(test_df), "| cols:", list(test_df.columns))
    _save_images(train_df, "train", OUT)
    _save_images(test_df, "test", OUT)
    print("output:", OUT)


if __name__ == "__main__":
    sys.exit(main())
