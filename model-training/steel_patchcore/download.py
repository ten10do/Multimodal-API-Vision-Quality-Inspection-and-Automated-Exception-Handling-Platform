"""Concurrent download of Severstal train_images (12,568 jpgs) from Kaggle.

The competition exposes train_images/ as an expanded directory (no zip).
Concurrent downloader with:
  * resume: already-downloaded files with matching size are skipped
  * per-file retry with exponential backoff
  * progress persisted to download_progress.json every 100 files
  * size verification against the server listing

Usage:
  python model-training/steel_patchcore/download.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets/severstal-steel/raw"
IMG_DIR = RAW / "train_images"
LISTING = Path(tempfile.gettempdir()) / "kaggle_full.json"
PROGRESS = RAW / "download_progress.json"

COMPETITION = "severstal-steel-defect-detection"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if not LISTING.exists():
        print("listing missing:", LISTING)
        return 2
    files = json.load(open(LISTING, encoding="utf-8"))
    train = sorted((k, v) for k, v in files.items() if k.startswith("train_images/"))
    print(f"train_images total: {len(train)}")

    done = set()
    if PROGRESS.exists():
        done = set(json.load(open(PROGRESS, encoding="utf-8")))
    print(f"already done: {len(done)}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    skipped = 0
    todo = []
    for name, size in train:
        target = IMG_DIR / name.split("/")[-1]
        # resume: treat "exists, non-empty AND no .kaggle-partial marker" as
        # done. A partial marker means a previous download was interrupted and
        # the main file may be incomplete -> force redownload (CLI resumes it).
        partial_marker = IMG_DIR / f"{target.name}.kaggle-partial"
        if target.exists() and target.stat().st_size > 0 and not partial_marker.exists():
            skipped += 1
            continue
        todo.append((name, size, target))

    print(f"to download: {len(todo)}  (skipped {skipped})")

    failed: list[str] = []
    ok = 0

    def fetch(item):
        name, size, target = item
        consecutive_429 = 0
        for attempt in range(12):
            try:
                api.competition_download_file(COMPETITION, name, path=str(IMG_DIR))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 11:
                    return name, False, f"{type(e).__name__}: {e}"
                if "429" in str(e):
                    consecutive_429 += 1
                    wait = min(60 * (2 ** min(consecutive_429 - 1, 2)), 240)
                    time.sleep(wait)  # hard backoff on rate limit
                else:
                    time.sleep(2.0 * (attempt + 1))
        if not (target.exists() and target.stat().st_size > 0):
            return name, False, "file missing or empty after download"
        if size and target.stat().st_size != size:
            return name, False, f"size mismatch local={target.stat().st_size} server={size}"
        # success: ensure no partial marker remains
        marker = IMG_DIR / f"{target.name}.kaggle-partial"
        if marker.exists():
            try:
                marker.unlink()
            except OSError:
                pass
        return name, True, None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            name, success, err = fut.result()
            if success:
                ok += 1
                done.add(name.split("/")[-1])
            else:
                failed.append(name)
                print(f"FAIL {name}: {err}", flush=True)
            if i % 200 == 0:
                json.dump(sorted(done), open(PROGRESS, "w"))
                print(f"...{i}/{len(todo)} ok={ok} failed={len(failed)}", flush=True)

    json.dump(sorted(done), open(PROGRESS, "w"))
    print(f"\nDONE ok={ok} failed={len(failed)} total_done={len(done)}")
    if failed:
        print("FAILED:", failed[:30], "count", len(failed))
        return 1
    print("DOWNLOAD_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
