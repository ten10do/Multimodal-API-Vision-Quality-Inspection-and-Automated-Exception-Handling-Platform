"""Extract the 8 chunk ZIPs into raw/train_images (overwrite same-name)."""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "datasets/severstal-steel/kernel-pack/chunk_out"
IMG_DIR = ROOT / "datasets/severstal-steel/raw/train_images"

PARTS = [f"part-{i:03d}.zip" for i in range(8)]


def main() -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = 0
    for name in PARTS:
        p = CHUNK_DIR / name
        if not p.exists():
            print("MISSING", name)
            return 2
        with zipfile.ZipFile(p) as z:
            for info in z.infolist():
                fname = Path(info.filename).name
                target = IMG_DIR / fname
                with z.open(info) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                total += 1
        print(f"{name}: extracted", flush=True)
    count = len(list(IMG_DIR.glob("*.jpg")))
    print(f"extracted {total} files, train_images jpg count: {count}, "
          f"elapsed {time.time()-t0:.0f}s", flush=True)
    if count != 12568:
        print("EXTRACT_COUNT_MISMATCH")
        return 1
    print("EXTRACT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
