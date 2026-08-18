"""RLE Validation Gate for Severstal data (K item).

Validates against the REAL downloaded train.csv:
  * schema (ImageId, ClassId, EncodedPixels)
  * every non-empty RLE decodes to a (256, 1600) mask
  * non-empty RLE -> non-empty mask
  * pixel indices within image bounds
  * round-trip: decode -> encode -> decode equivalence
  * cross-check with a community reference decoder (column-major assumption)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_patchcore.rle import rle_decode, rle_encode, rle_is_empty  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "datasets/severstal-steel/raw"
CSV = RAW / "train.csv"


def reference_decode(rle: str, height: int = 256, width: int = 1600) -> np.ndarray:
    """Community reference decoder (Kaggle Severstal kernel convention)."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle is None or not str(rle).strip():
        return mask.reshape(height, width)
    s = str(rle).split()
    starts, lengths = (np.asarray(s[0::2], dtype=int), np.asarray(s[1::2], dtype=int))
    starts -= 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(width, height).T


def main() -> int:
    if not CSV.exists():
        print("RLE_VALIDATION_BLOCKED: train.csv missing")
        return 2
    df = pd.read_csv(CSV)
    print(f"csv rows (incl header): {len(df)}")
    print(f"columns: {list(df.columns)}")
    assert list(df.columns) == ["ImageId", "ClassId", "EncodedPixels"], "unexpected schema"

    empty = df["EncodedPixels"].isna() | df["EncodedPixels"].astype(str).str.strip().eq("")
    print(f"empty-RLE rows: {int(empty.sum())}  non-empty: {int((~empty).sum())}")

    nonempty = df.loc[~empty]
    failures = 0
    cross_mismatch = 0
    roundtrip_fail = 0
    max_end = 0
    for i, row in nonempty.iterrows():
        rle = str(row["EncodedPixels"]).strip()
        try:
            m = rle_decode(rle)
        except Exception as e:  # noqa: BLE001
            print(f"DECODE_FAIL row={i} image={row['ImageId']} class={row['ClassId']}: {e}")
            failures += 1
            continue
        if m.shape != (256, 1600):
            print(f"SHAPE_FAIL row={i}: {m.shape}")
            failures += 1
            continue
        if int(m.sum()) == 0:
            print(f"EMPTY_MASK row={i} image={row['ImageId']}")
            failures += 1
            continue
        # cross-check with reference decoder
        ref = reference_decode(rle)
        if not np.array_equal(m, ref):
            cross_mismatch += 1
        # round-trip
        rt = rle_decode(rle_encode(m))
        if not np.array_equal(m, rt):
            roundtrip_fail += 1
        # bounds: decoded non-zero must lie inside
        ys, xs = np.nonzero(m)
        if ys.size and (ys.max() > 255 or xs.max() > 1599):
            print(f"BOUNDS_FAIL row={i}")
            failures += 1

    # verify first sample coordinates manually against known convention
    first = nonempty.iloc[0]
    m0 = rle_decode(str(first["EncodedPixels"]))
    ys0, xs0 = np.nonzero(m0)
    print(f"sample {first['ImageId']} class {first['ClassId']} nonzero={int(m0.sum())} "
          f"min_idx={int((m0.T.flatten()).argmax())} rows_range=({ys0.min()},{ys0.max()})")

    print(f"\nSUMMARY: decoded={len(nonempty)} failures={failures} "
          f"cross_mismatch={cross_mismatch} roundtrip_fail={roundtrip_fail}")
    if failures == 0 and cross_mismatch == 0 and roundtrip_fail == 0:
        print("RLE_VALIDATION_OK")
        return 0
    print("RLE_VALIDATION_BLOCKED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
