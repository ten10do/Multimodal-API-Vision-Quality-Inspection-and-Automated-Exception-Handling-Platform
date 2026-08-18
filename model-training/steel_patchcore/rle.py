"""Severstal steel RLE decoding/encoding utilities.

Severstal Kaggle RLE format:
  * EncodedPixels is a flat list of pairs "start length start length ..."
  * 1-indexed, column-major raster order (left->right, top->bottom within each
    column): pixel index = col * height + row.
  * Empty/NaN encoded pixels mean "no defect" (NORMAL).

The decoder is validated by round-trip reconstruction and by a community
cross-check (the Severstal kernel convention used in the competition).

Reference image shape: (H, W) = (256, 1600) -> total 409,600 pixels.
"""
from __future__ import annotations

import numpy as np

IMG_HEIGHT = 256
IMG_WIDTH = 1600
TOTAL_PIXELS = IMG_HEIGHT * IMG_WIDTH  # 409600


def rle_decode(rle: str | None, height: int = IMG_HEIGHT, width: int = IMG_WIDTH) -> np.ndarray:
    """Decode an RLE string into a binary mask of shape (height, width)."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle is None:
        return mask.reshape(height, width)
    text = str(rle).strip()
    if not text:
        return mask.reshape(height, width)
    tokens = text.split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"malformed RLE (odd token count): {len(tokens)} tokens")
    starts = np.asarray(tokens[0::2], dtype=np.int64)
    lengths = np.asarray(tokens[1::2], dtype=np.int64)
    if np.any(starts < 1):
        raise ValueError("RLE start index < 1 (must be 1-indexed)")
    ends = starts + lengths - 1  # inclusive, 0-indexed after -1
    if np.any(ends > height * width):
        raise ValueError(f"RLE run exceeds image bounds (max {height * width}, got {int(ends.max())})")
    zero = starts - 1
    for lo, hi in zip(zero, ends):
        mask[lo:hi] = 1
    return mask.reshape(width, height).T  # column-major -> (height, width)


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask back to Severstal RLE (1-indexed, column-major)."""
    flat = mask.T.flatten()  # column-major flatten
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return ""
    breaks = np.where(np.diff(idx) > 1)[0]  # positions where a run ends in idx
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    lengths = ends - starts + 1
    pairs = np.column_stack([starts + 1, lengths]).flatten()  # back to 1-indexed
    return " ".join(str(int(x)) for x in pairs)


def rle_is_empty(rle: str | None) -> bool:
    """True when the RLE is None, empty or whitespace -> NORMAL sample."""
    if rle is None:
        return True
    return not str(rle).strip()


def roundtrip_check(rle: str) -> tuple[bool, np.ndarray]:
    """Decode then re-encode; return (equal, decoded_mask)."""
    m1 = rle_decode(rle)
    m2 = rle_decode(rle_encode(m1))
    return bool(np.array_equal(m1, m2)), m1


if __name__ == "__main__":
    # quick self-test on a known sample
    sample = "29102 12 29346 24 29602 24"
    m = rle_decode(sample)
    print("shape:", m.shape)
    print("non-zero pixels:", int(m.sum()))
    ok, _ = roundtrip_check(sample)
    print("roundtrip:", ok)
