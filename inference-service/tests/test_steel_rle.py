"""Unit tests for Severstal RLE decode/encode (no data download needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-training"))

from steel_patchcore.rle import (  # noqa: E402
    rle_decode,
    rle_encode,
    rle_is_empty,
    roundtrip_check,
    IMG_HEIGHT,
    IMG_WIDTH,
)


def test_empty_rle_is_normal():
    assert rle_is_empty(None)
    assert rle_is_empty("")
    assert rle_is_empty("  ")
    assert not rle_is_empty("1 2")


def test_empty_decode_shape():
    m = rle_decode("")
    assert m.shape == (IMG_HEIGHT, IMG_WIDTH)
    assert int(m.sum()) == 0


def test_decode_known_sample_column_major():
    # single pixel at column 0 row 0 -> index 1 (1-indexed)
    m = rle_decode("1 1")
    assert m.shape == (256, 1600)
    assert int(m[0, 0]) == 1
    assert int(m.sum()) == 1
    # second column first row -> index 257 (1-indexed) = col 1, row 0
    m2 = rle_decode("257 1")
    assert int(m2[0, 1]) == 1


def test_nonempty_rle_nonempty_mask():
    m = rle_decode("29102 12 29346 24")
    assert int(m.sum()) == 36


def test_roundtrip_equivalence():
    samples = ["29102 12 29346 24", "1 409600", "409600 1", "100 5 200 3 300 7"]
    for s in samples:
        ok, m = roundtrip_check(s)
        assert ok, f"roundtrip failed for {s}"
        assert m.shape == (256, 1600)


def test_out_of_bounds_raises():
    with pytest.raises(ValueError):
        rle_decode("409601 1")
    with pytest.raises(ValueError):
        rle_decode("0 5")


def test_malformed_odd_tokens_raises():
    with pytest.raises(ValueError):
        rle_decode("1 2 3")


def test_encode_empty():
    assert rle_encode(np.zeros((256, 1600), dtype=np.uint8)) == ""


def test_mask_union_of_classes():
    # two class masks OR together must equal the union mask
    m1 = rle_decode("100 50")
    m2 = rle_decode("10000 30")
    union = np.maximum(m1, m2)
    assert int(union.sum()) == 80
