"""Unit tests for 7-tile coverage and mask/score stitch equivalence."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-training"))

from steel_patchcore.tile import (  # noqa: E402
    TILE,
    TILE_X0,
    tile_coords,
    coverage_check,
    tile_mask,
    stitch_binary,
    stitch_scores,
)


def test_seven_tiles_with_coverage():
    assert len(TILE_X0) == 7
    assert TILE_X0 == (0, 256, 512, 768, 1024, 1280, 1344)


def test_full_width_coverage_no_drop():
    ok, uncovered = coverage_check()
    assert ok
    assert uncovered == 0
    # last tile must reach width 1600
    x0, _, w, _ = tile_coords(6)
    assert x0 + w == 1600


def test_no_resize_no_silent_padding():
    for tid in range(7):
        x0, y0, w, h = tile_coords(tid)
        assert w == TILE and h == TILE
        assert x0 + w <= 1600
        assert y0 + h <= 256


def test_overlap_region_definition():
    # tiles 5 and 6 overlap in [1344, 1536)
    x5, *_ = tile_coords(5)
    x6, *_ = tile_coords(6)
    assert x5 == 1280 and x6 == 1344
    assert x6 < x5 + TILE  # overlap exists


def test_binary_stitch_pixel_equivalent():
    rng = np.random.default_rng(7)
    mask = (rng.random((256, 1600)) > 0.995).astype(np.uint8)
    tiles = [tile_mask(mask, tid) for tid in range(7)]
    stitched = stitch_binary(tiles)
    assert np.array_equal(mask, stitched)


def test_score_stitch_mean_in_overlap():
    rng = np.random.default_rng(11)
    base = rng.random((256, 1600)).astype(np.float32)
    tiles = [tile_mask(base, tid).astype(np.float32) for tid in range(7)]
    stitched = stitch_scores(tiles)
    assert stitched.shape == (256, 1600)
    # single-coverage regions must equal the source
    assert np.allclose(stitched[:, 100:200], base[:, 100:200], atol=1e-6)
    # overlap region [1344, 1536) is the mean of tiles 5 and 6
    exp = (base[:, 1344:1536] * 2) / 2.0
    assert np.allclose(stitched[:, 1344:1536], exp, atol=1e-6)
