"""Heatmap-only recovery candidates for the frozen D3 patch distances.

All functions operate after the immutable A0 image score has been calculated.
They cannot alter the model, bank, threshold, or returned image score.
"""
from __future__ import annotations

from typing import Final

import numpy as np
from PIL import Image

from steel_patchcore.tile import IMG_H, IMG_W, TILE, TILE_X0, stitch_scores

CANDIDATE_SPECS: Final[dict[str, str]] = {
    "H0": "current bilinear/mean map followed by per-image min-max normalization",
    "H1": "raw 18x18 D3 patch distances projected with nearest-neighbor and mean stitching",
    "H2": "whitened-feature cosine-1NN patch distances projected with bilinear and mean stitching",
    "H3": "3x3 local mean on each 18x18 distance grid, then bilinear/mean stitching",
    "H4": "5x5 Gaussian sigma=1 on each 18x18 distance grid, then bilinear/mean stitching",
    "H5": "bilinear patch maps with linear feathering in the 192-pixel tile overlap",
}


class HeatmapRecoveryError(RuntimeError):
    """Invalid heatmap input or image-score isolation violation."""


def frozen_a0_score(tile_grids: np.ndarray) -> float:
    grids = _validated_grids(tile_grids)
    return float(grids.max())


def assert_image_score_unchanged(before: float, after: float) -> None:
    if np.asarray(before, dtype=np.float32).tobytes() != np.asarray(after, dtype=np.float32).tobytes():
        raise HeatmapRecoveryError(f"D3_A0_IMAGE_SCORE_CHANGED:{before!r}:{after!r}")


def localization_gate(
    pixel_auroc: float,
    aupro: float,
    baseline_pixel_auroc: float,
    baseline_aupro: float,
) -> tuple[bool, dict[str, float]]:
    minimums = {
        "pixel_auroc_min": float(baseline_pixel_auroc) * 0.95,
        "aupro_min": float(baseline_aupro) * 0.95,
    }
    passed = float(pixel_auroc) >= minimums["pixel_auroc_min"] and float(aupro) >= minimums["aupro_min"]
    return passed, minimums


def _validated_grids(tile_grids: np.ndarray) -> np.ndarray:
    grids = np.asarray(tile_grids, dtype=np.float32)
    if grids.shape != (len(TILE_X0), 18, 18):
        raise HeatmapRecoveryError(f"PATCH_GRID_SHAPE_MISMATCH:{grids.shape}")
    if not np.isfinite(grids).all():
        raise HeatmapRecoveryError("PATCH_GRID_NONFINITE")
    return grids


def _resize(grid: np.ndarray, *, nearest: bool) -> np.ndarray:
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(grid, mode="F").resize((TILE, TILE), resampling), dtype=np.float32)


def _normalize(anomaly_map: np.ndarray) -> np.ndarray:
    minimum, maximum = float(anomaly_map.min()), float(anomaly_map.max())
    if maximum <= minimum:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return ((anomaly_map - minimum) / (maximum - minimum)).astype(np.float32, copy=False)


def _filter_grid(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    weights = np.asarray(kernel, dtype=np.float64)
    weights /= weights.sum()
    radius_y, radius_x = weights.shape[0] // 2, weights.shape[1] // 2
    padded = np.pad(grid.astype(np.float64), ((radius_y, radius_y), (radius_x, radius_x)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, weights.shape)
    return np.einsum("ijkl,kl->ij", windows, weights, optimize=True).astype(np.float32)


def local_mean_3x3(grid: np.ndarray) -> np.ndarray:
    return _filter_grid(grid, np.ones((3, 3), dtype=np.float64))


def gaussian_sigma1(grid: np.ndarray) -> np.ndarray:
    axis = np.arange(-2, 3, dtype=np.float64)
    one_dimensional = np.exp(-(axis**2) / 2.0)
    return _filter_grid(grid, np.outer(one_dimensional, one_dimensional))


def weighted_stitch(tile_maps: list[np.ndarray]) -> np.ndarray:
    if len(tile_maps) != len(TILE_X0) or any(np.asarray(tile).shape != (TILE, TILE) for tile in tile_maps):
        raise HeatmapRecoveryError("WEIGHTED_STITCH_INPUT_MISMATCH")
    acc = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    weights = np.zeros((IMG_H, IMG_W), dtype=np.float64)
    tile_weights = [np.ones(TILE, dtype=np.float64) for _ in TILE_X0]
    for left_index in range(len(TILE_X0) - 1):
        left_start, right_start = TILE_X0[left_index], TILE_X0[left_index + 1]
        overlap = left_start + TILE - right_start
        if overlap <= 0:
            continue
        ramp = np.linspace(1.0, 0.0, overlap + 2, dtype=np.float64)[1:-1]
        tile_weights[left_index][TILE - overlap:] = ramp
        tile_weights[left_index + 1][:overlap] = 1.0 - ramp
    for tile_map, x0, horizontal in zip(tile_maps, TILE_X0, tile_weights):
        weight_map = np.broadcast_to(horizontal, (TILE, TILE))
        acc[:, x0:x0 + TILE] += np.asarray(tile_map, dtype=np.float64) * weight_map
        weights[:, x0:x0 + TILE] += weight_map
    if np.any(weights == 0):
        raise HeatmapRecoveryError("WEIGHTED_STITCH_UNCOVERED_PIXEL")
    return (acc / weights).astype(np.float32)


def generate_heatmap_candidates(tile_grids: np.ndarray) -> dict[str, np.ndarray]:
    grids = _validated_grids(tile_grids)
    bilinear_maps = [_resize(grid, nearest=False) for grid in grids]
    current = stitch_scores(bilinear_maps)
    candidates = {
        "H0": _normalize(current),
        "H1": stitch_scores([_resize(grid, nearest=True) for grid in grids]),
        "H2": current,
        "H3": stitch_scores([_resize(local_mean_3x3(grid), nearest=False) for grid in grids]),
        "H4": stitch_scores([_resize(gaussian_sigma1(grid), nearest=False) for grid in grids]),
        "H5": weighted_stitch(bilinear_maps),
    }
    for name, anomaly_map in candidates.items():
        if anomaly_map.shape != (IMG_H, IMG_W) or not np.isfinite(anomaly_map).all():
            raise HeatmapRecoveryError(f"CANDIDATE_MAP_INVALID:{name}")
        anomaly_map.flags.writeable = False
    return candidates


__all__ = [
    "CANDIDATE_SPECS",
    "HeatmapRecoveryError",
    "assert_image_score_unchanged",
    "frozen_a0_score",
    "gaussian_sigma1",
    "generate_heatmap_candidates",
    "localization_gate",
    "local_mean_3x3",
    "weighted_stitch",
]
