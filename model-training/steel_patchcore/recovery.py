"""Deterministic primitives for Steel PatchCore recovery evidence."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


PROTOCOL_VERSION = "recovery_protocol_v1"
RECOVERY_SEED = 42
DEV_ANOMALY_COUNT = 3333
HOLDOUT_ANOMALY_COUNT = 3333
CAPTURE_ROLES = ("train_normal", "validation_normal", "recovery_dev_anomaly")
HOLDOUT_ROLES = ("test_normal", "recovery_holdout_anomaly")

CANDIDATE_GRID = (
    {"id": "A0", "method": "baseline_global_max"},
    {"id": "A1", "method": "percentile", "percentile": 99.0},
    {"id": "A2", "method": "percentile", "percentile": 99.5},
    {"id": "A3", "method": "percentile", "percentile": 99.9},
    {"id": "A4", "method": "top_percentage_mean", "fraction": 0.001},
    {"id": "A5", "method": "top_percentage_mean", "fraction": 0.005},
    {"id": "A6", "method": "top_percentage_mean", "fraction": 0.01},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_recovery_split_manifest(
    source_manifest: dict,
    source_split_sha256: str,
    *,
    created_at: str,
    seed: int = RECOVERY_SEED,
) -> dict:
    anomaly_ids = list(source_manifest["splits"]["test_anomaly"])
    if len(anomaly_ids) != DEV_ANOMALY_COUNT + HOLDOUT_ANOMALY_COUNT:
        raise ValueError(f"expected 6666 test anomalies, got {len(anomaly_ids)}")
    if len(set(anomaly_ids)) != len(anomaly_ids):
        raise ValueError("source test_anomaly contains duplicate IDs")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(anomaly_ids, dtype=object)[rng.permutation(len(anomaly_ids))].tolist()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "created_at": created_at,
        "source_split_sha256": source_split_sha256,
        "source_test_anomaly_count": len(anomaly_ids),
        "post_hoc_notice": (
            "This is a post-hoc recovery holdout because the complete Severstal "
            "baseline test was observed during Optimization 1; it is not a pristine test."
        ),
        "recovery_dev_anomaly": shuffled[:DEV_ANOMALY_COUNT],
        "recovery_holdout_anomaly": shuffled[DEV_ANOMALY_COUNT:],
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    validate_recovery_split_manifest(payload, source_manifest, source_split_sha256)
    return payload


def validate_recovery_split_manifest(
    recovery_manifest: dict,
    source_manifest: dict,
    source_split_sha256: str,
) -> None:
    if recovery_manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("recovery protocol version mismatch")
    if recovery_manifest.get("seed") != RECOVERY_SEED:
        raise ValueError("recovery seed mismatch")
    if recovery_manifest.get("source_split_sha256") != source_split_sha256:
        raise ValueError("recovery source split SHA256 mismatch")

    dev = list(recovery_manifest.get("recovery_dev_anomaly", []))
    holdout = list(recovery_manifest.get("recovery_holdout_anomaly", []))
    source = list(source_manifest["splits"]["test_anomaly"])
    if len(dev) != DEV_ANOMALY_COUNT or len(holdout) != HOLDOUT_ANOMALY_COUNT:
        raise ValueError("recovery anomaly split count mismatch")
    if len(set(dev)) != len(dev) or len(set(holdout)) != len(holdout):
        raise ValueError("duplicate recovery IDs")
    if set(dev) & set(holdout):
        raise ValueError("recovery dev/holdout overlap")
    if set(dev) | set(holdout) != set(source):
        raise ValueError("recovery dev/holdout union mismatch")

    expected_payload_sha = recovery_manifest.get("manifest_payload_sha256")
    payload = dict(recovery_manifest)
    payload.pop("manifest_payload_sha256", None)
    if expected_payload_sha != canonical_sha256(payload):
        raise ValueError("recovery manifest payload SHA256 mismatch")


def raw_distance_grid_from_embeddings(
    embeddings: np.ndarray,
    bank: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Reproduce the frozen predictor's raw cosine 1-NN distance grid."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)
    if embeddings.ndim != 2 or bank.ndim != 2 or embeddings.shape[1] != bank.shape[1]:
        raise ValueError("embedding/bank shape mismatch")
    if embeddings.shape[0] != grid_shape[0] * grid_shape[1]:
        raise ValueError("embedding count does not match grid shape")
    similarity = embeddings @ bank.T
    distance = 1.0 - np.max(similarity, axis=1)
    grid = distance.reshape(grid_shape).astype(np.float32, copy=False)
    if not np.isfinite(grid).all():
        raise ValueError("raw distance grid contains non-finite values")
    return grid


def infer_square_grid_shape(patch_count: int) -> tuple[int, int]:
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError(f"patch count {patch_count} is not a square grid")
    return side, side


def baseline_score(raw_tile_grids: np.ndarray) -> float:
    grids = np.asarray(raw_tile_grids, dtype=np.float32)
    if grids.ndim != 3 or not np.isfinite(grids).all():
        raise ValueError("raw tile grids must be finite [tiles,h,w]")
    return float(np.max(grids))


def stitch_raw_patch_grids(
    raw_tile_grids: np.ndarray,
    tile_x_offsets: tuple[int, ...] | list[int],
    *,
    tile_size: int,
    original_width: int,
) -> tuple[np.ndarray, int]:
    """Mean-stitch raw tile grids without per-tile normalization."""
    grids = np.asarray(raw_tile_grids, dtype=np.float32)
    if grids.ndim != 3 or grids.shape[0] != len(tile_x_offsets):
        raise ValueError("tile grid/offset count mismatch")
    if not np.isfinite(grids).all():
        raise ValueError("raw tile grids contain non-finite values")
    grid_h, grid_w = grids.shape[1:]
    if tile_size % grid_h or tile_size % grid_w:
        raise ValueError("tile size is not divisible by raw grid geometry")
    stride_y = tile_size // grid_h
    stride_x = tile_size // grid_w
    if stride_x != stride_y:
        raise ValueError("non-square raw patch stride")
    if original_width % stride_x:
        raise ValueError("original width is not divisible by raw patch stride")
    if any(offset % stride_x for offset in tile_x_offsets):
        raise ValueError("tile offset is not aligned to raw patch stride")

    global_width = original_width // stride_x
    accum = np.zeros((grid_h, global_width), dtype=np.float64)
    counts = np.zeros((grid_h, global_width), dtype=np.float64)
    for grid, offset in zip(grids, tile_x_offsets):
        x0 = offset // stride_x
        x1 = x0 + grid_w
        if x1 > global_width:
            raise ValueError("raw tile grid exceeds global geometry")
        accum[:, x0:x1] += grid
        counts[:, x0:x1] += 1.0
    if np.any(counts == 0):
        raise ValueError("raw stitched patch grid has uncovered cells")
    return (accum / counts).astype(np.float32), stride_x


def normalize_raw_grid_to_predictor_map(raw_grid: np.ndarray, image_size: int) -> np.ndarray:
    """Reproduce the predictor's per-tile min-max/uint8/resize map path."""
    grid = np.asarray(raw_grid, dtype=np.float32)
    scaled = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8) * 255.0
    resized = np.asarray(
        Image.fromarray(scaled.astype(np.uint8)).resize(
            (image_size, image_size), Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    minimum, maximum = float(resized.min()), float(resized.max())
    if maximum > minimum:
        resized = (resized - minimum) / (maximum - minimum)
    return resized.astype(np.float32, copy=False)


def candidate_score(
    candidate_id: str,
    raw_tile_grids: np.ndarray,
    stitched_raw_grid: np.ndarray,
) -> float:
    """Apply one frozen A0-A6 aggregation candidate deterministically."""
    config = next((item for item in CANDIDATE_GRID if item["id"] == candidate_id), None)
    if config is None:
        raise ValueError(f"unknown recovery candidate {candidate_id}")
    if candidate_id == "A0":
        return baseline_score(raw_tile_grids)

    values = np.asarray(stitched_raw_grid, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("stitched raw grid must be finite and non-empty")
    if config["method"] == "percentile":
        return float(np.percentile(values, config["percentile"], method="linear"))
    if config["method"] == "top_percentage_mean":
        count = max(1, math.ceil(float(config["fraction"]) * values.size))
        return float(np.partition(values, values.size - count)[-count:].mean(dtype=np.float64))
    raise ValueError(f"unsupported recovery method {config['method']}")
