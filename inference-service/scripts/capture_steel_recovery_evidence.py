"""Checkpointable dev-only raw evidence capture for Steel PatchCore recovery."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference-service"))
sys.path.insert(0, str(ROOT / "model-training"))

from inference_app.patchcore_predictor import PatchCorePredictor  # noqa: E402
from steel_patchcore.lifecycle import lifecycle_enter  # noqa: E402
from steel_patchcore.recovery import (  # noqa: E402
    CANDIDATE_GRID,
    CAPTURE_ROLES,
    PROTOCOL_VERSION,
    RECOVERY_SEED,
    baseline_score,
    build_recovery_split_manifest,
    infer_square_grid_shape,
    raw_distance_grid_from_embeddings,
    sha256_file,
    stitch_raw_patch_grids,
    validate_recovery_split_manifest,
)
from steel_patchcore.tile import IMG_W, TILE, TILE_X0, tile_coords  # noqa: E402


DS = ROOT / "model-training/datasets/severstal-steel"
SOURCE_SPLIT = DS / "split_manifest.json"
RECOVERY_SPLIT = DS / "recovery_split_manifest.json"
RECOVERY_SPLIT_SHA = DS / "recovery_split_manifest.sha256"
TRAIN_SCORES = DS / "train_normal_scores.json"
BASELINE_CHECKPOINT = DS / "raw/steel_eval_ckpt.json"
THRESHOLD = DS / "threshold.json"
IMG_DIR = DS / "raw/train_images"
BANK = ROOT / "inference-service/models/steel-patchcore/bank.npz"
BANK_META = ROOT / "inference-service/models/steel-patchcore/bank_meta.json"
PROTOCOL_DOC = ROOT / "docs/steel-patchcore-recovery-protocol.md"

EVIDENCE_ROOT = DS / "raw/recovery-evidence"
CAPTURE_CHECKPOINT = EVIDENCE_ROOT / "capture_checkpoint.json"
EVIDENCE_MANIFEST = DS / "recovery_evidence_manifest.json"
EVIDENCE_MANIFEST_SHA = DS / "recovery_evidence_manifest.sha256"

EXPECTED_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"
EXPECTED_SOURCE_SPLIT_SHA = "64df9f817a995e64c7ee4d00e16962d94f500878fd1404d95227f26e7f913d07"
EXPECTED_THRESHOLD = 0.490039
SHARD_SIZE = 100
RECONSTRUCTION_ATOL = 2e-6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def verify_frozen_artifacts() -> dict[str, str | float]:
    bank_sha = sha256_file(BANK)
    source_sha = sha256_file(SOURCE_SPLIT)
    threshold = json.loads(THRESHOLD.read_text(encoding="utf-8"))
    if bank_sha != EXPECTED_BANK_SHA:
        raise RuntimeError(f"FROZEN_BANK_SHA_MISMATCH:{bank_sha}")
    if source_sha != EXPECTED_SOURCE_SPLIT_SHA:
        raise RuntimeError(f"FROZEN_SOURCE_SPLIT_SHA_MISMATCH:{source_sha}")
    if float(threshold["threshold"]) != EXPECTED_THRESHOLD:
        raise RuntimeError(f"FROZEN_THRESHOLD_MISMATCH:{threshold['threshold']}")
    if threshold.get("bank_sha256") != bank_sha:
        raise RuntimeError("FROZEN_THRESHOLD_BANK_BINDING_MISMATCH")
    if threshold.get("source_split_manifest_sha256") != source_sha:
        raise RuntimeError("FROZEN_THRESHOLD_SPLIT_BINDING_MISMATCH")
    return {"bank_sha256": bank_sha, "source_split_sha256": source_sha, "threshold": EXPECTED_THRESHOLD}


def prepare_recovery_split(source_sha: str) -> tuple[dict, str]:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    if RECOVERY_SPLIT.exists():
        recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
        validate_recovery_split_manifest(recovery, source, source_sha)
    else:
        recovery = build_recovery_split_manifest(
            source, source_sha, created_at=utc_now(), seed=RECOVERY_SEED
        )
        atomic_write_json(RECOVERY_SPLIT, recovery)
    recovery_sha = sha256_file(RECOVERY_SPLIT)
    RECOVERY_SPLIT_SHA.write_text(recovery_sha + "\n", encoding="ascii")
    return recovery, recovery_sha


def capture_ids(source: dict, recovery: dict) -> dict[str, list[str]]:
    roles = {
        "train_normal": list(source["splits"]["train_normal"]),
        "validation_normal": list(source["splits"]["validation_normal"]),
        "recovery_dev_anomaly": list(recovery["recovery_dev_anomaly"]),
    }
    expected = {"train_normal": 4721, "validation_normal": 590, "recovery_dev_anomaly": 3333}
    if {name: len(ids) for name, ids in roles.items()} != expected:
        raise RuntimeError("CAPTURE_ROLE_COUNT_MISMATCH")
    flattened = [image_id for ids in roles.values() for image_id in ids]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("CAPTURE_ROLE_DUPLICATE_IDS")
    holdout = set(source["splits"]["test_normal"]) | set(recovery["recovery_holdout_anomaly"])
    if set(flattened) & holdout:
        raise RuntimeError("CAPTURE_ROLE_HOLDOUT_CONTAMINATION")
    return roles


def expected_baseline_scores(recovery: dict) -> dict[str, dict[str, float]]:
    train = json.loads(TRAIN_SCORES.read_text(encoding="utf-8"))
    train_scores = {
        image_id: float(score)
        for image_id, score in zip(train["image_ids"], train["original_image_scores"])
    }
    baseline = json.loads(BASELINE_CHECKPOINT.read_text(encoding="utf-8"))
    validation_scores = {
        image_id: float(row["score"])
        for image_id, row in baseline["validation_normal"].items()
    }
    anomaly_scores = {
        image_id: float(baseline["test_anomaly"][image_id]["score"])
        for image_id in recovery["recovery_dev_anomaly"]
    }
    return {
        "train_normal": train_scores,
        "validation_normal": validation_scores,
        "recovery_dev_anomaly": anomaly_scores,
    }


def load_tiles(image_id: str) -> list[Image.Image]:
    with Image.open(IMG_DIR / f"{image_id}.jpg") as source:
        image = source.convert("RGB") if source.mode != "RGB" else source.copy()
    return [
        image.crop((x0, y0, x0 + width, y0 + height))
        for x0, y0, width, height in (tile_coords(index) for index in range(len(TILE_X0)))
    ]


def capture_raw_grids(
    predictor: PatchCorePredictor, image_id: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int, tuple[int, int]]:
    grids = []
    grid_shape: tuple[int, int] | None = None
    for tile in load_tiles(image_id):
        embeddings = predictor._embed(tile)
        actual_shape = infer_square_grid_shape(embeddings.shape[0])
        if grid_shape is None:
            grid_shape = actual_shape
        elif actual_shape != grid_shape:
            raise RuntimeError("RAW_GRID_SHAPE_CHANGED_WITHIN_ORIGINAL")
        grids.append(raw_distance_grid_from_embeddings(embeddings, predictor._bank, actual_shape))
    raw = np.stack(grids).astype(np.float32, copy=False)
    stitched, stride = stitch_raw_patch_grids(
        raw, TILE_X0, tile_size=TILE, original_width=IMG_W
    )
    return raw, raw.max(axis=(1, 2)), grid_shape, stride, stitched.shape


def save_shard(
    role: str,
    shard_index: int,
    image_ids: list[str],
    raw_grids: np.ndarray,
    tile_scores: np.ndarray,
    baseline_scores: np.ndarray,
    patch_stride: int,
    stitched_shape: tuple[int, int],
) -> dict:
    role_dir = EVIDENCE_ROOT / role
    role_dir.mkdir(parents=True, exist_ok=True)
    path = role_dir / f"shard-{shard_index:03d}.npz"
    temporary = role_dir / f".shard-{shard_index:03d}.{os.getpid()}.tmp.npz"
    source_split = "test_anomaly" if role == "recovery_dev_anomaly" else role
    np.savez(
        temporary,
        original_ids=np.asarray(image_ids),
        recovery_role=np.asarray(role),
        source_split=np.asarray(source_split),
        tile_indices=np.arange(len(TILE_X0), dtype=np.int16),
        tile_x_offsets=np.asarray(TILE_X0, dtype=np.int32),
        raw_grids=np.asarray(raw_grids, dtype=np.float32),
        raw_tile_scores=np.asarray(tile_scores, dtype=np.float32),
        baseline_scores=np.asarray(baseline_scores, dtype=np.float32),
        patch_stride=np.asarray(patch_stride, dtype=np.int16),
        stitched_shape=np.asarray(stitched_shape, dtype=np.int32),
    )
    os.replace(temporary, path)
    return {
        "role": role,
        "path": str(path.relative_to(DS)).replace("\\", "/"),
        "count": len(image_ids),
        "first_id": image_ids[0],
        "last_id": image_ids[-1],
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_shards(
    roles: dict[str, list[str]],
    expected_scores: dict[str, dict[str, float]],
) -> tuple[dict[str, set[str]], list[dict], tuple[int, int] | None, int | None, tuple[int, int] | None, float]:
    completed = {role: set() for role in CAPTURE_ROLES}
    artifacts: list[dict] = []
    common_grid_shape = None
    common_stride = None
    common_stitched_shape = None
    max_error = 0.0

    for role in CAPTURE_ROLES:
        expected_ids = set(roles[role])
        role_dir = EVIDENCE_ROOT / role
        if not role_dir.exists():
            continue
        for path in sorted(role_dir.glob("shard-*.npz")):
            with np.load(path, allow_pickle=False) as data:
                ids = [str(value) for value in data["original_ids"].tolist()]
                grids = data["raw_grids"]
                tile_scores = data["raw_tile_scores"]
                scores = data["baseline_scores"]
                offsets = tuple(int(value) for value in data["tile_x_offsets"].tolist())
                role_value = str(data["recovery_role"])
                stride = int(data["patch_stride"])
                stitched_shape = tuple(int(value) for value in data["stitched_shape"].tolist())
            if role_value != role or offsets != TILE_X0:
                raise RuntimeError(f"SHARD_METADATA_MISMATCH:{path}")
            if grids.dtype != np.float32 or grids.ndim != 4 or grids.shape[1] != len(TILE_X0):
                raise RuntimeError(f"SHARD_GRID_SCHEMA_MISMATCH:{path}")
            if len(ids) != grids.shape[0] or tile_scores.shape != (len(ids), len(TILE_X0)):
                raise RuntimeError(f"SHARD_COUNT_MISMATCH:{path}")
            if not np.isfinite(grids).all() or not np.isfinite(tile_scores).all():
                raise RuntimeError(f"SHARD_NONFINITE:{path}")
            if not np.allclose(tile_scores, grids.max(axis=(2, 3)), rtol=0.0, atol=0.0):
                raise RuntimeError(f"SHARD_TILE_SCORE_MISMATCH:{path}")
            if len(ids) != len(set(ids)) or completed[role] & set(ids):
                raise RuntimeError(f"SHARD_DUPLICATE_IDS:{path}")
            if not set(ids) <= expected_ids:
                raise RuntimeError(f"SHARD_UNEXPECTED_ID:{path}")

            grid_shape = tuple(int(value) for value in grids.shape[2:])
            if common_grid_shape not in (None, grid_shape):
                raise RuntimeError("SHARD_GLOBAL_GRID_SHAPE_MISMATCH")
            if common_stride not in (None, stride) or common_stitched_shape not in (None, stitched_shape):
                raise RuntimeError("SHARD_GLOBAL_GEOMETRY_MISMATCH")
            common_grid_shape = grid_shape
            common_stride = stride
            common_stitched_shape = stitched_shape

            for image_id, actual_score in zip(ids, scores):
                error = abs(float(actual_score) - expected_scores[role][image_id])
                max_error = max(max_error, error)
                if error > RECONSTRUCTION_ATOL:
                    raise RuntimeError(f"RAW_EVIDENCE_RECONSTRUCTION_FAILED:{image_id}:{error}")
            completed[role].update(ids)
            artifacts.append({
                "role": role,
                "path": str(path.relative_to(DS)).replace("\\", "/"),
                "count": len(ids),
                "first_id": ids[0],
                "last_id": ids[-1],
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return completed, artifacts, common_grid_shape, common_stride, common_stitched_shape, max_error


def save_checkpoint(
    *,
    started_at: str,
    current_role: str,
    completed: dict[str, set[str]],
    artifacts: list[dict],
    max_error: float,
    recovery_split_sha: str,
    extractor_commit: str,
    protocol_sha: str,
) -> None:
    atomic_write_json(CAPTURE_CHECKPOINT, {
        "protocol_version": PROTOCOL_VERSION,
        "capture_started_at": started_at,
        "last_updated": utc_now(),
        "current_split": current_role,
        "completed_ids": {role: sorted(ids) for role, ids in completed.items()},
        "completed_counts": {role: len(ids) for role, ids in completed.items()},
        "completed_shards": artifacts,
        "artifact_hashes": {item["path"]: item["sha256"] for item in artifacts},
        "bank_sha256": EXPECTED_BANK_SHA,
        "source_split_sha256": EXPECTED_SOURCE_SPLIT_SHA,
        "recovery_split_sha256": recovery_split_sha,
        "extractor_commit": extractor_commit,
        "protocol_document_sha256": protocol_sha,
        "baseline_reconstruction_max_abs_error": max_error,
        "holdout_inference_count": 0,
    })


def write_evidence_manifest(
    *,
    started_at: str,
    completed_at: str,
    completed: dict[str, set[str]],
    artifacts: list[dict],
    grid_shape: tuple[int, int],
    patch_stride: int,
    stitched_shape: tuple[int, int],
    max_error: float,
    recovery_split_sha: str,
    extractor_commit: str,
    protocol_sha: str,
) -> str:
    bank_meta = json.loads(BANK_META.read_text(encoding="utf-8"))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "RECOVERY_EVIDENCE_READY",
        "model": "steel-patchcore",
        "model_version": "1.0.0",
        "bank_sha256": EXPECTED_BANK_SHA,
        "source_split_sha256": EXPECTED_SOURCE_SPLIT_SHA,
        "recovery_split_sha256": recovery_split_sha,
        "extractor_commit": extractor_commit,
        "protocol_document_sha256": protocol_sha,
        "backbone": "wide_resnet50_2 / IMAGENET1K_V1",
        "layers": ["layer2", "layer3"],
        "feature_dim": int(bank_meta["feature_dim"]),
        "memory_bank_size": int(bank_meta["row_count"]),
        "tiling_offsets": list(TILE_X0),
        "patch_grid_shape": list(grid_shape),
        "patch_stride": patch_stride,
        "raw_stitched_patch_grid_shape": list(stitched_shape),
        "raw_stitched_overlap": "mean; deterministically reconstructed from canonical tile grids",
        "raw_dtype": "float32",
        "shard_size_originals": SHARD_SIZE,
        "train_normal_count": len(completed["train_normal"]),
        "validation_normal_count": len(completed["validation_normal"]),
        "dev_anomaly_count": len(completed["recovery_dev_anomaly"]),
        "holdout_inference_count": 0,
        "candidate_grid_frozen_not_evaluated": list(CANDIDATE_GRID),
        "baseline_reconstruction_atol": RECONSTRUCTION_ATOL,
        "baseline_reconstruction_max_abs_error": max_error,
        "artifact_shards": artifacts,
        "artifact_total_size_bytes": sum(item["size_bytes"] for item in artifacts),
        "capture_started_at": started_at,
        "capture_completed_at": completed_at,
    }
    atomic_write_json(EVIDENCE_MANIFEST, payload)
    manifest_sha = sha256_file(EVIDENCE_MANIFEST)
    EVIDENCE_MANIFEST_SHA.write_text(manifest_sha + "\n", encoding="ascii")
    return manifest_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    frozen = verify_frozen_artifacts()
    if not PROTOCOL_DOC.exists():
        raise RuntimeError("RECOVERY_PROTOCOL_NOT_FROZEN")
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery, recovery_sha = prepare_recovery_split(str(frozen["source_split_sha256"]))
    roles = capture_ids(source, recovery)
    score_evidence = expected_baseline_scores(recovery)
    extractor_commit = git_head()
    protocol_sha = sha256_file(PROTOCOL_DOC)
    print("frozen artifacts:", frozen, flush=True)
    print("recovery_split_sha256:", recovery_sha, flush=True)
    print("capture roles:", {name: len(ids) for name, ids in roles.items()}, flush=True)
    print("holdout inference count: 0", flush=True)
    if args.prepare_only:
        print("RECOVERY_CAPTURE_PREPARED")
        return 0

    existing_checkpoint = (
        json.loads(CAPTURE_CHECKPOINT.read_text(encoding="utf-8"))
        if CAPTURE_CHECKPOINT.exists()
        else {}
    )
    if existing_checkpoint and existing_checkpoint.get("extractor_commit") != extractor_commit:
        raise RuntimeError("CAPTURE_EXTRACTOR_COMMIT_MISMATCH")
    if existing_checkpoint and existing_checkpoint.get("protocol_document_sha256") != protocol_sha:
        raise RuntimeError("CAPTURE_PROTOCOL_SHA_MISMATCH")
    started_at = existing_checkpoint.get("capture_started_at", utc_now())
    completed, artifacts, grid_shape, stride, stitched_shape, max_error = verify_shards(
        roles, score_evidence
    )
    print("verified existing counts:", {name: len(ids) for name, ids in completed.items()}, flush=True)
    if args.verify_only:
        print("RECOVERY_CAPTURE_VERIFIED")
        return 0

    lifecycle_result = lifecycle_enter(
        "recovery_evidence_capture", EXPECTED_BANK_SHA, CAPTURE_CHECKPOINT, EVIDENCE_ROOT
    )
    if lifecycle_result is not None:
        return lifecycle_result

    predictor = PatchCorePredictor(BANK, image_size=256)
    predictor._ensure_model()
    print("device:", predictor.device, "model:", predictor.model_name, predictor.model_version, flush=True)
    wall_started = time.perf_counter()

    for role in CAPTURE_ROLES:
        pending = [image_id for image_id in roles[role] if image_id not in completed[role]]
        existing_indices = [
            int(Path(item["path"]).stem.split("-")[-1])
            for item in artifacts
            if item["role"] == role
        ]
        next_shard = max(existing_indices, default=-1) + 1
        print(f"{role}: done={len(completed[role])} pending={len(pending)}", flush=True)
        for start in range(0, len(pending), SHARD_SIZE):
            shard_ids = pending[start:start + SHARD_SIZE]
            shard_grids = []
            shard_tile_scores = []
            shard_baseline_scores = []
            for image_id in shard_ids:
                raw, tile_scores, actual_grid_shape, actual_stride, actual_stitched_shape = capture_raw_grids(
                    predictor, image_id
                )
                reconstructed = baseline_score(raw)
                error = abs(reconstructed - score_evidence[role][image_id])
                max_error = max(max_error, error)
                if error > RECONSTRUCTION_ATOL:
                    raise RuntimeError(
                        f"RAW_EVIDENCE_RECONSTRUCTION_FAILED:{image_id}:"
                        f"expected={score_evidence[role][image_id]} actual={reconstructed} error={error}"
                    )
                if grid_shape not in (None, actual_grid_shape):
                    raise RuntimeError("CAPTURE_GRID_SHAPE_MISMATCH")
                if stride not in (None, actual_stride) or stitched_shape not in (None, actual_stitched_shape):
                    raise RuntimeError("CAPTURE_GEOMETRY_MISMATCH")
                grid_shape = actual_grid_shape
                stride = actual_stride
                stitched_shape = actual_stitched_shape
                shard_grids.append(raw)
                shard_tile_scores.append(tile_scores)
                shard_baseline_scores.append(reconstructed)

            artifact = save_shard(
                role,
                next_shard,
                shard_ids,
                np.stack(shard_grids),
                np.stack(shard_tile_scores),
                np.asarray(shard_baseline_scores, dtype=np.float32),
                stride,
                stitched_shape,
            )
            artifacts.append(artifact)
            completed[role].update(shard_ids)
            save_checkpoint(
                started_at=started_at,
                current_role=role,
                completed=completed,
                artifacts=artifacts,
                max_error=max_error,
                recovery_split_sha=recovery_sha,
                extractor_commit=extractor_commit,
                protocol_sha=protocol_sha,
            )
            next_shard += 1
            elapsed = time.perf_counter() - wall_started
            print(
                f"{role}: {len(completed[role])}/{len(roles[role])} "
                f"shard={artifact['path']} sha={artifact['sha256'][:12]} elapsed={elapsed:.0f}s",
                flush=True,
            )

    expected_counts = {"train_normal": 4721, "validation_normal": 590, "recovery_dev_anomaly": 3333}
    actual_counts = {role: len(ids) for role, ids in completed.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(f"RECOVERY_CAPTURE_COUNT_MISMATCH:{actual_counts}")
    if grid_shape is None or stride is None or stitched_shape is None:
        raise RuntimeError("RECOVERY_CAPTURE_GEOMETRY_MISSING")
    if sha256_file(BANK) != EXPECTED_BANK_SHA:
        raise RuntimeError("FROZEN_BANK_CHANGED_DURING_CAPTURE")

    completed_at = utc_now()
    manifest_sha = write_evidence_manifest(
        started_at=started_at,
        completed_at=completed_at,
        completed=completed,
        artifacts=artifacts,
        grid_shape=grid_shape,
        patch_stride=stride,
        stitched_shape=stitched_shape,
        max_error=max_error,
        recovery_split_sha=recovery_sha,
        extractor_commit=extractor_commit,
        protocol_sha=protocol_sha,
    )
    save_checkpoint(
        started_at=started_at,
        current_role="complete",
        completed=completed,
        artifacts=artifacts,
        max_error=max_error,
        recovery_split_sha=recovery_sha,
        extractor_commit=extractor_commit,
        protocol_sha=protocol_sha,
    )
    print("evidence_manifest_sha256:", manifest_sha)
    print("baseline_reconstruction_max_abs_error:", max_error)
    print("RECOVERY_EVIDENCE_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
